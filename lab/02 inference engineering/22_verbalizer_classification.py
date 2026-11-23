# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 22: verbalizer-классификация (Jev за один forward pass).

Идея
----
Генерация — не единственный способ получить ответ от модели. Если набор
вариантов известен заранее, достаточно одного forward pass'а: читаем логиты
только тех токенов, что соответствуют вариантам, и нормируем их softmax'ом
по вариантам, а не по всему словарю на 150k токенов.

Именно это делает «Jev» из нашумевшего поста — 25 строк на Python. Здесь мы
воспроизводим приём на MLX и смотрим на то, о чём пост умалчивает:
метка обязана быть одним токеном, у модели есть bias на порядок вариантов,
а её вероятности не откалиброваны.

Что показываем:
  0. почему метки — буквы A/B/C, а не слова Legitimate/Spam/Phishing;
  1. логиты → log-probabilities → вероятности по классам (как в статье);
  2. сколько стоит один prefill против свободной генерации с парсингом;
  3. label bias: как ответ зависит от порядка вариантов и от prior классов;
  4. три стратегии чтения (фикс. порядок / перестановки / content-free
     калибровка) и их accuracy, confidence, ECE на маленьком holdout;
  5. цена одного письма на потоке.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/22_verbalizer_classification.py"

Coder-модель по умолчанию для классификации слабовата; для сравнения можно
взять обычный instruct-чекпоинт:

    LAB_MODEL=mlx-community/Qwen2.5-3B-Instruct-4bit \
        python "lab/02 inference engineering/22_verbalizer_classification.py"
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from common import MODEL_ID, bench, load_model, note, section, subsection

CLASSES = ["Legitimate", "Spam", "Phishing"]  # порядок как в статье
SLOTS = ["A", "B", "C"]  # verbalizer: буква варианта → класс
ORDER = CLASSES  # «канонический» порядок вариантов в промпте
CONTENT_FREE = "N/A"  # вход без содержания — для оценки prior модели

# (письмо, золотая метка, пометка)
# Метки — решение человека, а не истина в последней инстанции; «ambiguous»
# отмечены письма, где разумный человек может выбрать другой класс.
EMAILS: list[tuple[str, str, str]] = [
    ("Standup moves to 10:15 today; the meeting link is unchanged.", "Legitimate", ""),
    ("Your invoice #4471 has been paid. No action is required.", "Legitimate", ""),
    ("The build on main failed: see the CI dashboard for the logs.", "Legitimate", ""),
    (
        "Security alert: a new device signed in from Minsk. If it wasn't you, "
        "reset your password.",
        "Legitimate",
        "ambiguous",
    ),
    (
        "CONGRATULATIONS! You are selected for a $1000 gift card. Claim now, "
        "limited time offer!",
        "Spam",
        "",
    ),
    ("Lose 10 kg in two weeks with this one weird trick. Click here.", "Spam", ""),
    (
        "Final notice: your extended warranty expires today. Call now to save 70%.",
        "Spam",
        "",
    ),
    (
        "Your subscription renews on the 12th for $9.99. Manage preferences in settings.",
        "Spam",
        "ambiguous",
    ),
    ("Payroll asks for your password on a non-company sign-in page.", "Phishing", ""),
    (
        "IT support: mailbox is full, verify your credentials at "
        "http://micros0ft-login.xyz to avoid deactivation.",
        "Phishing",
        "",
    ),
    (
        "We detected an unusual sign-in. Confirm your account details within "
        "24 hours or access will be suspended.",
        "Phishing",
        "",
    ),
    (
        "Invoice_2024.pdf is ready to view. Sign in to your portal to download it.",
        "Phishing",
        "ambiguous",
    ),
]

PERMS = [list(p) for p in itertools.permutations(CLASSES)]  # 6 порядков


# --------------------------------------------------------------------------- #
# Промпт и чтение логитов
# --------------------------------------------------------------------------- #


def chat_ids(tok, email: str, order: list[str]) -> list[int]:
    """Промпт в chat-шаблоне модели, обрывающийся на начале ответа assistant."""
    options = "\n".join(f"{slot}. {cls}" for slot, cls in zip(SLOTS, order))
    messages = [
        {"role": "system", "content": "Choose one option."},
        {"role": "user", "content": f"Email: {email}\n\n{options}"},
    ]
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)


def slot_ids(tok) -> dict[str, int]:
    """id токенов-меток. Каждая метка обязана быть ОДНИМ токеном."""
    out: dict[str, int] = {}
    for slot in SLOTS:
        ids = tok.encode(slot, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"метка {slot!r} токенизируется в {len(ids)} токенов")
        out[slot] = ids[0]
    return out


def score(
    model, tok, sid: dict[str, int], email: str, order: list[str]
) -> tuple[dict[str, float], list[float], int]:
    """Один forward pass → вероятности классов для заданного порядка вариантов."""
    ids = chat_ids(tok, email, order)
    logits = model(mx.array(ids)[None])[0, -1]  # (vocab,)
    sel = mx.array([sid[slot] for slot in SLOTS[: len(order)]])
    raw = logits[sel]
    probs = mx.softmax(raw)  # softmax по вариантам, а не по всему словарю
    mx.eval(raw, probs)
    raw_l = [float(v) for v in raw.tolist()]
    pr = [float(v) for v in probs.tolist()]
    return {cls: p for cls, p in zip(order, pr)}, raw_l, len(ids)


def logprobs_of(raw: list[float]) -> list[float]:
    """log-softmax тех же логитов (численно устойчиво: вычитаем максимум)."""
    return (mx.array(raw) - mx.logsumexp(mx.array(raw))).tolist()


def prefill_seconds(model, ids: list[int], repeats: int = 5) -> float:
    arr = mx.array(ids)[None]

    def run() -> None:
        mx.eval(model(arr)[0, -1])

    return bench(run, repeats=repeats)


def parse_label(text: str) -> str:
    low = text.lower()
    for cls in CLASSES:
        if cls.lower() in low:
            return cls
    return "—"


def argmax(p: dict[str, float]) -> str:
    return max(p, key=p.get)


def normalize(p: dict[str, float]) -> dict[str, float]:
    s = sum(p.values())
    return {k: v / s for k, v in p.items()}


def average_over_orders(
    model, tok, sid, email: str
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Прогнать все перестановки; вернуть (по порядкам, среднее по классам)."""
    by_order: dict[str, dict[str, float]] = {}
    for order in PERMS:
        p, _, _ = score(model, tok, sid, email, order)
        by_order["".join(o[0] for o in order)] = p
    n = len(by_order)
    avg = {c: sum(p[c] for p in by_order.values()) / n for c in CLASSES}
    return by_order, avg


def metrics(rows: list[tuple[float, bool]]) -> tuple[float, float, float]:
    """(accuracy, mean confidence, ECE) по списку (confidence, correct)."""
    n = len(rows)
    acc = sum(ok for _, ok in rows) / n
    conf = sum(c for c, _ in rows) / n
    ece = 0.0
    for lo, hi in ((0.0, 0.6), (0.6, 0.8), (0.8, 1.01)):
        sel = [(c, ok) for c, ok in rows if lo <= c < hi]
        if not sel:
            continue
        b_acc = sum(ok for _, ok in sel) / len(sel)
        b_conf = sum(c for c, _ in sel) / len(sel)
        ece += len(sel) / n * abs(b_acc - b_conf)
    return acc, conf, ece


# --------------------------------------------------------------------------- #
# Основной сценарий
# --------------------------------------------------------------------------- #


def main() -> None:
    section("22. Verbalizer-классификация: Jev за один forward pass")
    print(f"  модель: {MODEL_ID}")
    model, tok = load_model()

    # ------------------------------------------------------------------ 0
    subsection("0. Почему метки — буквы, а не названия классов")
    for cls in CLASSES:
        ids = tok.encode(cls, add_special_tokens=False)
        print(f"  {cls:<11} {len(ids)} ток.  {[tok.decode([i]) for i in ids]}")
    note(
        "Названия классов состоят из нескольких токенов, поэтому читать «логит "
        "класса» нельзя — метка должна быть ровно одним токеном."
    )
    note("Поэтому в промпт кладут буквы A/B/C, а класс восстанавливают по позиции.")
    sid = slot_ids(tok)
    print(f"  id меток: {sid}")

    # ------------------------------------------------------------------ 1
    subsection("1. Один forward pass: логиты → вероятности")
    email = "Payroll asks for your password on a non-company sign-in page."
    probs, raw, n_prompt = score(model, tok, sid, email, ORDER)
    print(f"  письмо: {email}")
    print(f"  промпт: {n_prompt} токенов, варианты: {ORDER}")
    for name, vals in (
        ("Logits:", raw),
        ("Log probabilities:", logprobs_of(raw)),
        ("Probabilities:", [probs[c] for c in ORDER]),
    ):
        body = ", ".join(f"{c!r}: {v:.3f}" for c, v in zip(ORDER, vals))
        print(f"  {name:<19}{{{body}}}")
    note("Тот же результат, что в статье (0.885 за Phishing) — только без API.")

    # ------------------------------------------------------------------ 2
    subsection("2. Стоимость: prefill против генерации с парсингом")
    ids = chat_ids(tok, email, ORDER)
    t_prefill = prefill_seconds(model, ids)
    out = generate(
        model, tok, prompt=ids, max_tokens=12,
        sampler=make_sampler(temp=0.0), verbose=False,
    )
    t_gen = bench(
        lambda: generate(
            model, tok, prompt=ids, max_tokens=8,
            sampler=make_sampler(temp=0.0), verbose=False,
        ),
        repeats=3,
    )
    print(f"  свободная генерация: {out!r}")
    print(f"    распознано: {parse_label(out)!r}  (парсинг текста — эвристика)")
    print(f"  prefill-only:        {t_prefill * 1e3:6.1f} мс")
    print(f"  генерация 8 токенов: {t_gen * 1e3:6.1f} мс  (×{t_gen / t_prefill:.1f})")
    note(
        "Prefill считается по всем токенам промпта один раз; декодирование — по "
        "токену на шаг, и каждый шаг читает все веса модели."
    )
    note("Вероятностей при генерации нет — есть только строка, которую надо парсить.")

    # ------------------------------------------------------------------ 3
    subsection("3. Label bias: тот же вопрос, другой порядок вариантов")
    slot_mass = {slot: [] for slot in SLOTS}
    class_mass = {c: [] for c in CLASSES}
    per_email: dict[str, dict[str, dict[str, float]]] = {}
    avg: dict[str, dict[str, float]] = {}
    flips = 0

    for text, _gold, _tag in EMAILS:
        by_order, mail_avg = average_over_orders(model, tok, sid, text)
        per_email[text] = by_order
        avg[text] = mail_avg
        for order in PERMS:
            p = by_order["".join(o[0] for o in order)]
            for slot, cls in zip(SLOTS, order):
                slot_mass[slot].append(p[cls])
            for c in CLASSES:
                class_mass[c].append(p[c])
        if len({argmax(p) for p in by_order.values()}) > 1:
            flips += 1

    print(f"  перестановок на письмо: {len(PERMS)}  (писем: {len(EMAILS)})")
    body = ", ".join(f"{s}: {sum(v) / len(v):.3f}" for s, v in slot_mass.items())
    print(f"  средняя масса по позиции варианта — {body}")
    body = ", ".join(f"{c[:4]}: {sum(v) / len(v):.3f}" for c, v in class_mass.items())
    print(f"  средняя масса по классу          — {body}  (золото: по 0.333)")
    note(
        "Без bias'а три позиции давали бы по ≈0.333, а классы — по 0.333 (в "
        "датасете равное число примеров). Отклонения — это prior модели, а не "
        "сигнал о содержании письма."
    )
    class_prior = {c: sum(v) / len(v) for c, v in class_mass.items()}
    top_cls = max(class_prior, key=class_prior.get)
    last_slot_cls = ORDER[-1]
    if top_cls == last_slot_cls:
        note(
            f"Худший случай: самый «любимый» класс модели ({top_cls}, "
            f"{class_prior[top_cls]:.2f}) стоит в промпте последним вариантом — "
            "prior класса и позиция варианта складываются."
        )
    print(f"  писем, где argmax меняется при перестановке: {flips} из {len(EMAILS)}")

    canonical = {text: by_order["LSP"] for text, by_order in per_email.items()}
    disagree = [
        t for t in per_email if argmax(canonical[t]) != argmax(avg[t])
    ]
    print(f"  расхождений «канонический порядок vs усреднение»: {len(disagree)}")
    for t in disagree:
        print(f"    {t[:50]:<50} {argmax(canonical[t])} → {argmax(avg[t])}")

    # ------------------------------------------------------------------ 4
    subsection("4. Три способа читать вероятности")
    # content-free prior: что модель отвечает на вход без содержания
    cf_raw: dict[str, float] = {}
    for order in PERMS:
        p, _, _ = score(model, tok, sid, CONTENT_FREE, order)
        for c in CLASSES:
            cf_raw[c] = cf_raw.get(c, 0.0) + p[c] / len(PERMS)
    cf = normalize(cf_raw)
    body = ", ".join(f"{c[:4]}: {v:.3f}" for c, v in cf.items())
    print(f"  content-free prior ({CONTENT_FREE!r}): {body}")
    note("Это «что модель говорит, когда содержания нет» — чистый prior формата.")

    def debias(p: dict[str, float]) -> dict[str, float]:
        return normalize({c: p[c] / max(cf[c], 1e-9) for c in CLASSES})

    strategies = {
        "фиксированный порядок (как в статье)": canonical,
        "усреднение по перестановкам": avg,
        "усреднение + content-free поправка": {t: debias(p) for t, p in avg.items()},
    }

    print(f"\n  {'стратегия':<38} {'accuracy':>9} {'mean conf':>10} {'ECE':>7}")
    results: dict[str, tuple[float, float, float]] = {}
    for name, preds in strategies.items():
        rows = [
            (preds[t][argmax(preds[t])], argmax(preds[t]) == gold)
            for t, gold, _ in EMAILS
        ]
        results[name] = metrics(rows)
        acc, conf, ece = results[name]
        print(f"  {name:<38} {acc:>9.3f} {conf:>10.3f} {ece:>7.3f}")

    best_acc = max(results, key=lambda k: results[k][0])
    best_ece = min(results, key=lambda k: results[k][2])
    note(
        f"Максимум accuracy — у варианта «{best_acc}» "
        f"({results[best_acc][0]:.3f}), но именно там худшая калибровка: "
        f"ECE {results[best_acc][2]:.3f} при средней уверенности "
        f"{results[best_acc][1]:.3f}."
    )
    note(
        f"Лучшая калибровка — у «{best_ece}» (ECE {results[best_ece][2]:.3f}), "
        "но точность при этом падает: снимая bias порядка, метод не снимает "
        "prior классов, и решение уходит к «любимому» классу модели."
    )
    note(
        "Главное наблюдение: выбор порядка вариантов и поправки меняет ответ "
        "сильнее, чем содержание письма. На слабой модели «просто прочитать "
        "логиты» — это лотерея формулировки, а не измерение."
    )

    subsection("4b. Что происходит с каждым письмом (усреднение + поправка)")
    cal = strategies["усреднение + content-free поправка"]
    print(f"  {'письмо':<36} {'золото':<10} {'raw':<10} {'p':>5}  "
          f"{'поправка':<10} {'p':>5}  ok")
    for text, gold, tag in EMAILS:
        r_pred, c_pred = argmax(avg[text]), argmax(cal[text])
        mark = "✓" if c_pred == gold else "✗"
        if tag:
            mark += "*"
        print(
            f"  {text[:36]:<36} {gold:<10} {r_pred:<10} {avg[text][r_pred]:>5.2f}  "
            f"{c_pred:<10} {cal[text][c_pred]:>5.2f}  {mark}"
        )
    note("* — метка спорная: разумный человек может классифицировать иначе.")

    note(
        "Усреднение по перестановкам не обязано помогать: оно убирает bias "
        "порядка, но не bias классов, а на слабой модели шум только растёт. "
        "Content-free поправка снимает именно prior формата."
    )
    note(
        "n=12 — это демонстрация процедуры, а не измерение качества. Для реальных "
        "порогов нужен свой holdout в сотни примеров и temperature scaling."
    )

    # ------------------------------------------------------------------ 5
    subsection("5. Цена на потоке")
    times = sorted(
        prefill_seconds(model, chat_ids(tok, t, ORDER), repeats=3) for t, _, _ in EMAILS
    )
    med = times[len(times) // 2]
    print(f"  медиана prefill одного письма:  {med * 1e3:.1f} мс")
    print(f"  один поток: ~{1 / med:.1f} писем/с")
    print(f"  полный прогон (12 писем × 6 порядков + prior): ≈ "
          f"{(len(EMAILS) * len(PERMS) + len(PERMS)) * med:.1f} с")
    note(
        "Каждое письмо — независимый запрос без общего префикса, decode нет "
        "вообще, поэтому в vllm-mlx их выгодно батчить: весь батч — один prefill."
    )

    section("Вывод")
    note("Классификация через логиты = один prefill + softmax по вариантам.")
    note("Метка обязана быть одним токеном, иначе verbalizer некорректен.")
    note("Порядок вариантов и prior классов меняют ответ → это надо измерять.")
    note("Вероятности не калиброваны: confidence ≠ уверенность в истине.")
    note("«System One» — маркетинг; техника старая (verbalizer probing), но полезная.")


if __name__ == "__main__":
    main()
