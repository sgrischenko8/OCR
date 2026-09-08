def levenshtein(a: str, b: str) -> int:
    """Класична DP-реалізація відстані Левенштейна (без сторонніх пакетів)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        curr_row = [i] + [0] * len(b)
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            curr_row[j] = min(
                prev_row[j] + 1,      # видалення символу
                curr_row[j - 1] + 1,  # вставка символу
                prev_row[j - 1] + cost,  # заміна символу
            )
        prev_row = curr_row
    return prev_row[-1]

def text_similarity(a: str, b: str) -> float:
    """
    Нормалізована схожість двох рядків у діапазоні 0..1 (1 = ідентичні).
    Дозволяє розпізнати "ABC-123" і "ABC-128" як одну й ту саму фразу
    з невеликим OCR-шумом, а не як два різних тексти.
    """
    a, b = a.strip(), b.strip()
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / max_len
