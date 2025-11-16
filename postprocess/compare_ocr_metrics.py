import argparse
import os

def levenshtein(a, b):
    """Посимвольное расстояние Левенштейна."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ca = a[i - 1]
        for j in range(1, m + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # удаление
                dp[i][j - 1] + 1,      # вставка
                dp[i - 1][j - 1] + cost  # замена
            )
    return dp[n][m]

def cer(ref_lines, hyp_lines):
    """Character Error Rate."""
    total_dist = 0
    total_chars = 0
    for r, h in zip(ref_lines, hyp_lines):
        r = r.strip()
        h = h.strip()
        if not r:
            continue
        total_dist += levenshtein(r, h)
        total_chars += len(r)
    if total_chars == 0:
        return 0.0
    return total_dist / total_chars

def wer(ref_lines, hyp_lines):
    """Word Error Rate."""
    total_dist = 0
    total_words = 0
    for r, h in zip(ref_lines, hyp_lines):
        r_words = r.strip().split()
        h_words = h.strip().split()
        if not r_words:
            continue
        total_dist += levenshtein(r_words, h_words)
        total_words += len(r_words)
    if total_words == 0:
        return 0.0
    return total_dist / total_words

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="эталонный txt")
    parser.add_argument("--hyp", required=True, help="гипотеза (распознавание)")
    args = parser.parse_args()

    if not os.path.exists(args.ref):
        print("Не найден ref:", args.ref)
        return
    if not os.path.exists(args.hyp):
        print("Не найден hyp:", args.hyp)
        return

    with open(args.ref, "r", encoding="utf-8") as f:
        ref_lines = f.readlines()
    with open(args.hyp, "r", encoding="utf-8") as f:
        hyp_lines = f.readlines()

    # выравниваем по длине (обрезаем лишние строки)
    n = min(len(ref_lines), len(hyp_lines))
    ref_lines = ref_lines[:n]
    hyp_lines = hyp_lines[:n]

    cer_value = cer(ref_lines, hyp_lines)
    wer_value = wer(ref_lines, hyp_lines)

    print(f"CER: {cer_value:.4f}")
    print(f"WER: {wer_value:.4f}")

if __name__ == "__main__":
    main()
