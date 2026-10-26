"""Лабораторная № 1: абсолютная погрешность численного дифференцирования."""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ScalarFunction = Callable[[float], float]


@dataclass(frozen=True)
class Function:
    name: str
    title: str
    value: ScalarFunction
    derivative: ScalarFunction


FUNCTIONS = (
    Function("sin_x_squared", r"$f(x)=\sin(x^2)$",
             lambda x: math.sin(x * x), lambda x: 2 * x * math.cos(x * x)),
    Function("cos_sin_x", r"$f(x)=\cos(\sin x)$",
             lambda x: math.cos(math.sin(x)),
             lambda x: -math.sin(math.sin(x)) * math.cos(x)),
    Function("exp_sin_cos_x", r"$f(x)=e^{\sin(\cos x)}$",
             lambda x: math.exp(math.sin(math.cos(x))),
             lambda x: -math.exp(math.sin(math.cos(x)))
             * math.cos(math.cos(x)) * math.sin(x)),
    Function("log_x_plus_3", r"$f(x)=\ln(x+3)$",
             lambda x: math.log(x + 3), lambda x: 1 / (x + 3)),
    Function("sqrt_x_plus_3", r"$f(x)=\sqrt{x+3}$",
             lambda x: math.sqrt(x + 3), lambda x: 1 / (2 * math.sqrt(x + 3))),
)


def forward(f: ScalarFunction, x: float, h: float) -> float:
    """Формула 1: правая разность, O(h)."""
    return (f(x + h) - f(x)) / h


def backward(f: ScalarFunction, x: float, h: float) -> float:
    """Формула 2: левая разность, O(h)."""
    return (f(x) - f(x - h)) / h


def central(f: ScalarFunction, x: float, h: float) -> float:
    """Формула 3: центральная разность, O(h²)."""
    return (f(x + h) - f(x - h)) / (2 * h)


def fourth_order(f: ScalarFunction, x: float, h: float) -> float:
    """Формула 4: комбинация центральных разностей, O(h⁴)."""
    return (4 / 3) * central(f, x, h) - (1 / 3) * central(f, x, 2 * h)


def sixth_order(f: ScalarFunction, x: float, h: float) -> float:
    """Формула 5: комбинация центральных разностей, O(h⁶)."""
    return ((3 / 2) * central(f, x, h)
            - (3 / 5) * central(f, x, 2 * h)
            + (1 / 10) * central(f, x, 3 * h))


METHODS = (forward, backward, central, fourth_order, sixth_order)
LABELS = ("1. Правая, O(h)", "2. Левая, O(h)", "3. Центральная, O(h²)",
          "4. Формула O(h⁴)", "5. Формула O(h⁶)")
COLORS = ("#2563eb", "#d97706", "#059669", "#9333ea", "#dc2626")
MARKERS = ("o", "s", "^", "D", "v")
STYLES = ("-", "--", "-.", ":", "-")


def steps() -> list[float]:
    """Ровно 21 шаг: h_n = 2 / 2**n, n = 1, ..., 21."""
    return [2.0 / 2**n for n in range(1, 22)]


def calculate(function: Function, x: float) -> tuple[float, list[dict]]:
    reference = function.derivative(x)
    rows = []
    for n, h in enumerate(steps(), start=1):
        values = [method(function.value, x, h) for method in METHODS]
        errors = [abs(value - reference) for value in values]
        if not all(math.isfinite(v) for v in [reference, *values, *errors]):
            raise ValueError("Получены бесконечные значения; выберите другую точку x0.")
        rows.append({"n": n, "h": h, "values": values, "errors": errors})
    return reference, rows


def write_csv(path: Path, x: float, reference: float, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["n", "h", "x0", "reference_derivative"]
                        + [f"D{i}" for i in range(1, 6)]
                        + [f"error{i}" for i in range(1, 6)])
        for row in rows:
            writer.writerow([row["n"], row["h"], x, reference,
                             *row["values"], *row["errors"]])


def plot_errors(function: Function, x: float, rows: list[dict], path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5), layout="constrained")
    # Слева малые h, справа большие; расчётные строки CSV идут по n.
    ordered = sorted(rows, key=lambda row: row["h"])
    zeros = 0
    for j, label in enumerate(LABELS):
        errors = [row["errors"][j] for row in ordered]
        zeros += errors.count(0.0)
        # Ноль не определён на log-оси. Оставляем разрыв, не подменяем ошибку.
        visible = [error if error > 0 else math.nan for error in errors]
        ax.loglog([row["h"] for row in ordered], visible,
                  color=COLORS[j], marker=MARKERS[j], linestyle=STYLES[j],
                  linewidth=1.5, markersize=4, markerfacecolor="none", label=label)
    ax.set_title(f"{function.title},  $x_0={x:g}$", fontsize=17, pad=14)
    ax.set_xlabel("Шаг дифференцирования h", fontsize=12)
    ax.set_ylabel(r"Абсолютная погрешность $|D(h)-f'(x_0)|$", fontsize=12)
    ax.grid(which="major", alpha=0.30)
    ax.grid(which="minor", alpha=0.10)
    ax.legend(fontsize=10, loc="best", framealpha=0.95)
    note = "21 шаг: hₙ = 2 / 2ⁿ, n = 1, …, 21. Обе оси логарифмические."
    if zeros:
        note += f"\nНулевых ошибок: {zeros}; в этих точках линии прерваны."
    fig.supxlabel(note, fontsize=9, color="#475569")
    fig.savefig(path, dpi=170)
    return fig


def write_summary(path: Path, x: float, experiments: list) -> None:
    lines = ["# Результаты вычислений", "",
             f"Точка: x₀ = {x:g}. Для каждой функции использован 21 шаг от 1 до 2⁻²⁰.",
             "", "Эталон вычислен по аналитической формуле в арифметике float.",
             "Минимум выбирается только среди шагов из задания, а не по всем h.", "",
             "| Функция | Формула | Минимальная ошибка | Шаг h | Ошибка при минимальном h | Нулевых ошибок |",
             "|---|---|---:|---:|---:|---:|"]
    for function, reference, rows in experiments:
        for j, label in enumerate(LABELS):
            best = min(rows, key=lambda row: row["errors"][j])
            zeros = sum(row["errors"][j] == 0 for row in rows)
            lines.append(f"| {function.title} | {label} | {best['errors'][j]:.6e} "
                         f"| {best['h']:.6e} | {rows[-1]['errors'][j]:.6e} | {zeros} |")
    lines += ["", "## Как интерпретировать", "",
              "При уменьшении h сначала уменьшается ошибка усечения. "
              "Для формул 1–5 её порядки соответственно: 1, 1, 2, 4, 6.", "",
              "При малых h вычитаются близкие значения функции. "
              "Ошибки округления усиливаются при делении на h, поэтому "
              "кривые могут выходить на плато или расти. "
              "Высокий порядок не гарантирует лучшую точность при любом шаге.", "",
              "На графиках h возрастает слева направо: уменьшению шага "
              "соответствует движение справа налево. "
              "Не каждая кривая обязана пройти минимум на заданном диапазоне.", "",
              "Нулевая ошибка означает совпадение двух чисел float, "
              "а не доказанную точность в вещественной арифметике. "
              "Нули сохранены в CSV, на логарифмическом графике пропущены.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x0", type=float, default=1.0,
                        help="точка дифференцирования (по умолчанию 1)")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent,
                        help="папка для графиков и таблиц")
    parser.add_argument("--show", action="store_true",
                        help="открыть пять графиков в отдельных окнах")
    args = parser.parse_args()
    # Максимальный шаблон использует x0 - 3h, h_max = 1.
    # Для логарифма x0 - 3 + 3 > 0, поэтому x0 > 0.
    if not math.isfinite(args.x0) or args.x0 <= 0:
        parser.error("x0 должен быть конечным и > 0: иначе ln(x+3) вне области определения.")

    # Вычисляем всё до записи файлов, чтобы при неверной точке не оставить часть результатов.
    experiments = []
    try:
        for function in FUNCTIONS:
            reference, rows = calculate(function, args.x0)
            experiments.append((function, reference, rows))
    except (ValueError, OverflowError, ZeroDivisionError) as error:
        parser.error(str(error))

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output.mkdir(parents=True, exist_ok=True)
    for i, (function, reference, rows) in enumerate(experiments, start=1):
        stem = f"{i:02d}_{function.name}"
        write_csv(args.output / f"{stem}.csv", args.x0, reference, rows)
        fig = plot_errors(function, args.x0, rows, args.output / f"{stem}.png")
        if not args.show:
            plt.close(fig)
        print(f"{i}. {function.name}: f'({args.x0:g}) = {reference:.12g}")
    write_summary(args.output / "summary.md", args.x0, experiments)
    print(f"\nСохранено: 5 графиков, 5 CSV и summary.md\nПапка: {args.output.resolve()}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
