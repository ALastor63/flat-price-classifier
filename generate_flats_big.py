#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор большого датасета квартир для обучения нейросети.
Создаёт файл flats_big.csv с синтетическими данными.
"""

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    n = 1000

    # Количество комнат: 1, 2 или 3
    rooms = rng.integers(1, 4, size=n)

    # Площадь зависит от количества комнат
    area = (
        rng.normal(30, 5, size=n) * (rooms == 1)
        + rng.normal(50, 7, size=n) * (rooms == 2)
        + rng.normal(75, 10, size=n) * (rooms == 3)
    )

    # Этаж от 1 до 25
    floor = rng.integers(1, 26, size=n)

    # Базовая цена за м² + небольшой шум
    base_price_per_m2 = (
        120_000 * (rooms == 1) + 140_000 * (rooms == 2) + 160_000 * (rooms == 3)
    )
    noise = rng.normal(0, 300_000, size=n)
    price = area * base_price_per_m2 + noise

    df = pd.DataFrame(
        {
            "price": price.round(0),
            "area": area.round(1),
            "rooms": rooms,
            "floor": floor,
        }
    )

    df.to_csv("flats_big.csv", index=False)
    print("Сохранён файл flats_big.csv с", len(df), "квартирами.")


if __name__ == "__main__":
    main()


