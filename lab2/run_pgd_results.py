import time
import yaml
import numpy as np

from methods.pgd_armijo import PGDArmijo, MediaMixConstrainedOracle


# Загружаем параметры задачи
with open("configs/media_mix_pgd_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Создаём объект задачи
oracle = MediaMixConstrainedOracle(
    **config["oracle_config"]
)

# Создаём оптимизатор PGD + Armijo
solver = PGDArmijo(
    oracle,
    **config["method_config"]
)

# Запускаем оптимизацию
start_time = time.perf_counter()
hist = solver.run()
execution_time = time.perf_counter() - start_time

print(f"Время выполнения PGD + Armijo: {execution_time:.6f} секунд")

# Получаем оптимальное распределение бюджета
x_opt = hist.x[-1]

# Рассчитываем отклик каждого канала
channel_responses = oracle.response(x_opt)

# Суммарный отклик
total_response = np.sum(channel_responses)

# Начальный отклик
initial_response = np.sum(oracle.response(hist.x[0]))

# Количество итераций
iterations = len(hist.x) - 1


# Выводим результаты
print("\n========== РЕЗУЛЬТАТЫ PGD + ARMIJO ==========")

print(f"Количество итераций: {iterations}")
print(f"Начальный отклик: {initial_response:.6f}")
print(f"Итоговый суммарный отклик: {total_response:.6f}")
print(f"Итоговая целевая функция F(x): {hist.f[-1]:.6f}")
print(f"Норма проецированного градиента: {hist.grad_norm[-1]:.10e}")

print("\n========== РАСПРЕДЕЛЕНИЕ БЮДЖЕТА И ОТКЛИКА ==========")

for channel, budget, response in zip(
    oracle.channels, x_opt, channel_responses
):
    share = response / total_response * 100

    print(
        f"{channel:10s} | "
        f"Бюджет: {budget:10.6f} | "
        f"Отклик: {response:10.6f} | "
        f"Доля отклика: {share:6.2f}%"
    )

print("--------------------------------------------------")
print(f"Общий бюджет: {np.sum(x_opt):.6f}")
print(f"Общий отклик: {total_response:.6f}")
