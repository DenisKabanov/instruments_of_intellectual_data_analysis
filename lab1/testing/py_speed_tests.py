import time
import numpy as np
from multiprocessing import Pool
from numba import njit


cores_number = 2 # число ядер для запуска
runs = 5 # количество запусков для усреднения времени


# наивные функции =============================================================================================================================================
def match_timestamps_naive_nonjit_arange(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray: 
    frames_count = timestamps1.shape[0] # число кадров на камере 1
    correspondence = np.zeros(frames_count, dtype=np.int32) # создаём массив под номера кадров с типом int32 для уменьшения потребляемой памяти
    for frame in np.arange(frames_count): # идём по номерам кадров
        correspondence[frame] = np.absolute(timestamps2 - timestamps1[frame]).argmin()
    return correspondence


def match_timestamps_naive_nonjit_enumerate(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray: 
    frames_count = timestamps1.shape[0] # число кадров на камере 1
    correspondence = np.zeros(frames_count, dtype=np.int32) # создаём массив под номера кадров с типом int32 для уменьшения потребляемой памяти
    for frame, frame_time in enumerate(timestamps1): # идём по кадрам
        correspondence[frame] = np.absolute(timestamps2 - frame_time).argmin()
    return correspondence


@njit
def match_timestamps_naive_njit_arange(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray: 
    frames_count = timestamps1.shape[0] # число кадров на камере 1
    correspondence = np.zeros(frames_count, dtype=np.int32) # создаём массив под номера кадров с типом int32 для уменьшения потребляемой памяти
    for frame in np.arange(frames_count): # идём по номерам кадров
        correspondence[frame] = np.absolute(timestamps2 - timestamps1[frame]).argmin()
    return correspondence


@njit
def match_timestamps_naive_njit_enumerate(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray: 
    frames_count = timestamps1.shape[0] # число кадров на камере 1
    correspondence = np.zeros(frames_count, dtype=np.int32) # создаём массив под номера кадров с типом int32 для уменьшения потребляемой памяти
    for frame, frame_time in enumerate(timestamps1): # идём по кадрам
        correspondence[frame] = np.absolute(timestamps2 - frame_time).argmin()
    return correspondence

# итерированные функции =======================================================================================================================================
def match_timestamps_iterated_nonjit_arange(timestamps1: np.ndarray, timestamps2: np.ndarray, cam2_frame: np.int32=0) -> np.ndarray: 
    frames_count = timestamps1.shape[0] # число кадров на камере 1
    max_frame = timestamps2.shape[0] - 1 # максимальный номер кадра, что может соответствовать кадру с первой камеры (-1 из-за индексации с нуля)
    correspondence = np.zeros(frames_count, dtype=np.int32) # создаём массив под номера кадров с типом int32 для уменьшения потребляемой памяти
    if 0 > cam2_frame or cam2_frame > max_frame: # защита от дурака с cam2_frame
        cam2_frame = 0 # зануление начального кадра второй камеры
    delta = np.abs(timestamps2[cam2_frame] - timestamps1[0]) # текущая разность времени
    while np.abs(timestamps2[max(0, cam2_frame-1)] - timestamps1[0]) < delta: # если дельта уменьшается при предыдущем кадре, то переходим на него (условие — строго меньше, так как времена всех кадров уникальны)
        cam2_frame -= 1 # переходим на предыдущий кадр (без max, так как при нуле мы бы сюда не зашли в цикл)
        delta = np.abs(timestamps2[cam2_frame] - timestamps1[0]) # обновляем текущее значение delta (без max, так как при нуле мы бы не зашли в цикл)
    for frame in np.arange(frames_count): # идём по номерам кадров с первой камеры
        delta = np.abs(timestamps2[cam2_frame] - timestamps1[frame]) # текущая разность времени
        while np.abs(timestamps2[min(cam2_frame+1, max_frame)] - timestamps1[frame]) < delta: # если при переходе на следующий кадр дельта (разница времени) уменьшилась, то обновляем delta и соответствующий кадр на второй камере 
            cam2_frame += 1 # переходим на следующий кадр
            delta = np.abs(timestamps2[cam2_frame] - timestamps1[frame])  # обновляем delta
        correspondence[frame] = cam2_frame # если дельта перестала уменьшаеться — записываем найденный кадр
    return correspondence


def match_timestamps_iterated_nonjit_enumerate(timestamps1: np.ndarray, timestamps2: np.ndarray, cam2_frame: np.int32=0) -> np.ndarray: 
    frames_count = timestamps1.shape[0] # число кадров на камере 1
    max_frame = timestamps2.shape[0] - 1 # максимальный номер кадра, что может соответствовать кадру с первой камеры (-1 из-за индексации с нуля)
    correspondence = np.zeros(frames_count, dtype=np.int32) # создаём массив под номера кадров с типом int32 для уменьшения потребляемой памяти
    if 0 > cam2_frame or cam2_frame > max_frame: # защита от дурака с cam2_frame
        cam2_frame = 0 # зануление начального кадра второй камеры
    delta = np.abs(timestamps2[cam2_frame] - timestamps1[0]) # текущая разность времени
    while np.abs(timestamps2[max(0, cam2_frame-1)] - timestamps1[0]) < delta: # если дельта уменьшается при предыдущем кадре, то переходим на него (условие — строго меньше, так как времена всех кадров уникальны)
        cam2_frame -= 1 # переходим на предыдущий кадр (без max, так как при нуле мы бы сюда не зашли в цикл)
        delta = np.abs(timestamps2[cam2_frame] - timestamps1[0]) # обновляем текущее значение delta (без max, так как при нуле мы бы не зашли в цикл)
    for frame, frame_time in enumerate(timestamps1): # идём по кадрам
        delta = np.abs(timestamps2[cam2_frame] - frame_time) # текущая разность времени
        while np.abs(timestamps2[min(cam2_frame+1, max_frame)] - frame_time) < delta: # если при переходе на следующий кадр дельта (разница времени) уменьшилась, то обновляем delta и соответствующий кадр на второй камере 
            cam2_frame += 1 # переходим на следующий кадр
            delta = np.abs(timestamps2[cam2_frame] - frame_time)  # обновляем delta
        correspondence[frame] = cam2_frame # если дельта перестала уменьшаеться — записываем найденный кадр
    return correspondence


@njit
def match_timestamps_iterated_njit_arange(timestamps1: np.ndarray, timestamps2: np.ndarray, cam2_frame: np.int32=0) -> np.ndarray: 
    frames_count = timestamps1.shape[0] # число кадров на камере 1
    max_frame = timestamps2.shape[0] - 1 # максимальный номер кадра, что может соответствовать кадру с первой камеры (-1 из-за индексации с нуля)
    correspondence = np.zeros(frames_count, dtype=np.int32) # создаём массив под номера кадров с типом int32 для уменьшения потребляемой памяти
    if 0 > cam2_frame or cam2_frame > max_frame: # защита от дурака с cam2_frame
        cam2_frame = 0 # зануление начального кадра второй камеры
    delta = np.abs(timestamps2[cam2_frame] - timestamps1[0]) # текущая разность времени
    while np.abs(timestamps2[max(0, cam2_frame-1)] - timestamps1[0]) < delta: # если дельта уменьшается при предыдущем кадре, то переходим на него (условие — строго меньше, так как времена всех кадров уникальны)
        cam2_frame -= 1 # переходим на предыдущий кадр (без max, так как при нуле мы бы сюда не зашли в цикл)
        delta = np.abs(timestamps2[cam2_frame] - timestamps1[0]) # обновляем текущее значение delta (без max, так как при нуле мы бы не зашли в цикл)
    for frame in np.arange(frames_count): # идём по номерам кадров с первой камеры
        delta = np.abs(timestamps2[cam2_frame] - timestamps1[frame]) # текущая разность времени
        while np.abs(timestamps2[min(cam2_frame+1, max_frame)] - timestamps1[frame]) < delta: # если при переходе на следующий кадр дельта (разница времени) уменьшилась, то обновляем delta и соответствующий кадр на второй камере 
            cam2_frame += 1 # переходим на следующий кадр
            delta = np.abs(timestamps2[cam2_frame] - timestamps1[frame])  # обновляем delta
        correspondence[frame] = cam2_frame # если дельта перестала уменьшаеться — записываем найденный кадр
    return correspondence


@njit
def match_timestamps_iterated_njit_enumerate(timestamps1: np.ndarray, timestamps2: np.ndarray, cam2_frame: np.int32=0) -> np.ndarray: 
    frames_count = timestamps1.shape[0] # число кадров на камере 1
    max_frame = timestamps2.shape[0] - 1 # максимальный номер кадра, что может соответствовать кадру с первой камеры (-1 из-за индексации с нуля)
    correspondence = np.zeros(frames_count, dtype=np.int32) # создаём массив под номера кадров с типом int32 для уменьшения потребляемой памяти
    if 0 > cam2_frame or cam2_frame > max_frame: # защита от дурака с cam2_frame
        cam2_frame = 0 # зануление начального кадра второй камеры
    delta = np.abs(timestamps2[cam2_frame] - timestamps1[0]) # текущая разность времени
    while np.abs(timestamps2[max(0, cam2_frame-1)] - timestamps1[0]) < delta: # если дельта уменьшается при предыдущем кадре, то переходим на него (условие — строго меньше, так как времена всех кадров уникальны)
        cam2_frame -= 1 # переходим на предыдущий кадр (без max, так как при нуле мы бы сюда не зашли в цикл)
        delta = np.abs(timestamps2[cam2_frame] - timestamps1[0]) # обновляем текущее значение delta (без max, так как при нуле мы бы не зашли в цикл)
    for frame, frame_time in enumerate(timestamps1): # идём по кадрам
        delta = np.abs(timestamps2[cam2_frame] - frame_time) # текущая разность времени
        while np.abs(timestamps2[min(cam2_frame+1, max_frame)] - frame_time) < delta: # если при переходе на следующий кадр дельта (разница времени) уменьшилась, то обновляем delta и соответствующий кадр на второй камере 
            cam2_frame += 1 # переходим на следующий кадр
            delta = np.abs(timestamps2[cam2_frame] - frame_time)  # обновляем delta
        correspondence[frame] = cam2_frame # если дельта перестала уменьшаеться — записываем найденный кадр
    return correspondence


# параллелизация ==============================================================================================================================================
def match_timestamps_naive_parallel(timestamps1: np.ndarray, timestamps2: np.ndarray, func, cores_number) -> np.ndarray: 
    frames_count = timestamps1.shape[0]
    with Pool(cores_number) as p:
        correspondence = np.concatenate(p.starmap(func, [(timestamps1[int(i*frames_count/cores_number):int((i+1)*frames_count/cores_number)], timestamps2) for i in np.arange(cores_number)]), dtype=np.int32)
    return correspondence


def match_timestamps_iterated_parallel(timestamps1: np.ndarray, timestamps2: np.ndarray, func, cores_number: np.int8) -> np.ndarray: 
    frames_count = timestamps1.shape[0] # число кадров с первой камеры
    with Pool(cores_number) as p: # запускаем cores_number процессов
        correspondence = np.concatenate(p.starmap(func, [(timestamps1[int(i*frames_count/cores_number):int((i+1)*frames_count/cores_number)], timestamps2, int(2*i*frames_count/cores_number)) for i in np.arange(cores_number)]), dtype=np.int32)
    return correspondence


# основные функции ============================================================================================================================================
def make_timestamps(fps: int, st_ts: float, fn_ts: float) -> np.ndarray:
    timestamps = np.linspace(st_ts, fn_ts, int((fn_ts - st_ts) * fps))
    timestamps += np.random.randn(len(timestamps))
    timestamps = np.unique(np.sort(timestamps))
    return timestamps


def main():
    timestamps1 = make_timestamps(30, time.time() - 100, time.time() + 900 * 2)
    timestamps2 = make_timestamps(60, time.time() + 200, time.time() + 900 * 2.5)

# naive no njit + np.arange cycle
    time_start = time.perf_counter()
    for run in range(runs):
        true_answer = match_timestamps_naive_nonjit_arange(timestamps1, timestamps2)
    print(f"naive no njit + np.arange cycle average time: {(time.perf_counter() - time_start) / runs}")


    time_start = time.perf_counter()
    for run in range(runs):
        answer = match_timestamps_naive_parallel(timestamps1, timestamps2, match_timestamps_naive_nonjit_arange, cores_number)
    print(f"naive parallel no njit + np.arange cycle average time: {(time.perf_counter() - time_start) / runs}")

    if not np.all(answer == true_answer):
        print("Error!")

# naive no njit + enumerate cycle
    time_start = time.perf_counter()
    for run in range(runs):
        answer = match_timestamps_naive_nonjit_enumerate(timestamps1, timestamps2)
    print(f"naive no njit + enumerate cycle average time: {(time.perf_counter() - time_start) / runs}")


    time_start = time.perf_counter()
    for run in range(runs):
        answer = match_timestamps_naive_parallel(timestamps1, timestamps2, match_timestamps_naive_nonjit_enumerate, cores_number)
    print(f"naive parallel no njit + enumerate cycle average time: {(time.perf_counter() - time_start) / runs}")

    if not np.all(answer == true_answer):
        print("Error!")

# naive njit + np.arange cycle
    time_start = time.perf_counter()
    for run in range(runs):
        answer = match_timestamps_naive_njit_arange(timestamps1, timestamps2)
    print(f"naive njit + np.arange cycle average time: {(time.perf_counter() - time_start) / runs}")


    time_start = time.perf_counter()
    for run in range(runs):
        answer = match_timestamps_naive_parallel(timestamps1, timestamps2, match_timestamps_naive_njit_arange, cores_number)
    print(f"naive parallel njit + np.arange cycle average time: {(time.perf_counter() - time_start) / runs}")

    if not np.all(answer == true_answer):
        print("Error!")

# naive njit + enumerate cycle
    time_start = time.perf_counter()
    for run in range(runs):
        answer = match_timestamps_naive_njit_enumerate(timestamps1, timestamps2)
    print(f"naive njit + enumerate cycle average time: {(time.perf_counter() - time_start) / runs}")


    time_start = time.perf_counter()
    for run in range(runs):
        answer = match_timestamps_naive_parallel(timestamps1, timestamps2, match_timestamps_naive_njit_enumerate, cores_number)
    print(f"naive parallel njit + enumerate cycle average time: {(time.perf_counter() - time_start) / runs}")

    if not np.all(answer == true_answer):
        print("Error!")

# iterated no njit + np.arange cycle
    time_start = time.perf_counter()
    for run in range(runs):
        true_answer = match_timestamps_iterated_nonjit_arange(timestamps1, timestamps2)
    print(f"iterated no njit + np.arange cycle average time: {(time.perf_counter() - time_start) / runs}")


    time_start = time.perf_counter()
    for run in range(runs):
        answer = match_timestamps_iterated_parallel(timestamps1, timestamps2, match_timestamps_iterated_nonjit_arange, cores_number)
    print(f"iterated parallel no njit + np.arange cycle average time: {(time.perf_counter() - time_start) / runs}")

    if not np.all(answer == true_answer):
        print("Error!")

# iterated no njit + enumerate cycle
    time_start = time.perf_counter()
    for run in range(runs):
        true_answer = match_timestamps_iterated_nonjit_enumerate(timestamps1, timestamps2)
    print(f"iterated no njit + enumerate cycle average time: {(time.perf_counter() - time_start) / runs}")


    time_start = time.perf_counter()
    for run in range(runs):
        answer = match_timestamps_iterated_parallel(timestamps1, timestamps2, match_timestamps_iterated_nonjit_enumerate, cores_number)
    print(f"iterated parallel no njit + enumerate cycle average time: {(time.perf_counter() - time_start) / runs}")

    if not np.all(answer == true_answer):
        print("Error!")

# iterated njit + np.arange cycle
    time_start = time.perf_counter()
    for run in range(runs):
        true_answer = match_timestamps_iterated_njit_arange(timestamps1, timestamps2)
    print(f"iterated njit + np.arange cycle average time: {(time.perf_counter() - time_start) / runs}")


    time_start = time.perf_counter()
    for run in range(runs):
        answer = match_timestamps_iterated_parallel(timestamps1, timestamps2, match_timestamps_iterated_njit_arange, cores_number)
    print(f"iterated parallel njit + np.arange cycle average time: {(time.perf_counter() - time_start) / runs}")

    if not np.all(answer == true_answer):
        print("Error!")

# iterated njit + enumerate cycle
    time_start = time.perf_counter()
    for run in range(runs):
        true_answer = match_timestamps_iterated_njit_enumerate(timestamps1, timestamps2)
    print(f"iterated njit + enumerate cycle average time: {(time.perf_counter() - time_start) / runs}")


    time_start = time.perf_counter()
    for run in range(runs):
        answer = match_timestamps_iterated_parallel(timestamps1, timestamps2, match_timestamps_iterated_njit_enumerate, cores_number)
    print(f"iterated parallel njit + enumerate cycle average time: {(time.perf_counter() - time_start) / runs}")

    if not np.all(answer == true_answer):
        print("Error!")


if __name__ == '__main__':
    main()