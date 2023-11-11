import time
import numpy as np
from numba import njit # импорт компилятора для ускорения питона


@njit
def match_timestamps(timestamps1: np.ndarray, timestamps2: np.ndarray, cam2_frame: np.int32=0) -> np.ndarray: 
    """
    * Ищем начальный кадр для второй камеры - O(2n), однако реальная сложность меньше, так как cam2_frame уже найден эвристически с учётом числа кадров на первой и второй камере
    * Делаем для каждого элемента (времени) с первой камеры: - O(n)
            * Вычитаем из соответствующего времени кадра второй камеры значение времени кадра с первой камеры - O(2n) в худшем случае, O(2) ~ O(1) в среднем, так как кадры распределены равномерно и во втором массиве не на порядок больше элементов\n
                    * если дельта уменьшилась - переходим к следующему кадру на второй камере и так далее, пока дельта уменьшается\n
                    * если дельта не уменьшилась - возвращаем предыдущий кадр, как самый близкий по времени\n
    Итоговая сложность: O(размер_второго_массива + размер_первого_массива * размер_второго_массива) ~ [в среднем] ~ O(размер_второго_массива + размер_первого_массива * 1) ~ O(n)\n
    Parameters:
    * timestamps1: массив с временами с камеры 1
    * timestamps2: массив с временами с камеры 2
    * cam2_frame: с какого элемента рассматривать кадры с камеры 2 (важен при параллельном запуске)\n
    Returns:
    * np.ndarray: массив соответствий кадров с камеры 1 к кадрам камеры 2
    """

    frames_count = timestamps1.shape[0] # число кадров на камере 1
    max_frame = timestamps2.shape[0] - 1 # максимальный номер кадра, что может соответствовать кадру с первой камеры (-1 из-за индексации с нуля)
    correspondence = np.zeros(frames_count, dtype=np.int32) # создаём массив под номера кадров с типом int32 для уменьшения потребляемой памяти

    if 0 > cam2_frame or cam2_frame > max_frame: # защита от дурака с cam2_frame
        cam2_frame = 0 # зануление начального кадра второй камеры

    # подбираем кадр на второй камере, с которого будем начинать (эвристически уже передали ожидаемый cam2_frame, нужно его лишь правильно сместить, если он "обогнал" кадры с камеры 1)
    delta = np.abs(timestamps2[cam2_frame] - timestamps1[0]) # текущая разность времени
    while np.abs(timestamps2[max(0, cam2_frame-1)] - timestamps1[0]) < delta: # если дельта уменьшается при предыдущем кадре, то переходим на него (условие — строго меньше, так как времена всех кадров уникальны)
        cam2_frame -= 1 # переходим на предыдущий кадр (без max, так как при нуле мы бы сюда не зашли в цикл)
        delta = np.abs(timestamps2[cam2_frame] - timestamps1[0]) # обновляем текущее значение delta (без max, так как при нуле мы бы не зашли в цикл)
    # после этого цикла мы либо находимся на оптимальном соответствии первого кадра первой камеры с кадром со второй камеры, либо оптимум — далее по времени (но не раньше!)

    for frame, frame_time in enumerate(timestamps1): # идём по кадрам
        delta = np.abs(timestamps2[cam2_frame] - frame_time) # текущая разность времени
        while np.abs(timestamps2[min(cam2_frame+1, max_frame)] - frame_time) < delta: # если при переходе на следующий кадр дельта (разница времени) уменьшилась, то обновляем delta и соответствующий кадр на второй камере 
            cam2_frame += 1 # переходим на следующий кадр
            delta = np.abs(timestamps2[cam2_frame] - frame_time)  # обновляем delta
        correspondence[frame] = cam2_frame # если дельта перестала уменьшаеться — записываем найденный кадр

    return correspondence


def make_timestamps(fps: int, st_ts: float, fn_ts: float) -> np.ndarray:
    """
    Create array of timestamps. This array is discretized with fps,
    but not evenly.
    Timestamps are assumed sorted and unique.
    Parameters:
    - fps: int
        Average frame per second
    - st_ts: float
        First timestamp in the sequence
    - fn_ts: float
        Last timestamp in the sequence
    Returns:
        np.ndarray: synthetic timestamps
    """
    # generate uniform timestamps
    timestamps = np.linspace(st_ts, fn_ts, int((fn_ts - st_ts) * fps))
    # add an fps noise
    timestamps += np.random.randn(len(timestamps))
    timestamps = np.unique(np.sort(timestamps))
    return timestamps


def main():
    """
    Setup:
        Say we have two cameras, each filming the same scene. We make
        a prediction based on this scene (e.g. detect a human pose).
        To improve the robustness of the detection algorithm,
        we average the predictions from both cameras at each moment.
        The camera data is a pair (frame, timestamp), where the timestamp
        represents the moment when the frame was captured by the camera.
        The structure of the prediction does not matter here. 

    Problem:
        For each frame of camera1, we need to find the index of the
        corresponding frame received by camera2. The frame i from camera2
        corresponds to the frame j from camera1, if
        abs(timestamps[i] - timestamps[j]) is minimal for all i.

    Estimation criteria:
        - The solution has to be optimal algorithmically. If the
    best solution turns out to have the O(n^3) complexity [just an example],
    the solution with O(n^3 * logn) will have -1 point,
    the solution O(n^4) will have -2 points and so on.
    Make sure your algorithm cannot be optimized!
        - The solution has to be optimal python-wise.
    If it can be optimized ~x5 times by rewriting the algorithm in Python,
    this will be a -1 point. x20 times optimization will give -2 points, and so on.
    You may use any library, even write your own
    one in C++.
        - All corner cases must be handled correctly. A wrong solution
    will have -3 points.
        - Top 3 solutions get 10 points. The measurement will be done in a single thread. 
        - The base score is 9.
        - Shipping the solution in a Docker container results in +1 point.
    Such solution must contain a Dockerfile, which later will be built
    via `docker build ...`, and the hw1.py script will be called from this container.
    Try making this container as small as possible in Mb!
        - Parallel implementation adds +1 point, provided it is effective
    (cannot be optimized x5 times)
        - Maximal score is 10 points, minimal score is 5 points.
        - The deadline is November 21 23:59. Failing the deadline will
    result in -2 points, and each additional week will result in -1 point.
        - The solution can be improved/fixed after the deadline provided that the initial
    version is submitted on time.

    Optimize the solution to work with ~2-3 hours of data.
    Good luck!
    """
    # generate timestamps for the first camera
    timestamps1 = make_timestamps(30, time.time() - 100, time.time() + 3600 * 2)
    # generate timestamps for the second camera
    timestamps2 = make_timestamps(60, time.time() + 200, time.time() + 3600 * 2.5)
    matching = match_timestamps(timestamps1, timestamps2)


if __name__ == '__main__':
    main()