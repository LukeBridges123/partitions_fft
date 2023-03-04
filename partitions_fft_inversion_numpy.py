import numpy as np
from numpy.polynomial import polynomial as P
import math, time
from numpy.fft import fft, ifft

def pents(N):
    m = 1
    pentagonal_nums = [1]
    while (abs(pentagonal_nums[len(pentagonal_nums)-1]) < N):
        pentagonal_nums.append((-1)**m * (((3 * m**2) - m) // 2))
        pentagonal_nums.append((-1)**m * (((3 * m**2) + m) // 2))
        m += 1
    
    return(np.array(pentagonal_nums, dtype = np.int64))
#implementation of polynomial multiplication with FFTs taken from Jeremy Kun at:
#https://jeremykun.com/2022/11/16/polynomial-multiplication-using-the-fft/
def fft_poly_mul(p1, p2):
    """Multiply two polynomials.
 
    p1 and p2 are arrays of coefficients in degree-increasing order.
    """
    deg1 = p1.shape[0] - 1
    deg2 = p2.shape[0] - 1
    # Would be 2*(deg1 + deg2) + 1, but the next-power-of-2 handles the +1
    total_num_pts = 2 * (deg1 + deg2)
    next_power_of_2 = 1 << (total_num_pts - 1).bit_length()
 
    ff_p1 = fft(np.pad(p1, (0, next_power_of_2 - p1.shape[0])))
    ff_p2 = fft(np.pad(p2, (0, next_power_of_2 - p2.shape[0])))
    product = ff_p1 * ff_p2
    inverted = ifft(product)
    rounded = np.round(np.real(inverted)).astype(np.int64)
    return np.trim_zeros(rounded, trim='b')
def p(N):
    #initialize the polynomial that will end up containing the first N terms of the generating function
    #of p(n).
    partition_generating_funct = np.array([1], dtype = np.int64)
    pentagonal_nums = pents(N)
    f_i = np.zeros(abs(pentagonal_nums[pentagonal_nums.size - 1]) + 2, dtype = np.int64)
    f_i[0] = 1
    for num in pentagonal_nums:
        f_i[abs(num)] = num//abs(num)
    max_pow = 2**(int(math.ceil(math.log2(f_i.size + 1))))
    i = 1
    while (2**i <= max_pow):
        f_i_neg = f_i.copy()
        for k in range(2**(i-1), f_i_neg.size, 2**i):
            f_i_neg[k] *= -1
        partition_generating_funct = P.polymul(partition_generating_funct, f_i_neg)
        partition_generating_funct = partition_generating_funct[0:N+1]
        f_i = np.polymul(f_i, f_i_neg)
        f_i = f_i[0:N+1]
        i += 1
    return partition_generating_funct
def p_fft(N):
    partition_generating_funct = np.array([1], dtype = np.int64)
    pentagonal_nums = pents(N)
    f_i = np.zeros(abs(pentagonal_nums[pentagonal_nums.size - 1]) + 2, dtype = np.int64)
    f_i[0] = 1
    for num in pentagonal_nums:
        f_i[abs(num)] = num//abs(num)
    max_pow = 2**(int(math.ceil(math.log2(f_i.size + 1))))
    i = 1
    while (2**i <= max_pow):
        f_i_neg = f_i.copy()
        for k in range(2**(i-1), f_i_neg.size, 2**i):
            f_i_neg[k] *= -1
        partition_generating_funct = fft_poly_mul(partition_generating_funct, f_i_neg)
        partition_generating_funct = partition_generating_funct[0:N+1]
        f_i = fft_poly_mul(f_i, f_i_neg)
        f_i = f_i[0:N+1]
        i += 1
    return partition_generating_funct

start = time.perf_counter_ns()
p_fft(500000)
end = time.perf_counter_ns()
diff = end - start
print(diff / 10**9)
