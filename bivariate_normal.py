"""
CDF of the bivariate normal distribution, but FASSST.
"""
from numba import vectorize, float64
from scipy.special import erfinv
import math


@vectorize([float64(float64)], nopython=True)
def norm_cdf(X):
    """
    A function for computing the cumulative distribution function
    of the standard normal distribution.
    """
    return 0.5 * (1 + math.erf(X / math.sqrt(2)))


@vectorize([float64(float64)], nopython=True)
def norm_ppf(x):
    return math.sqrt(2) * erfinv(2 * x - 1)


@vectorize([float64(float64, float64, float64)], nopython=True, cache=True)
def bvn_cdf(SH, SK, R):
    """
    CDF of the bivariate normal distribution.
    Adapted from
    <https://github.com/scipy/scipy/blob/de80faf9d3480b9dbb9b888568b64499e0e70c19/scipy/stats/mvndst.f>
    """
    # test infinite values
    if SH == -math.inf or SK == -math.inf:
        return 0
    if SH == math.inf:
        return norm_cdf(SK)
    if SK == math.inf:
        return norm_cdf(SH)
    ZERO = 0
    TWOPI = 6.283185307179586
    X = [
        [-0.9324695142031522, -0.6612093864662647, -0.2386191860831970],
        [
            -0.9815606342467191,
            -0.9041172563704750,
            -0.7699026741943050,
            -0.5873179542866171,
            -0.3678314989981802,
            -0.1252334085114692,
        ],
        [
            -0.9931285991850949,
            -0.9639719272779138,
            -0.9122344282513259,
            -0.8391169718222188,
            -0.7463319064601508,
            -0.6360536807265150,
            -0.5108670019508271,
            -0.3737060887154196,
            -0.2277858511416451,
            -0.07652652113349733,
        ],
    ]

    W = [
        [0.1713244923791705, 0.3607615730481384, 0.4679139345726904],
        [
            0.04717533638651177,
            0.1069393259953183,
            0.1600783285433464,
            0.2031674267230659,
            0.2334925365383547,
            0.2491470458134029,
        ],
        [
            0.01761400713915212,
            0.04060142980038694,
            0.06267204833410906,
            0.08327674157670475,
            0.1019301198172404,
            0.1181945319615184,
            0.1316886384491766,
            0.1420961093183821,
            0.1491729864726037,
            0.1527533871307259,
        ],
    ]
    LG = [3, 6, 10]
    if abs(R) < 0.3:
        NG = 1
    elif abs(R) < 0.75:
        NG = 2
    else:
        NG = 3
    H = -SH
    K = -SK
    HK = H * K
    BVN = 0
    if abs(R) < 0.925:
        HS = (H * H + K * K) / 2
        ASR = math.asin(R)
        for i in range(LG[NG - 1]):
            SN = math.sin(ASR * (X[NG - 1][i] + 1) / 2)
            BVN += W[NG - 1][i] * math.exp((SN * HK - HS) / (1 - SN * SN))
            SN = math.sin(ASR * (-X[NG - 1][i] + 1) / 2)
            BVN += W[NG - 1][i] * math.exp((SN * HK - HS) / (1 - SN * SN))
        BVN = BVN * ASR / (2 * TWOPI) + norm_cdf(-H) * norm_cdf(-K)
    else:
        if R < 0:
            K = -K
            HK = -HK
        if abs(R) < 1:
            AS = (1 - R) * (1 + R)
            A = math.sqrt(AS)
            BS = (H - K) ** 2
            C = (4 - HK) / 8
            D = (12 - HK) / 16
            BVN = (
                A
                * math.exp(-(BS / AS + HK) / 2)
                * (1 - C * (BS - AS) * (1 - D * BS / 5) / 3 + C * D * AS * AS / 5)
            )
            if HK > -160:
                B = math.sqrt(BS)
                BVN -= (
                    math.exp(-HK / 2)
                    * math.sqrt(TWOPI)
                    * norm_cdf(-B / A)
                    * B
                    * (1 - C * BS * (1 - D * BS / 5) / 3)
                )
            A = A / 2
            for i in range(LG[NG - 1]):
                XS = (A * (X[NG - 1][i] + 1)) ** 2
                RS = math.sqrt(1 - XS)
                BVN += (
                    A
                    * W[NG - 1][i]
                    * (
                        math.exp(-BS / (2 * XS) - HK / (1 + RS)) / RS
                        - math.exp(-(BS / XS + HK) / 2) * (1 + C * XS * (1 + D * XS))
                    )
                )
                XS = AS * (-X[NG - 1][i] + 1) ** 2 / 4
                RS = math.sqrt(1 - XS)
                BVN += (
                    A
                    * W[NG - 1][i]
                    * math.exp(-(BS / XS + HK) / 2)
                    * (
                        math.exp(-HK * (1 - RS) / (2 * (1 + RS))) / RS
                        - (1 + C * XS * (1 + D * XS))
                    )
                )
            BVN = -BVN / TWOPI
        if R > 0:
            BVN += norm_cdf(-max(H, K))
        if R < 0:
            BVN = -BVN + max(ZERO, norm_cdf(-H) - norm_cdf(-K))
    return BVN


if __name__ == "__main__":
    from scipy.stats import multivariate_normal
    import random

    for i in range(1000):
        cov = random.random() * 2 - 1
        x = random.normalvariate(0, 1)
        y = random.normalvariate(0, 1)
        # do the same with abseps=1e-6, releps=1e-6
        a = multivariate_normal(mean=[0, 0], cov=[[1, cov], [cov, 1]]).cdf([x, y])
        b = bvn_cdf(x, y, cov)
        assert abs(a - b) < 1e-10
