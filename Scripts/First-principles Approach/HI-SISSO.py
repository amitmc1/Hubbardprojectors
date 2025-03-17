import numpy as np
import sys

import argparse
import numpy as np

# Define a function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Run bruteforce calculations with given constants.')

    # Define command-line arguments for constants
    parser.add_argument('--DFT_x1', type=float, default=0.08463)
    parser.add_argument('--DFT_x2', type=float, default=0.22705)
    parser.add_argument('--DFT_x3', type=float, default=0.06966)
    parser.add_argument('--DFT_x4', type=float, default=0.10708)
    parser.add_argument('--DFT_x5', type=float, default=0.06966)
    parser.add_argument('--DFT_x6', type=float, default=0.04721)
    parser.add_argument('--DFT_x7', type=float, default=0.08463)
    parser.add_argument('--DFT_p1', type=float, default=1.7503)
    parser.add_argument('--DFT_p2', type=float, default=1.7503)
    parser.add_argument('--DFT_p3', type=float, default=1.7503)

    parser.add_argument('--pbe0_x1', type=float, default=0.05782)
    parser.add_argument('--pbe0_x2', type=float, default=0.17793)
    parser.add_argument('--pbe0_x3', type=float, default=0.04824)
    parser.add_argument('--pbe0_x4', type=float, default=0.07219)
    parser.add_argument('--pbe0_x5', type=float, default=0.04824)
    parser.add_argument('--pbe0_x6', type=float, default=0.03387)
    parser.add_argument('--pbe0_x7', type=float, default=0.05782)
    parser.add_argument('--pbe0_p1', type=float, default=1.69136)
    parser.add_argument('--pbe0_p2', type=float, default=1.69136)
    parser.add_argument('--pbe0_p3', type=float, default=1.69136)

    # Additional parameters
    parser.add_argument('--R', type=float, default=204)
    parser.add_argument('--EN', type=float, default=1.12)
    parser.add_argument('--QN', type=int, default=4)
    parser.add_argument('--Subshell', type=int, default=7)
    parser.add_argument('--Ion_OE', type=int, default=0)
    parser.add_argument('--Auxiliary_type', type=int, default=2)
    parser.add_argument('--Auxiliary_zval', type=float, default=7.6)

    parser.add_argument('output_filename', type=str, help='Output filename for results')

    return parser.parse_args()


# Define the constants from command-line arguments
def define_constants(args):
    global DFT_x1, DFT_x2, DFT_x3, DFT_x4, DFT_x5, DFT_x6, DFT_x7, DFT_p1, DFT_p2, DFT_p3
    global pbe0_x1, pbe0_x2, pbe0_x3, pbe0_x4, pbe0_x5, pbe0_x6, pbe0_x7, pbe0_p1, pbe0_p2, pbe0_p3
    global R, EN, QN, Subshell, Ion_OE, Auxiliary_type, Auxiliary_zval

    DFT_x1 = args.DFT_x1
    DFT_x2 = args.DFT_x2
    DFT_x3 = args.DFT_x3
    DFT_x4 = args.DFT_x4
    DFT_x5 = args.DFT_x5
    DFT_x6 = args.DFT_x6
    DFT_x7 = args.DFT_x7
    DFT_p1 = args.DFT_p1
    DFT_p2 = args.DFT_p2
    DFT_p3 = args.DFT_p3

    pbe0_x1 = args.pbe0_x1
    pbe0_x2 = args.pbe0_x2
    pbe0_x3 = args.pbe0_x3
    pbe0_x4 = args.pbe0_x4
    pbe0_x5 = args.pbe0_x5
    pbe0_x6 = args.pbe0_x6
    pbe0_x7 = args.pbe0_x7
    pbe0_p1 = args.pbe0_p1
    pbe0_p2 = args.pbe0_p2
    pbe0_p3 = args.pbe0_p3

    R = args.R
    EN = args.EN
    QN = args.QN
    Subshell = args.Subshell
    Ion_OE = args.Ion_OE
    Auxiliary_type = args.Auxiliary_type
    Auxiliary_zval = args.Auxiliary_zval

# Defining SISSO step 1 constants
c0_x1 = -1.951236129073281e-02
a0_x1 = 5.128449136947671e-02
a1_x1 = 1.785034248838293e+00

c0_x2 = -7.445728969329735e-02
a0_x2 = 2.216771677440891e-02
a1_x2 = -3.059511969511726e-05
a2_x2 = 1.079519469820436e+00

c0_x3 = 3.208064089564413e-01
a0_x3 = 2.591690596845641e-05
a1_x3 = 8.847072738629081e-03
a2_x3 = -3.785611734913045e-01

c0_x4 = -3.909920306940013e-02
a0_x4 = 3.477936607128601e-01
a1_x4 = 7.986501878636137e-04
a2_x4 = 1.793752994773251e+00

c0_x5 = 3.430127454194143e-01
a0_x5 = 7.754277220220173e-03
a1_x5 = 6.204613664611021e-03
a2_x5 = -3.795065856532934e-01

c0_x6 = -9.350363160816184e-07
a0_x6 = 1.938784268992287e-02
a1_x6 = -2.376427331550605e-02
a2_x6 = 1.396924718552828e-01

c0_x7 = -1.279760152763598e-06
a0_x7 = 1.736143050320491e-01
a1_x7 = 7.761144764187236e-01
a2_x7 = 1.789223309696820e+00

c0_p1 = 9.079617982840955e-01
a0_p1 = -7.499799609685134e-02
a1_p1 = -1.212959054110834e-02
a2_p1 = 1.284831979908578e-01

c0_p2 = 1.889403536308601e+00
a0_p2 = 1.009336131826201e-01
a1_p2 = 1.936307550717508e-02
a2_p2 = -1.322444133783561e+00

c0_p3 = 1.218420960538720e+00
a0_p3 = 9.366081516156602e-02
a1_p3 = 2.471503313894520e-04
a2_p3 = 1.183199615449630e-01

# Defining SISSO step 2 constants

c0_x1_s2 = 8.173958056058755e-03
a0_x1_s2 = 7.883730063115383e-03
a1_x1_s2 = -1.828681031329470e-07
a2_x1_s2 = -1.611072667429941e+02

c0_x2_s2 = 3.464532620360462e-03
a0_x2_s2 = -3.634673128194995e-02
a1_x2_s2 = -1.639774818304222e-02
a2_x2_s2 = 2.007217960127654e+00

c0_x3_s2 = 4.913894360257094e-03
a0_x3_s2 = 2.744222765470687e-02
a1_x3_s2 = -1.295798287928023e-03
a2_x3_s2 = 1.246325878840032e+01

c0_x4_s2 = 1.887312571444848e-02
a0_x4_s2 = -1.157856294865445e-02
a1_x4_s2 = 6.257547055145304e-02
a2_x4_s2 = 1.514973711070728e+02

c0_x5_s2 = 2.743156412042411e-02
a0_x5_s2 = -1.408541160430133e-04
a1_x5_s2 = -3.152487375898972e-02
a2_x5_s2 = 1.970158616780767e+02

c0_x6_s2 = -4.020585402820685e-05
a0_x6_s2 = 1.434398201701655e-05
a1_x6_s2 = -1.628826616181706e-04
a2_x6_s2 = 2.413322081833866e-05

c0_x7_s2 = -3.829891043911724e-04
a0_x7_s2 = 4.468894551606105e-04
a1_x7_s2 = -6.041413899323906e-04
a2_x7_s2 = -5.738353492844799e-04

c0_p1_s2 = 1.769961291026731e+00
a0_p1_s2 = 1.487987392547155e-04
a1_p1_s2 = -1.336923954072562e-02
a2_p1_s2 = -1.984506137409937e-01

c0_p2_s2 = 1.770008071342302e+00
a0_p2_s2 = 2.520947599033304e-02
a1_p2_s2 = -6.559065395065979e-03
a2_p2_s2 = -1.226778233424393e+00

c0_p3_s2 = 1.771617793473578e+00
a0_p3_s2 = -1.396235856577293e-02
a1_p3_s2 = -9.443642535351371e-03
a2_p3_s2 = -5.828042795791698e+00


def calculate_hubbard_x1_1(U, c1, c2):
    sisso_x1 = (c0_x1 + a0_x1 * ((U / np.log(DFT_x1)) * (np.cbrt(c2) + np.log(DFT_p1))) + a1_x1 * (((DFT_x1 / DFT_x3) * DFT_x5) * (np.sin(c1) ** 3)))
    return sisso_x1

def calculate_hubbard_x1_2(U, c1, c2):
    sisso_x1 = (c0_x1 + a0_x1 * ((U / np.log(DFT_x1)) * (np.cbrt(c2) + np.log(DFT_p1))) + a1_x1 * (((DFT_x1 / DFT_x3) * DFT_x5) * (np.sin(c1) ** 3)))
    SISSO_x1_s2 = c0_x1_s2 + a0_x1_s2 * ((np.log(EN) * (Auxiliary_zval - Ion_OE)) - np.sin((Auxiliary_zval**3))) + a1_x1_s2 * (((sisso_x1 * Auxiliary_zval)**3) / (np.cbrt(R) - (sisso_x1**2))) + a2_x1_s2 * (np.sin(np.exp(Auxiliary_type)) / (np.sin(R) - (R / sisso_x1)))
    return SISSO_x1_s2

# Defining SISSO correlation for DFT+U_x2
def calculate_hubbard_x2_1(U, c1, c2):
    sisso_x2 = c0_x2 + a0_x2 * (((DFT_x3 + U) - (DFT_p2**3)) / np.log(DFT_x4)) + a1_x2 * (((U**2)**2) / (np.sin(c2) + (DFT_p1 - c1))) + a2_x2 * (((DFT_x7 + DFT_x1) + DFT_x7) * (c1**2))
    return sisso_x2

def calculate_hubbard_x2_2(U, c1, c2):
    sisso_x2 = c0_x2 + a0_x2 * (((DFT_x3 + U) - (DFT_p2**3)) / np.log(DFT_x4)) + a1_x2 * (((U**2)**2) / (np.sin(c2) + (DFT_p1 - c1))) + a2_x2 * (((DFT_x7 + DFT_x1) + DFT_x7) * (c1**2))
    SISSO_x2_s2 = c0_x2_s2 + a0_x2_s2 * np.sin(((Subshell - QN) * (sisso_x2**3))) + a1_x2_s2 * ((np.cbrt(Auxiliary_zval) - Auxiliary_type) / (np.sin(sisso_x2) - np.log(sisso_x2))) + a2_x2_s2 * (np.sqrt(np.exp(QN)) * ((sisso_x2 / QN) / QN))
    return SISSO_x2_s2

# Defining SISSO correlation for DFT+U_x3
def calculate_hubbard_x3_1(U, c1, c2):
    sisso_x3 = c0_x3 + a0_x3 * (((U**2)**2) / ((DFT_p2 + c2) - c1)) + a1_x3 * ((np.log(DFT_x4) + U) / (np.log(DFT_x3) + np.sin(c1))) + a2_x3 * (np.sqrt(np.exp(c2)) - ((DFT_x5 * c1) * np.exp(c1)))
    return sisso_x3

def calculate_hubbard_x3_2(U, c1, c2):
    sisso_x3 = c0_x3 + a0_x3 * (((U**2)**2) / ((DFT_p2 + c2) - c1)) + a1_x3 * ((np.log(DFT_x4) + U) / (np.log(DFT_x3) + np.sin(c1))) + a2_x3 * (np.sqrt(np.exp(c2)) - ((DFT_x5 * c1) * np.exp(c1)))
    SISSO_x3_s2 = c0_x3_s2 + a0_x3_s2 * (np.sin((sisso_x3**3)) + ((Auxiliary_type**2) / (sisso_x3 * R))) + a1_x3_s2 * (np.sin(QN) / (np.log(QN) + (sisso_x3 - EN))) + a2_x3_s2 * (np.sin(np.exp(Auxiliary_type)) * ((sisso_x3 / Subshell) / np.sqrt(Subshell)))
    return SISSO_x3_s2

# Defining SISSO correlation for DFT+U_x4
def calculate_hubbard_x4_1(U, c1, c2):
    sisso_x4 = c0_x4 + a0_x4 * (((c2**2)**2) / ((DFT_x5**2) - (c1 + U))) + a1_x4 * ((U**3) / ((DFT_x5 * c2) + np.log(DFT_x4))) + a2_x4 * (((DFT_x6 + DFT_x1) * np.sin(DFT_p1)) * (np.sin(c1)**3))
    return sisso_x4

def calculate_hubbard_x4_2(U, c1, c2):
    sisso_x4 = c0_x4 + a0_x4 * (((c2**2)**2) / ((DFT_x5**2) - (c1 + U))) + a1_x4 * ((U**3) / ((DFT_x5 * c2) + np.log(DFT_x4))) + a2_x4 * (((DFT_x6 + DFT_x1) * np.sin(DFT_p1)) * (np.sin(c1)**3))
    SISSO_x4_s2 = c0_x4_s2 + a0_x4_s2 * (np.sin((EN / sisso_x4)) / ((EN**3) - (1.0 / sisso_x4))) + a1_x4_s2 * (np.sin((Auxiliary_zval + Subshell)) * np.sin((sisso_x4 * Ion_OE))) + a2_x4_s2 * (np.sin(np.exp(Auxiliary_type)) / ((R / sisso_x4) - (Auxiliary_type**2)))
    return SISSO_x4_s2

# Defining SISSO correlation for DFT+U_x5
def calculate_hubbard_x5_1(U, c1, c2):
    sisso_x5 = c0_x5 + a0_x5 * (((c2**3) * (DFT_x2**2)) / (np.exp(c1) - U)) + a1_x5 * ((np.exp(c2) * U) / (np.log(DFT_x3) + np.sin(c1))) + a2_x5 * ((np.exp(c2)**(1/3)) - ((DFT_x5 * c1) * np.exp(c1)))
    return sisso_x5

def calculate_hubbard_x5_2(U, c1, c2):
    sisso_x5 = c0_x5 + a0_x5 * (((c2**3) * (DFT_x2**2)) / (np.exp(c1) - U)) + a1_x5 * ((np.exp(c2) * U) / (np.log(DFT_x3) + np.sin(c1))) + a2_x5 * ((np.exp(c2)**(1/3)) - ((DFT_x5 * c1) * np.exp(c1)))
    SISSO_x5_s2 = c0_x5_s2 + a0_x5_s2 * ((np.exp(Auxiliary_type) - Auxiliary_zval) / (np.log(EN) - np.sin(sisso_x5))) + a1_x5_s2 * np.sin((np.sqrt(QN) * (sisso_x5 * Subshell))) + a2_x5_s2 * (np.sin(np.exp(Auxiliary_type)) / ((R / sisso_x5) * np.cbrt(EN)))
    return SISSO_x5_s2

# Defining SISSO correlation for DFT+U_x6
def calculate_hubbard_x6_1(U, c1, c2):
    sisso_x6 = c0_x6 + a0_x6 * (((c2 / DFT_x2) + np.sin(U)) * DFT_x7) + a1_x6 * (((c1**3)**3) * (DFT_x6 / (DFT_x2 - U))) + a2_x6 * (((c1 / DFT_x2) - (U)**(1/3)) * DFT_x7)
    return sisso_x6

def calculate_hubbard_x6_2(U, c1, c2):
    sisso_x6 = c0_x6 + a0_x6 * (((c2 / DFT_x2) + np.sin(U)) * DFT_x7) + a1_x6 * (((c1**3)**3) * (DFT_x6 / (DFT_x2 - U))) + a2_x6 * (((c1 / DFT_x2) - (U)**(1/3)) * DFT_x7)
    SISSO_x6_s2 = c0_x6_s2 + a0_x6_s2 * (1.0 / ((sisso_x6 * R) + (sisso_x6 - Subshell))) + a1_x6_s2 * (1.0 / ((sisso_x6 * R) - (Auxiliary_type + QN))) + a2_x6_s2 * (((R**2) * np.sin(sisso_x6)) + np.sin((R / sisso_x6)))
    return SISSO_x6_s2

# Defining SISSO correlation for DFT+U_x7
def calculate_hubbard_x7_1(U, c1, c2):
    sisso_x7 = c0_x7 + a0_x7 * (np.sin((DFT_p3 * U)) * (np.log(c1) * DFT_x7)) + a1_x7 * ((DFT_x6 / np.exp(c2)) * ((c2**3) + np.log(c1))) + a2_x7 * ((DFT_x6 * c1) / np.exp((DFT_x1 * U)))
    return sisso_x7

def calculate_hubbard_x7_2(U, c1, c2):
    sisso_x7 = c0_x7 + a0_x7 * (np.sin((DFT_p3 * U)) * (np.log(c1) * DFT_x7)) + a1_x7 * ((DFT_x6 / np.exp(c2)) * ((c2**3) + np.log(c1))) + a2_x7 * ((DFT_x6 * c1) / np.exp((DFT_x1 * U)))
    SISSO_x7_s2 = c0_x7_s2 + a0_x7_s2 * np.sin((np.exp(QN) * (sisso_x7 * R))) + a1_x7_s2 * np.sin(((Subshell * R) * (sisso_x7 * QN))) + a2_x7_s2 * (np.sin((1.0 / sisso_x7)) - ((sisso_x7 * QN) * (Auxiliary_zval**3)))
    return SISSO_x7_s2

# Defining SISSO correlation for DFT+U_p1
def calculate_hubbard_p1_1(U, c1, c2):
    sisso_p1 = c0_p1 + a0_p1 * (((DFT_x2 + c2) / np.sqrt(DFT_x5)) * np.sin((1.0 / DFT_x5))) + a1_p1 * (np.sin((U / DFT_x1)) / ((DFT_p2**3) - DFT_p3)) + a2_p1 * ((np.sqrt(U) + np.exp(DFT_p3)) + (c2 * U))
    return sisso_p1

def calculate_hubbard_p1_2(U, c1, c2):
    sisso_p1 = c0_p1 + a0_p1 * (((DFT_x2 + c2) / np.sqrt(DFT_x5)) * np.sin((1.0 / DFT_x5))) + a1_p1 * (np.sin((U / DFT_x1)) / ((DFT_p2**3) - DFT_p3)) + a2_p1 * ((np.sqrt(U) + np.exp(DFT_p3)) + (c2 * U))
    SISSO_p1_s2 = c0_p1_s2 + a0_p1_s2 * (np.exp((Subshell / sisso_p1)) / (np.log(sisso_p1) - np.sin(Auxiliary_type))) + a1_p1_s2 * (np.sin((sisso_p1**3)) / (np.sin(sisso_p1) - (sisso_p1 - Auxiliary_type))) + a2_p1_s2 * (np.sin((sisso_p1**2)) * (np.log(R) + (sisso_p1 - Subshell)))
    return SISSO_p1_s2

# Defining SISSO correlation for DFT+U_p2
def calculate_hubbard_p2_1(U, c1, c2):
    sisso_p2 = c0_p2 + a0_p2 * (np.sin((DFT_p1 / DFT_x5)) * np.sin(DFT_x4 * U)) + a1_p2 * (np.sin(DFT_p1 ** 2) / (np.cbrt(c2) + (DFT_x2 - c2))) + a2_p2 * (np.sin(np.cbrt(DFT_x5)) / ((DFT_p2 ** 3) - (DFT_p3 - U)))
    return sisso_p2

def calculate_hubbard_p2_2(U, c1, c2):
    sisso_p2 = c0_p2 + a0_p2 * (np.sin((DFT_p1 / DFT_x5)) * np.sin(DFT_x4 * U)) + a1_p2 * (np.sin(DFT_p1 ** 2) / (np.cbrt(c2) + (DFT_x2 - c2))) + a2_p2 * (np.sin(np.cbrt(DFT_x5)) / ((DFT_p2 ** 3) - (DFT_p3 - U)))
    SISSO_p2_s2 = c0_p2_s2 + a0_p2_s2 * np.sin(((QN - R) / np.exp(sisso_p2))) + a1_p2_s2 * (np.sin(np.exp(QN)) / ((sisso_p2**2) - (sisso_p2 + Auxiliary_type))) + a2_p2_s2 * (np.sin((sisso_p2**2)) / ((Subshell - EN) + np.sin(R)))
    return SISSO_p2_s2

# Defining SISSO correlation for DFT+U_p3
def calculate_hubbard_p3_1(U, c1, c2):
    sisso_p3 = c0_p3 + a0_p3 * (((c1 * U) / (DFT_p1)**(1/3)) - np.exp((U)**(1/3))) + a1_p3 * (((c2**2) - (DFT_x7 * U)) / (DFT_x1**3)) + a2_p3 * ((np.sqrt(U) + (DFT_p3**3)) + (c2 * U))
    return sisso_p3

def calculate_hubbard_p3_2(U, c1, c2):
    sisso_p3 = c0_p3 + a0_p3 * (((c1 * U) / (DFT_p1)**(1/3)) - np.exp((U)**(1/3))) + a1_p3 * (((c2**2) - (DFT_x7 * U)) / (DFT_x1**3)) + a2_p3 * ((np.sqrt(U) + (DFT_p3**3)) + (c2 * U))
    SISSO_p3_s2 = c0_p3_s2 + a0_p3_s2 * np.sin(((QN**2) * (sisso_p3 / EN))) + a1_p3_s2 * (np.sin((Auxiliary_zval * QN)) * ((EN**3) - (sisso_p3**2))) + a2_p3_s2 * ((np.log(sisso_p3) - (1.0 / sisso_p3)) / (np.sin(Ion_OE) - Subshell))
    return SISSO_p3_s2

# Define a function to perform parameter search
def perform_search(output_file):
    U_values = np.linspace(0, 5, 50)  # Example range and number of values for U
    c1_values = np.linspace(1, 0.5, 50)  # Example range and number of values for c1
    c2_values = np.linspace(0, -0.6, 50)  # Example range and number of values for c2

    # Open the file for writing
    with open(output_file, 'w') as f:
        f.write("U,c1,c2,SISSO_x1_1,SISSO_x1_2,SISSO_x2_1,SISSO_x2_2,SISSO_x3_1,SISSO_x3_2,"
               "SISSO_x4_1,SISSO_x4_2,SISSO_x5_1,SISSO_x5_2,SISSO_x6_1,SISSO_x6_2,"
               "SISSO_x7_1,SISSO_x7_2,SISSO_p1_1,SISSO_p1_2,SISSO_p2_1,SISSO_p2_2,"
               "SISSO_p3_1,SISSO_p3_2\n")

        for U in U_values:
            for c1 in c1_values:
                for c2 in c2_values:
                    SISSO_x1_1 = calculate_hubbard_x1_1(U, c1, c2)
                    SISSO_x1_2 = calculate_hubbard_x1_2(U, c1, c2)
                    SISSO_x2_1 = calculate_hubbard_x2_1(U, c1, c2)
                    SISSO_x2_2 = calculate_hubbard_x2_2(U, c1, c2)
                    SISSO_x3_1 = calculate_hubbard_x3_1(U, c1, c2)
                    SISSO_x3_2 = calculate_hubbard_x3_2(U, c1, c2)
                    SISSO_x4_1 = calculate_hubbard_x4_1(U, c1, c2)
                    SISSO_x4_2 = calculate_hubbard_x4_2(U, c1, c2)
                    SISSO_x5_1 = calculate_hubbard_x5_1(U, c1, c2)
                    SISSO_x5_2 = calculate_hubbard_x5_2(U, c1, c2)
                    SISSO_x6_1 = calculate_hubbard_x6_1(U, c1, c2)
                    SISSO_x6_2 = calculate_hubbard_x6_2(U, c1, c2)
                    SISSO_x7_1 = calculate_hubbard_x7_1(U, c1, c2)
                    SISSO_x7_2 = calculate_hubbard_x7_2(U, c1, c2)
                    SISSO_p1_1 = calculate_hubbard_p1_1(U, c1, c2)
                    SISSO_p1_2 = calculate_hubbard_p1_2(U, c1, c2)
                    SISSO_p2_1 = calculate_hubbard_p2_1(U, c1, c2)
                    SISSO_p2_2 = calculate_hubbard_p2_2(U, c1, c2)
                    SISSO_p3_1 = calculate_hubbard_p3_1(U, c1, c2)
                    SISSO_p3_2 = calculate_hubbard_p3_2(U, c1, c2)

                    f.write(f"{U},{c1},{c2},{SISSO_x1_1},{SISSO_x1_2},{SISSO_x2_1},{SISSO_x2_2},"
                           f"{SISSO_x3_1},{SISSO_x3_2},{SISSO_x4_1},{SISSO_x4_2},{SISSO_x5_1},{SISSO_x5_2},"
                           f"{SISSO_x6_1},{SISSO_x6_2},{SISSO_x7_1},{SISSO_x7_2},{SISSO_p1_1},{SISSO_p1_2},"
                           f"{SISSO_p2_1},{SISSO_p2_2},{SISSO_p3_1},{SISSO_p3_2}\n")


def main():
    args = parse_args()
    define_constants(args)
    perform_search(args.output_filename)

if __name__ == "__main__":
    main()