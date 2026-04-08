import numpy as np
import pandas as pd

# -----------------------------
# AGMA FACTORS
# -----------------------------
max_Vt = 0
cur_Vt = 0
def Kv(Vt, Av=8):  # quality assumed from JIS grade 8 which is AGMA 8-9
    B = 0.25*(Av-5.0)**(0.667)
    C = 50 + 56 * (1 - B)
    global max_Vt, cur_Vt
    max_Vt = (C+(14-Av))**2 / 196.85
    if cur_Vt < Vt:
        cur_Vt = Vt
    return (C / (C + np.sqrt(196.85 * Vt))) ** -B


# def Km(F):
#     if F < 50:
#         return 1.6
#     elif F >= 500:
#         return 2.0
#     elif 50 <= F < 150:
#         return 1.6 + (1.7 - 1.6) * (F - 50) / (150 - 50)
#     elif 150 <= F < 250:
#         return 1.7 + (1.8 - 1.7) * (F - 150) / (250 - 150)
#     elif 250 <= F < 500:
#         return 1.8 + (2.0 - 1.8) * (F - 250) / (500 - 250)

def Km(face, dw1, Av=8):

    KHmc = 1.0

    # KHpf = face / (10 * face)   # = 0.1 (fallback assumption)
    ratio_term = face / (10 * dw1)
    if ratio_term < 0.05:
        ratio_term = 0.05

    # Determine KHpf based on face width (b) ranges
    if face <= 25:
        # Equation (39)
        KHpf = ratio_term - 0.025

    elif face <= 432:  # Equation (40)
        KHpf = ratio_term - 0.0375 + (0.000492 * face)

    elif face <= 1020:  # Equation (41)
        KHpf = ratio_term - 0.1109 + (0.000815 * face) - (0.000000353 * (face ** 2))

    KHpm = 1.0  # unknwn val. conservative val used

    A = 0.127
    B = 0.622e-3
    C = -1.69e-7

    KHma = A + B * face + C * face**2

    return 1 + KHmc * (KHpf + KHpm + KHma)

def Ka():
    return 1.0

def KT(temp = 40):  # deg C
    if temp <= 120:
        return 1.0
    else:
        return (temp + 273) / 393

def Ks():
    return 1.0


def KB():
    return 1.0


def KI():
    return 1.0


def I(phi, Ng, Np):
    return (np.sin(np.deg2rad(phi)) * np.cos(np.deg2rad(phi))) * (Ng / (Np + Ng)) / 2


def Cp(v, E):
    return np.sqrt(1 / (np.pi * 2 * ((1 - v**2) / E)))


# -----------------------------
# GEAR OBJECT
# -----------------------------

class Gear:

    def __init__(self, name, N, module, face, J, pdiam, HB, phi=20):

        self.name = f"{name} ({N})"
        self.N = N
        self.m = module
        self.face = face
        self.J = J
        self.pdiam = pdiam
        self.HB = HB
        self.sfb_uc = 0.533*HB + 88.3
        self.sfc_uc = 2.22*HB + 200

        self.phi = phi


# -----------------------------
# GEAR MESH
# -----------------------------

class GearMesh:

    def __init__(self, gear1, gear2, Wt, Vt, v=0.3, E=200e3):

        self.g1 = gear1  # pinion
        self.g2 = gear2  # gear

        self.Wt = Wt
        self.Vt = Vt
        self.Km_val = Km(min(self.g1.face, self.g2.face), min(self.g1.pdiam, self.g2.pdiam))
        self.v = v
        self.E = E

    # -------------------------
    # BENDING
    # -------------------------

    def bending_stress(self, gear):

        return ((self.Wt * Ka() * self.Km_val * Ks() * KB() * KI() * Kv(self.Vt)) /
                (gear.face * gear.m * gear.J ))


    def bending_strength(self, gear):

        KL = KR = 1

        return KL * gear.sfb_uc / (KT() * KR)

    def bending_SF(self, gear):

        return self.bending_strength(gear) / self.bending_stress(gear)

    # -------------------------
    # CONTACT
    # -------------------------

    def surface_stress(self):

        Ng = max(self.g1.N, self.g2.N)
        Np = min(self.g1.N, self.g2.N)

        Cp_val = Cp(self.v, self.E)
        face = min(self.g1.face, self.g2.face)
        return Cp_val * np.sqrt(
            (self.Wt * Ka() * self.Km_val * Ks() * Kv(self.Vt)) /
            (self.g1.face * I(self.g1.phi, Ng, Np) * self.g1.pdiam)
        )

    def surface_strength(self, gear):

        CL = CH = KR = 1

        return (CL * CH * gear.sfc_uc) / (KT() * KR)

    def surface_SF(self, gear):

        return (self.surface_strength(gear) / self.surface_stress())**2


# -----------------------------
# GEAR TRAIN
# -----------------------------

class GearTrain:

    def __init__(self):

        self.meshes = []

    def add_mesh(self, mesh):

        self.meshes.append(mesh)

    def report(self):

        print("\nGEAR SAFETY FACTORS")
        print("-------------------")
        rows = []
        for mesh in self.meshes:

            for gear in [mesh.g1, mesh.g2]:

                print(f"{gear.name}")
                print("  Bending SF:", round(mesh.bending_SF(gear), 3))
                print("  Surface SF:", round(mesh.surface_SF(gear), 3))
                print()
                rows.append({
                    'Gear': gear.name,
                    'Bending SF': round(mesh.bending_SF(gear), 3),
                    'Surface SF': round(mesh.surface_SF(gear), 3),
                })
        print("max=", max_Vt)
        print("current =", cur_Vt)
        rows.append({'Gear': 'Max Vt', 'Bending SF': max_Vt})
        rows.append({'Gear': 'Current Vt', 'Bending SF': cur_Vt})

        df_gears = pd.DataFrame(rows)

        return df_gears

