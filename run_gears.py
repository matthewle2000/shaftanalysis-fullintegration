import numpy as np
import pandas as pd
from classes import Gear, GearTrain, GearMesh


mod = 5  # for all gears, phi=20 is built in to object gear, E and v are also built in (same for all gears)
def run_gear_system(T12, T34, excel_path):
    # Load the data
    df_Gspecs = pd.read_csv(excel_path) if excel_path.endswith('.csv') else pd.read_excel(excel_path, sheet_name='Gear Specs', index_col=0)



    g1 = Gear(df_Gspecs.loc["Name", "Gear 1"],
              df_Gspecs.loc["N", "Gear 1"],
              df_Gspecs.loc["Module", "Gear 1"],
              df_Gspecs.loc["Face Width", "Gear 1"],
              df_Gspecs.loc["J", "Gear 1"],
              df_Gspecs.loc["Pitch Diameter", "Gear 1"],
              df_Gspecs.loc["HB(<=)", "Gear 1"])

    g2 = Gear(df_Gspecs.loc["Name", "Gear 2"],
              df_Gspecs.loc["N", "Gear 2"],
              df_Gspecs.loc["Module", "Gear 2"],
              df_Gspecs.loc["Face Width", "Gear 2"],
              df_Gspecs.loc["J", "Gear 2"],
              df_Gspecs.loc["Pitch Diameter", "Gear 2"],
              df_Gspecs.loc["HB(<=)", "Gear 2"])

    g3 = Gear(df_Gspecs.loc["Name", "Gear 3"],
              df_Gspecs.loc["N", "Gear 3"],
              df_Gspecs.loc["Module", "Gear 3"],
              df_Gspecs.loc["Face Width", "Gear 3"],
              df_Gspecs.loc["J", "Gear 3"],
              df_Gspecs.loc["Pitch Diameter", "Gear 3"],
              df_Gspecs.loc["HB(<=)", "Gear 3"])

    g4 = Gear(df_Gspecs.loc["Name", "Gear 4"],
              df_Gspecs.loc["N", "Gear 4"],
              df_Gspecs.loc["Module", "Gear 4"],
              df_Gspecs.loc["Face Width", "Gear 4"],
              df_Gspecs.loc["J", "Gear 4"],
              df_Gspecs.loc["Pitch Diameter", "Gear 4"],
              df_Gspecs.loc["HB(<=)", "Gear 4"])

    v = df_Gspecs.loc["v", "All"]
    E = df_Gspecs.loc["E", "All"]

    Wt12 = 2 * (T12 / 1000) * (g2.pdiam / g2.N) # N
    Wt34 = 2 * (T34 / 1000) * (g3.pdiam / g3.N) # N
    omega = 2400 * 19/40  * np.pi / 30 # RPM to rad/s
    pdiam =  g2.pdiam # mm. from 40T bc used in both meshes
    dp = g2.N/g2.pdiam  # mm. = N/pdiam. from 40T bc used in both meshes
    mesh1 = GearMesh(g1, g2, Wt=Wt12, Vt=(np.pi/30) * 2400*(19/40) * pdiam/(2*1000),v=v, E=E)  # Wt=N, Vt=m/s
    mesh2 = GearMesh(g3, g4, Wt=Wt34, Vt=(np.pi/30) * 2400*(19/40)*(20/40) * pdiam/(2*1000), v=v, E=E) # can specify E and v if req)
     # *np.cos(np.deg2rad(20))

    train = GearTrain()

    train.add_mesh(mesh1)
    train.add_mesh(mesh2)

    return train.report()
