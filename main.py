from Shaft_Deflection_FullV5 import run_full_system
from run_gears import run_gear_system
from write_file import write_file


gear_T, specs, segmented = run_full_system('Gear_Train_Specs3.xlsx', 36.58*1000, 20)
gears = run_gear_system(T12=gear_T[1], T34=gear_T[2], excel_path='Gear_Train_Specs3.xlsx')

write_file(specs, segmented, gears, output_path="Gear_Train_Results.xlsx")

