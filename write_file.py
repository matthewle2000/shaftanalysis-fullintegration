import pandas as pd

def write_file(df_specs, df_segmented, df_gears, output_path):
    try:
        with pd.ExcelWriter(output_path, if_sheet_exists='replace', mode='a') as writer:
            # Sheet 1: Original format with overall Max Moment/Torque per shaft
            df_specs.to_excel(writer, sheet_name='Shaft_Summary', index=False)

            # Sheet 2: Detailed breakdown for every diameter segment
            df_segmented.to_excel(writer, sheet_name='Segment_Details', index=False)

            # Sheet 3: Gear SF's and max allowable Vt vs max curent Vt
            df_gears.to_excel(writer, sheet_name='Gear Safety Factors', index=False)

        print(f"\nSuccess! Summary, segment and gear results saved to: {output_path}")

    except PermissionError:
        print(f"\n Error: Please close '{output_path}' and run the script again.")
