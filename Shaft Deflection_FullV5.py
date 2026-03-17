# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:34:29 2026

@author: Matthew Le
"""

import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import SingularityFunction as SF
from datetime import datetime
import os

def universal_shaft_solver(L, diameters, step_positions, loads, brg1_pos, brg2_pos, E_val=207000):
    x = sp.Symbol('x')
    
    # Calculate I for each segment
    I_vals = [(sp.pi * d**4) / 64 for d in diameters]
    
    # Build Piecewise conditions dynamically
    conditions = []
    if not step_positions:
        Ix = I_vals[0]
    else:
        for i in range(len(step_positions)):
            conditions.append((I_vals[i], x < step_positions[i]))
        conditions.append((I_vals[-1], True)) # Final segment
        Ix = sp.Piecewise(*conditions)

    # Reaction Forces
    Rv1, Rv2 = sp.symbols('Rv1 Rv2')
    qv = Rv1*SF(x, brg1_pos, -1) + Rv2*SF(x, brg2_pos, -1)
    
    for pos, force in loads:
        qv -= force * SF(x, pos, -1) 

    Vv = sp.integrate(qv, x)
    Mv = sp.integrate(Vv, x)
    
    # Solve reactions
    reacts = sp.solve([Vv.subs(x, brg2_pos + 0.1), Mv.subs(x, brg2_pos + 0.1)], [Rv1, Rv2])
    
    V_final, M_final = Vv.subs(reacts), Mv.subs(reacts)
    M_pw = M_final.rewrite(sp.Piecewise)
    
    # Deflection
    C1, C2 = sp.symbols('C1 C2')
    theta_expr = sp.integrate(M_pw / (E_val * Ix), x).doit() + C1
    y_expr = sp.integrate(theta_expr, x).doit() + C2
    consts = sp.solve([y_expr.subs(x, brg1_pos), y_expr.subs(x, brg2_pos)], [C1, C2])
    
    # --- ADD THIS LINE TO EXTRACT/PRINT THE VALUES ---
    print(f"Integration constants for current shaft: C1 = {consts[C1]}, C2 = {consts[C2]}")
    
    return (sp.lambdify(x, V_final.rewrite(sp.Piecewise), 'numpy'),
            sp.lambdify(x, M_pw, 'numpy'),
            sp.lambdify(x, theta_expr.subs(consts).rewrite(sp.Piecewise), 'numpy'),
            sp.lambdify(x, y_expr.subs(consts).rewrite(sp.Piecewise), 'numpy'),
            reacts)


def plot_fbd_dynamic(s_id, L, diameters, step_positions, loads, reactions, brg1_pos, brg2_pos):
    """Generates an FBD with collision-avoidance, bold labeling, and staggered axial dimensioning."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. Handle Reactions
    rv1_val, rv2_val = 0.0, 0.0
    if isinstance(reactions, list) and len(reactions) > 0: reactions = reactions[0]
    if isinstance(reactions, dict):
        rv1_val = float(reactions.get(sp.Symbol('Rv1'), 0))
        rv2_val = float(reactions.get(sp.Symbol('Rv2'), 0))

    # 2. Draw Shaft
    ax.hlines(0, 0, L, colors='black', linewidth=4, zorder=2)
    
    # 3. Step Indicator (Only plot if the shaft is stepped)
    if len(diameters) > 1:
        for i, pos in enumerate(step_positions):
            # Ensure the step is actually within the shaft length
            if 0 < pos < L:
                ax.axvline(x=pos, color='orange', linestyle='--', alpha=0.6)
                d_before = diameters[i] if i < len(diameters) else 0
                d_after = diameters[i+1] if i + 1 < len(diameters) else 0
                label_text = f'{d_before:.0f}→{d_after:.0f}mm'
                ax.text(pos, 2.8, label_text, ha='center', fontsize=9, fontweight='bold', color='orange', 
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 3.5. Add Dimension Arrows for Steps from Left (Origin)
    # Start below the gear dimensions
    y_base = -2.5 - (len(loads) * 0.8) - 0.5 
    # Track the lowest Y point reached by steps to place reaction labels safely
    y_lowest_step = y_base
    
    for i, pos in enumerate(step_positions):
        if 0 < pos < L:
            y_step_dim = y_base - (i * 0.6) 
            y_lowest_step = y_step_dim # Update tracker
            
            # Draw dimension line
            ax.annotate('', xy=(0, y_step_dim), xytext=(pos, y_step_dim),
                        arrowprops=dict(arrowstyle='<->', color='orange', linewidth=1.5, linestyle='--'))
            # Add text label
            ax.text(pos / 2, y_step_dim + 0.2, f'{pos:.0f}mm', ha='center', 
                    fontsize=8, fontweight='bold', color='orange')
            
    # 4. Plot Bearings and Direction Arrows
    bearings = [(brg1_pos, rv1_val, 'Rv1'), (brg2_pos, rv2_val, 'Rv2')]
    # Place reaction labels safely below the lowest step dimension
    y_reaction_labels = y_lowest_step - 0.8 
    
    for pos, val, label in bearings:
        # Plot Support
        ax.plot(pos, 0, 'b^', markersize=12, zorder=5)
        
        # Plot Direction Arrow
        if abs(val) > 0.1:
            arrow_len = 1.0
            dy = arrow_len if val > 0 else -arrow_len
            start_y = -1.5 if val > 0 else 1.5
            ax.arrow(pos, start_y, 0, dy, head_width=2, head_length=0.4, fc='blue', ec='blue', zorder=6)

        # Plot Reaction Label (Now using dynamic Y position)
        ax.annotate(f'{label}: {abs(val):.1f}N', xy=(pos, 0), xytext=(pos, y_reaction_labels),
                    ha='center', fontsize=10, fontweight='bold', arrowprops=dict(arrowstyle='->'))

    # Update axis limits to ensure the new deeper labels are visible
        ax.set_ylim(y_reaction_labels - 2.0, 4.0)

    # 5. Plot Gear Loads & Staggered Axial Dimensions
    for i, (pos, force) in enumerate(loads):
        ax.axvline(x=pos, color='gray', linestyle=':', alpha=0.4)
        
        # Gear label
        y_label = 2.0 if i % 2 == 0 else 2.5
        ax.text(pos, y_label, f'G{i+1}', ha='center', fontsize=10, fontweight='bold')
        
        # Force vector
        color = 'red' if force > 0 else 'green'
        ax.annotate(f'{abs(force):.1f}N', xy=(pos, 0), xytext=(pos, 1.5 if force > 0 else -1.5),
                    arrowprops=dict(facecolor=color, shrink=0.05), ha='center', fontsize=9, fontweight='bold')
        ax.plot(pos, 0, 'ko', markersize=8, zorder=4)
        
        # Axial Dimension Arrow
        y_dim = -2.5 - (i * 0.8)
        ax.annotate('', xy=(0, y_dim), xytext=(pos, y_dim),
                    arrowprops=dict(arrowstyle='<->', color='gray', linewidth=1.5))
        ax.text(pos / 2, y_dim - 0.4, f'{pos:.0f}mm', ha='center', fontsize=9, fontweight='bold', color='gray')
        # 6. Add X-Axis Label
        ax.set_xlabel("Shaft Position (mm)", fontsize=12, fontweight='bold')
        
        # 7. Add Coordinate System (Top Right)
        # This plots a small arrow set at the top right of the figure
        ax.annotate('', xy=(L + 15, 3.5), xytext=(L + 15, 2.5),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ax.annotate('', xy=(L + 20, 3.0), xytext=(L + 10, 3.0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ax.text(L + 16, 3.6, 'y', fontsize=12, fontweight='bold')
        ax.text(L + 21, 2.6, 'x', fontsize=12, fontweight='bold')
        
    ax.set_title(f"Free Body Diagram: {s_id} Shaft", fontsize=16, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(-20, L + 20)
    ax.set_ylim(-6.5, 4.0)
    plt.tight_layout()
    plt.show()

def validate_geometry(df):
    """Checks only for hard logical errors that break the physics/math."""
    errors = []
    for idx, row in df.iterrows():
        # Logical: Gear must be on the shaft
        if row['Gear1_Pos'] > row['L_val'] or row['Gear1_Pos'] < 0 or row['Gear2_Pos'] > row['L_val'] or row['Gear2_Pos'] < 0:
            errors.append(f"❌ {row['Shaft_ID']}: Gear 1 or Gear 2 is outside shaft bounds.")
            
        # Mathematical: Bearings must be distinct
        if row['Brg1_Pos'] == row['Brg2_Pos']:
            errors.append(f"❌ {row['Shaft_ID']}: Bearings are at the same spot (Solver will crash).")

    if errors:
        print("\n--- CRITICAL GEOMETRY ERRORS ---")
        print("\n".join(errors))
        return False
    return True

def run_full_system(excel_path, T_initial, phi_deg):
    """Main execution loop with CSV export and ALL PLOTS."""
    # Load the data
    df_specs = pd.read_csv(excel_path) if excel_path.endswith('.csv') else pd.read_excel(excel_path)
    
    #Validation checker
    if not validate_geometry(df_specs):
        return
    
    # Filter only for the valid shafts to ignore annotation notes
    valid_shafts = ['Input', 'Lay', 'Output']
    df_specs = df_specs[df_specs['Shaft_ID'].isin(valid_shafts)].copy()
    
    # Ensure all numeric columns are forced to numbers (fixes "Object" type issues)
    cols_to_num = ['L_val', 'Gear1_Pos', 'Gear1_D', 'Gear2_Pos', 'Gear2_D', 'Brg1_Pos', 'Brg2_Pos']
    
    for col in cols_to_num:
        if col in df_specs.columns:
            df_specs[col] = pd.to_numeric(df_specs[col], errors='coerce')
        else:
            # Set defaults only for missing simple numeric columns
            if col == 'Brg1_Pos': df_specs[col] = 0.0
            if col == 'Brg2_Pos': df_specs[col] = df_specs['L_val']
    
    phi = np.deg2rad(phi_deg)
    current_torque = T_initial
    mesh_forces = {}
    all_data, summary_data, force_report = [], [], []
    date_str = datetime.now().strftime("%Y-%m-%d")
    results_map = {}
    torque_map = {}
    all_results = []
    for i, row in df_specs.iterrows():
        s_id = row['Shaft_ID']
        shaft_loads = []

        if s_id == 'Input':
            Ft = 2 * current_torque / row['Gear1_D']
            Fr = Ft * np.tan(phi)
            mesh_forces['In_to_Lay'] = (Fr, Ft)
            shaft_loads.append((row['Gear1_Pos'], Fr))
            force_report.append({'Shaft': s_id, 'Gear': 'G1', 'Role': 'Driver', 'Ft_N': Ft, 'Fr_N': Fr})
        elif s_id == 'Lay':
            prev_d = df_specs[df_specs['Shaft_ID'] == 'Input']['Gear1_D'].values[0]
            current_torque *= (row['Gear1_D'] / prev_d)
            Fr_in, Ft_in = mesh_forces['In_to_Lay']
            shaft_loads.append((row['Gear1_Pos'], -Fr_in))
            force_report.append({'Shaft': s_id, 'Gear': 'G1', 'Role': 'Driven', 'Ft_N': Ft_in, 'Fr_N': Fr_in})
            if pd.notna(row['Gear2_D']):
                Ft_out = 2 * current_torque / row['Gear2_D']
                Fr_out = Ft_out * np.tan(phi)
                mesh_forces['Lay_to_Out'] = (Fr_out, Ft_out)
                shaft_loads.append((row['Gear2_Pos'], Fr_out))
                force_report.append({'Shaft': s_id, 'Gear': 'G2', 'Role': 'Driver', 'Ft_N': Ft_out, 'Fr_N': Fr_out})
        elif s_id == 'Output':
            prev_d = df_specs[df_specs['Shaft_ID'] == 'Lay']['Gear2_D'].values[0]
            current_torque *= (row['Gear1_D'] / prev_d)
            Fr_lay, Ft_lay = mesh_forces['Lay_to_Out']
            shaft_loads.append((row['Gear1_Pos'], -Fr_lay))
            force_report.append({'Shaft': s_id, 'Gear': 'G1', 'Role': 'Driven', 'Ft_N': Ft_lay, 'Fr_N': Fr_lay})
        
        d_list = [float(x.strip()) for x in str(row['diameters']).replace('"', '').split(',')]
        s_list = [float(x.strip()) for x in str(row['step_pos']).replace('"', '').split(',')]
        
        # Call the new solver with the lists
        v_f, m_f, t_f, y_f, reacts = universal_shaft_solver(
            row['L_val'], d_list, s_list, shaft_loads, row['Brg1_Pos'], row['Brg2_Pos']
        )
        
        # --- UPDATED: SEGMENT & TRANSITION MOMENT CALCULATION ---
        boundaries = [0] + sorted(s_list) + [row['L_val']]
        
        for j in range(len(d_list)):
            x_start, x_end = boundaries[j], boundaries[j+1]
            
            # 1. Local peak moment in the span
            x_seg = np.linspace(x_start, x_end, 100)
            max_m_seg = np.max(np.abs(m_f(x_seg)))
            
            # 2. NEW: Exact moments at the transition "shoulders"
            # Evaluates m_f at the very start and very end of the diameter segment
            m_at_shoulder_start = abs(m_f(x_start))
            m_at_shoulder_end = abs(m_f(x_end))
            
            # Store a copy of the original row but with segment-specific results
            result_row = row.copy()
            result_row['Segment_D'] = d_list[j]
            result_row['Segment_Range'] = f"{x_start}-{x_end}mm"
            result_row['Max_Moment_Nmm'] = round(max_m_seg, 2)
            
            # NEW COLUMNS for the Excel export
            result_row['Moment_at_Step_Start'] = round(m_at_shoulder_start, 2)
            result_row['Moment_at_Step_End'] = round(m_at_shoulder_end, 2)
            
            result_row['Torque_Nm'] = round(current_torque / 1000, 2)
            all_results.append(result_row)
        
        # 3. Update the Plotting Call
        # Update plot_fbd signature to accept lists and loop through steps for drawing
        plot_fbd_dynamic(s_id, row['L_val'], d_list, s_list, shaft_loads, reacts, row['Brg1_Pos'], row['Brg2_Pos'])
       
        # PLOT 2: Analysis Results
        x_plot = np.linspace(0, row['L_val'], 1000)
        V, M, T, Y = v_f(x_plot), m_f(x_plot), t_f(x_plot), y_f(x_plot)

        fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        fig.suptitle(f"Analysis Results: {s_id} Shaft", fontsize=14, fontweight='bold')
        configs = [(V,'red','Shear (N)'), (M,'blue','Moment (N-mm)'), (T,'green','Slope (rad)'), (Y,'black','Deflect (mm)')]
        for idx, (data, color, label) in enumerate(configs):
            axs[idx].plot(x_plot, data, color=color)
            axs[idx].set_ylabel(label, fontweight='bold')
            axs[idx].grid(True, alpha=0.3)
            if idx == 3: 
                axs[idx].invert_yaxis()
                axs[idx].set_xlabel("Shaft Position (mm)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Data collection
        all_data.append(pd.DataFrame({'Shaft': s_id, 'Pos': x_plot, 'V': V, 'M': M, 'T': T, 'Y': Y}))
        summary_data.append({'Shaft': s_id, 'Max_V': np.max(np.abs(V)), 'Max_M': np.max(np.abs(M)), 
                             'Max_T': np.max(np.abs(T)), 'Max_Y': np.max(np.abs(Y))})
        
        max_m = np.max(np.abs(M))
        
        # Store for Excel mapping (results_map should be initialized as {} before the loop)
        results_map[s_id] = round(max_m, 2)
        torque_map[s_id] = round(current_torque, 2)

    # EXPORT Logic
    iteration = 1
    while True:
        fname = f"GearTrain_Master_{date_str}_iter{iteration}.csv"
        if not os.path.exists(fname): break
        iteration += 1

    with open(fname, 'w') as f:
        f.write("--- GEAR FORCE ANALYSIS ---\n")
        pd.DataFrame(force_report).to_csv(f, index=False)
        f.write("\n--- SUMMARY MAX VALUES ---\n")
        pd.DataFrame(summary_data).to_csv(f, index=False)
        f.write("\n--- DISTRIBUTED DATA ---\n")
        pd.concat(all_data).to_csv(f, index=False)
    print(f"Analysis complete. Results saved to: {fname}")
    # Map the results back to a new column and save to a NEW file to avoid permission errors
  # --- UPDATED EXPORT LOGIC ---
    output_path = "Gear_Train_Results.xlsx"
    
    # 1. Prepare the Summary DataFrame (Original Format)
    # This adds the overall Max Moment and Torque back to your original spec sheet
    df_specs['Max_Moment_Nmm'] = df_specs['Shaft_ID'].map(results_map)
    df_specs['Torque_Nm'] = df_specs['Shaft_ID'].map(torque_map)
    
    # 2. Prepare the Segmented DataFrame (New Printout)
    df_segmented = pd.DataFrame(all_results)
    
    try:
        with pd.ExcelWriter(output_path) as writer:
            # Sheet 1: Original format with overall Max Moment/Torque per shaft
            df_specs.to_excel(writer, sheet_name='Shaft_Summary', index=False)
            
            # Sheet 2: Detailed breakdown for every diameter segment
            df_segmented.to_excel(writer, sheet_name='Segment_Details', index=False)
            
        print(f"\nSuccess! Both summary and segment results saved to: {output_path}")
        
    except PermissionError:
        print(f"\n Error: Please close '{output_path}' and run the script again.")

# --- EXECUTE ---
run_full_system('Gear_Train_Specs3.xlsx', 36.58*1000, 20)