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
from IPython.display import display
import os

def universal_shaft_solver(L, diameters, step_positions, loads, brg1_pos, brg2_pos, E_val=207000):
    x = sp.Symbol('x')
    Ix_sym = sp.Symbol('I(x)') # Create the symbolic placeholder
    E = sp.Symbol('E')

    # --- 1. Piecewise I(x) for the final report ---
    I_vals = [(sp.pi * d**4) / 64 for d in diameters]
    conditions = []
    if not step_positions:
        Ix_pw = I_vals[0]
    else:
        for i in range(len(step_positions)):
            conditions.append((I_vals[i], x < step_positions[i]))
        conditions.append((I_vals[-1], True))
        Ix_pw = sp.Piecewise(*conditions)

    # --- 2. Solve Reactions (Same as before) ---
    R1, R2 = sp.symbols('R1 R2')
    q = R1*SF(x, brg1_pos, -1) + R2*SF(x, brg2_pos, -1)
    for pos, force in loads:
        q -= force * SF(x, pos, -1) 

    V_expr = sp.integrate(q, x)
    M_expr = sp.integrate(V_expr, x)
    reacts = sp.solve([V_expr.subs(x, L + 0.1), M_expr.subs(x, L + 0.1)], [R1, R2])
    
    V_final = V_expr.subs(reacts)
    M_final = M_expr.subs(reacts)

    # --- 3. Symbolic Deflection (keeping I(x) as a symbol) ---
    C1, C2 = sp.symbols('C1 C2')
    # We use M_final (Singularity form) for the LaTeX output to keep it pretty
    theta_sym = sp.integrate(M_final / (E * Ix_sym), x) + C1
    y_sym = sp.integrate(theta_sym, x) + C2

    # Note: For the numerical solver (lambdify), we still need the actual Piecewise math
    M_pw = M_final.rewrite(sp.Piecewise)
    theta_numeric = sp.integrate(M_pw / (E_val * Ix_pw), x).doit() + C1
    y_numeric = sp.integrate(theta_numeric, x).doit() + C2
    consts = sp.solve([y_numeric.subs(x, brg1_pos), y_numeric.subs(x, brg2_pos)], [C1, C2])

    q_final = q.subs(reacts)
    theta_solved_sym = theta_sym.subs(consts)
    y_solved_sym = y_sym.subs(consts)

    return (sp.lambdify(x, V_final.rewrite(sp.Piecewise), 'numpy'),
            sp.lambdify(x, M_pw, 'numpy'),
            sp.lambdify(x, theta_numeric.subs(consts).rewrite(sp.Piecewise), 'numpy'),
            sp.lambdify(x, y_numeric.subs(consts).rewrite(sp.Piecewise), 'numpy'),
            reacts,q_final,
            V_final, M_final, theta_solved_sym, y_solved_sym, Ix_pw)


def plot_fbd_dynamic(s_id, L, diameters, step_positions, loads, reactions, brg1_pos, brg2_pos, plane_name="Vertical"):
    """Generates an FBD for a specific plane with absolute value labels and coordinate-specific naming."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. Handle Reactions (Generic R1, R2 symbols)
    r1_val, r2_val = 0.0, 0.0
    R1_sym, R2_sym = sp.symbols('R1 R2')
    
    if isinstance(reactions, list) and len(reactions) > 0: reactions = reactions[0]
    if isinstance(reactions, dict):
        r1_val = float(reactions.get(R1_sym, 0))
        r2_val = float(reactions.get(R2_sym, 0))

    # 2. Draw Shaft
    ax.hlines(0, 0, L, colors='black', linewidth=4, zorder=2)
    
    # 3. Step Indicator
    if len(diameters) > 1:
        for i, pos in enumerate(step_positions):
            if 0 < pos < L:
                ax.axvline(x=pos, color='orange', linestyle='--', alpha=0.6)
                d_before = diameters[i] if i < len(diameters) else 0
                d_after = diameters[i+1] if i + 1 < len(diameters) else 0
                label_text = f'{d_before:.0f}→{d_after:.0f}mm'
                ax.text(pos, 2.8, label_text, ha='center', fontsize=9, fontweight='bold', color='orange', 
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 3.5. Staggered Step Dimensions
    y_base = -2.5 - (len(loads) * 0.8) - 0.5 
    y_lowest_step = y_base
    for i, pos in enumerate(step_positions):
        if 0 < pos < L:
            y_step_dim = y_base - (i * 0.6) 
            y_lowest_step = y_step_dim
            ax.annotate('', xy=(0, y_step_dim), xytext=(pos, y_step_dim),
                        arrowprops=dict(arrowstyle='<->', color='orange', linewidth=1.5, linestyle='--'))
            ax.text(pos / 2, y_step_dim + 0.2, f'{pos:.0f}mm', ha='center', 
                    fontsize=8, fontweight='bold', color='orange')
            
    # 4. Plot Bearings and Direction Arrows
    # UPDATED: Use explicit y or z axis labels instead of generic V/H
    coord_axis = 'y' if plane_name == "Vertical" else 'z'
    bearings = [(brg1_pos, r1_val, f'R1_{coord_axis}'), (brg2_pos, r2_val, f'R2_{coord_axis}')]
    y_reaction_labels = y_lowest_step - 0.8 
    
    for pos, val, label in bearings:
        ax.plot(pos, 0, 'b^', markersize=12, zorder=5)
        if abs(val) > 0.1:
            arrow_len = 1.0
            dy = arrow_len if val > 0 else -arrow_len
            # Arrows still point in the correct direction mathematically
            start_y = -1.5 if val > 0 else 1.5
            ax.arrow(pos, start_y, 0, dy, head_width=2, head_length=0.4, fc='blue', ec='blue', zorder=6)
        
        # UPDATED: Use abs(val) to remove signs from labels
        ax.annotate(f'{label}: {abs(val):.1f}N', xy=(pos, 0), xytext=(pos, y_reaction_labels),
                    ha='center', fontsize=10, fontweight='bold', arrowprops=dict(arrowstyle='->'))

    # 5. Plot Gear Loads & Staggered Axial Dimensions
    for i, (pos, force) in enumerate(loads):
        ax.axvline(x=pos, color='gray', linestyle=':', alpha=0.4)
        y_label = 2.0 if i % 2 == 0 else 2.5
        ax.text(pos, y_label, f'G{i+1}', ha='center', fontsize=10, fontweight='bold')
        
        color = 'red' if force > 0 else 'green'
        # UPDATED: Use abs(force) to remove signs from labels
        ax.annotate(f'{abs(force):.1f}N', xy=(pos, 0), xytext=(pos, 1.5 if force > 0 else -1.5),
                    arrowprops=dict(facecolor=color, shrink=0.05), ha='center', fontsize=9, fontweight='bold')
        ax.plot(pos, 0, 'ko', markersize=8, zorder=4)
        
        y_dim = -2.5 - (i * 0.8)
        ax.annotate('', xy=(0, y_dim), xytext=(pos, y_dim),
                    arrowprops=dict(arrowstyle='<->', color='gray', linewidth=1.5))
        ax.text(pos / 2, y_dim - 0.4, f'{pos:.0f}mm', ha='center', fontsize=9, fontweight='bold', color='gray')

    # 6. Formatting & Coordinate System
    ax.set_xlabel("Shaft Position (mm)", fontsize=12, fontweight='bold')
    
    # Coordinate system labels
    ax.annotate('', xy=(L + 15, 3.5), xytext=(L + 15, 2.5), arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(L + 20, 3.0), xytext=(L + 10, 3.0), arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(L + 16, 3.6, coord_axis, fontsize=12, fontweight='bold')
    ax.text(L + 21, 2.6, 'x', fontsize=12, fontweight='bold')
    
    ax.set_title(f"FBD: {s_id} Shaft - {plane_name} Plane", fontsize=16, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(-20, L + 25)
    ax.set_ylim(y_reaction_labels - 2.0, 4.0)
    plt.tight_layout()
    
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
    """Main execution loop with 2-plane (3D) analysis and CSV/Excel export."""
    # Load the data
    df_specs = pd.read_csv(excel_path) if excel_path.endswith('.csv') else pd.read_excel(excel_path, sheet_name='Shaft Specs')
    
    # Validation checker
    if not validate_geometry(df_specs):
        print("Geometry validation failed. Returning empty data to prevent crash.")
        # Returning three empty objects allows 'main.py' to unpack them safely
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    # Filter only for the valid shafts
    valid_shafts = ['Input', 'Lay', 'Output']
    df_specs = df_specs[df_specs['Shaft_ID'].isin(valid_shafts)].copy()
    
    # Ensure numeric columns
    cols_to_num = ['L_val', 'Gear1_Pos', 'Gear1_D', 'Gear2_Pos', 'Gear2_D', 'Brg1_Pos', 'Brg2_Pos']
    for col in cols_to_num:
        if col in df_specs.columns:
            df_specs[col] = pd.to_numeric(df_specs[col], errors='coerce')
    
    phi = np.deg2rad(phi_deg)
    current_torque = T_initial
    mesh_forces = {}
    all_data, summary_data, force_report = [], [], []
    date_str = datetime.now().strftime("%Y-%m-%d")
    results_map = {}
    torque_map = {}
    r1y_map, r1z_map = {}, {}
    r2y_map, r2z_map = {}, {}
    all_results = []

    for i, row in df_specs.iterrows():
        s_id = row['Shaft_ID']
        loads_v = [] # Vertical Plane (Radial Forces - Fr)
        loads_h = [] # Horizontal Plane (Tangential Forces - Ft)

        # --- DUAL PLANE FORCE LOGIC ---
        if s_id == 'Input':
            Ft = 2 * current_torque / row['Gear1_D']
            Fr = Ft * np.tan(phi)
            mesh_forces['In_to_Lay'] = (Fr, Ft)
            loads_v.append((row['Gear1_Pos'], Fr))
            loads_h.append((row['Gear1_Pos'], Ft))
            force_report.append({'Shaft': s_id, 'Gear': 'G1', 'Role': 'Driver', 'Ft_N': Ft, 'Fr_N': Fr})
        
        elif s_id == 'Lay':
            prev_d = df_specs[df_specs['Shaft_ID'] == 'Input']['Gear1_D'].values[0]
            current_torque *= (row['Gear1_D'] / prev_d)
            Fr_in, Ft_in = mesh_forces['In_to_Lay']
            # Reactions are opposite to the driving force
            loads_v.append((row['Gear1_Pos'], -Fr_in))
            loads_h.append((row['Gear1_Pos'], -Ft_in))
            force_report.append({'Shaft': s_id, 'Gear': 'G1', 'Role': 'Driven', 'Ft_N': Ft_in, 'Fr_N': Fr_in})
            
            if pd.notna(row['Gear2_D']):
                Ft_out = 2 * current_torque / row['Gear2_D']
                Fr_out = Ft_out * np.tan(phi)
                mesh_forces['Lay_to_Out'] = (Fr_out, Ft_out)
                loads_v.append((row['Gear2_Pos'], Fr_out))
                loads_h.append((row['Gear2_Pos'], Ft_out))
                force_report.append({'Shaft': s_id, 'Gear': 'G2', 'Role': 'Driver', 'Ft_N': Ft_out, 'Fr_N': Fr_out})
        
        elif s_id == 'Output':
            prev_d = df_specs[df_specs['Shaft_ID'] == 'Lay']['Gear2_D'].values[0]
            current_torque *= (row['Gear1_D'] / prev_d)
            Fr_lay, Ft_lay = mesh_forces['Lay_to_Out']
            loads_v.append((row['Gear1_Pos'], -Fr_lay))
            loads_h.append((row['Gear1_Pos'], -Ft_lay))
            force_report.append({'Shaft': s_id, 'Gear': 'G1', 'Role': 'Driven', 'Ft_N': Ft_lay, 'Fr_N': Fr_lay})
        
        # --- SOLVER EXECUTION (RUN TWICE) ---
        d_list = [float(x.strip()) for x in str(row['diameters']).replace('"', '').split(',')]
        s_list = [float(x.strip()) for x in str(row['step_pos']).replace('"', '').split(',')]
        
        # Call the existing solver for Vertical and Horizontal planes
        # 1. Vertical Plane Analysis (Added Qv_s)
        vv_f, mv_f, tv_f, yv_f, reacts_v, Qv_s, Vv_s, Mv_s, Tv_solved, Yv_solved, I_pw = universal_shaft_solver(
            row['L_val'], d_list, s_list, loads_v, row['Brg1_Pos'], row['Brg2_Pos']
        )

        # 2. Horizontal Plane Analysis (Added Qh_s)
        vh_f, mh_f, th_f, yh_f, reacts_h, Qh_s, Vh_s, Mh_s, Th_solved, Yh_solved, _ = universal_shaft_solver(
            row['L_val'], d_list, s_list, loads_h, row['Brg1_Pos'], row['Brg2_Pos']
        )
       
       
        # --- COMBINE INTO RESULTANTS ---
        x_plot = np.linspace(0, row['L_val'], 1000)
        
        # Calculate magnitudes using Pythagoras
        V_res = np.sqrt(vv_f(x_plot)**2 + vh_f(x_plot)**2)
        M_res = np.sqrt(mv_f(x_plot)**2 + mh_f(x_plot)**2)
        T_res = np.sqrt(tv_f(x_plot)**2 + th_f(x_plot)**2)
        Y_res = np.sqrt(yv_f(x_plot)**2 + yh_f(x_plot)**2)
        
        
        
        print(f"Generating Rounded Equation Previews for {s_id}...")
        
        def format_constants_only(expr, prec=3):
            from sympy import Float, Number, Integer, Pow, Piecewise
            
            def to_int_if_whole(n):
                if not isinstance(n, Number):
                    return n
                try:
                    val = float(n)
                    return int(val) if val.is_integer() else n
                except:
                    return n
        
            def clean_num(n):
                if isinstance(n, Integer): return n
                val = float(n)
                if val == 0: return 0
                if 0 < abs(val) < 0.001:
                    return Float(val, prec) 
                return round(val, 2)
        
            # 1. Handle Piecewise conditions (x < 10.0 -> x < 10)
            # We do this first manually to avoid the replace() error
            if expr.has(Piecewise):
                new_pw_args = []
                for val, cond in expr.args:
                    # Use xreplace on the condition specifically to target numbers
                    new_cond = cond.xreplace({n: to_int_if_whole(n) for n in cond.atoms(Number)})
                    new_pw_args.append((val, new_cond))
                expr = Piecewise(*new_pw_args)
        
            # 2. Target SingularityFunction exponents and offsets
            expr = expr.replace(SF, lambda x_v, a, n: SF(x_v, to_int_if_whole(a), to_int_if_whole(n)))
            
            # 3. Target standard Powers (x**3.0 -> x**3)
            expr = expr.replace(Pow, lambda b, e: Pow(b, to_int_if_whole(e)))
        
            # 4. Final pass for all other numbers (Scientific/Rounding)
            return expr.xreplace({n: clean_num(n) for n in expr.atoms(Number)})
        # Define the numerical E to get the final constant values
        E_num = 207000 

        # Prepare the expressions by plugging in E and rounding/formatting
        Tv_disp = format_constants_only(Tv_solved.subs(sp.Symbol('E'), E_num))
        Yv_disp = format_constants_only(Yv_solved.subs(sp.Symbol('E'), E_num))
        Th_disp = format_constants_only(Th_solved.subs(sp.Symbol('E'), E_num))
        Yh_disp = format_constants_only(Yh_solved.subs(sp.Symbol('E'), E_num))

        # 1. Geometry (Updated to use the better formatter)
        sp.preview(sp.Eq(sp.Symbol('I(x)'), format_constants_only(I_pw)), viewer='file', 
                   filename=f'{s_id}_I_pw.png', dvioptions=['-D', '300'])

        # 2. Vertical Plane (y)
        # Change round_expr to format_constants_only here
        sp.preview(sp.Eq(sp.Symbol('V_v(x)'), format_constants_only(Vv_s)), viewer='file', 
                   filename=f'{s_id}_V_v.png', dvioptions=['-D', '300'])
        
        sp.preview(sp.Eq(sp.Symbol('M_v(x)'), format_constants_only(Mv_s)), viewer='file', 
                   filename=f'{s_id}_M_v.png', dvioptions=['-D', '300'])
        
        sp.preview(sp.Eq(sp.Symbol(r'\theta_v(x)'), Tv_disp), viewer='file', 
                   filename=f'{s_id}_Theta_v.png', dvioptions=['-D', '300'])
        
        sp.preview(sp.Eq(sp.Symbol('y_v(x)'), Yv_disp), viewer='file', 
                   filename=f'{s_id}_y_v.png', dvioptions=['-D', '600'])
        # Loading Function Previews
        sp.preview(sp.Eq(sp.Symbol('q_v(x)'), format_constants_only(Qv_s)), viewer='file', 
                   filename=f'{s_id}_q_v.png', dvioptions=['-D', '300'])
        
        # 3. Horizontal Plane (z)
        # Change round_expr to format_constants_only here
        sp.preview(sp.Eq(sp.Symbol('V_h(x)'), format_constants_only(Vh_s)), viewer='file', 
                   filename=f'{s_id}_V_h.png', dvioptions=['-D', '300'])
        
        sp.preview(sp.Eq(sp.Symbol('M_h(x)'), format_constants_only(Mh_s)), viewer='file', 
                   filename=f'{s_id}_M_h.png', dvioptions=['-D', '300'])
        
        sp.preview(sp.Eq(sp.Symbol(r'\theta_h(x)'), Th_disp), viewer='file', 
                   filename=f'{s_id}_Theta_h.png', dvioptions=['-D', '300'])
        
        sp.preview(sp.Eq(sp.Symbol('y_h(x)'), Yh_disp), viewer='file', 
                   filename=f'{s_id}_y_h.png', dvioptions=['-D', '600'])
        
        sp.preview(sp.Eq(sp.Symbol('q_h(x)'), format_constants_only(Qh_s)), viewer='file', 
                   filename=f'{s_id}_q_h.png', dvioptions=['-D', '300'])
        
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f"Resultant (3D) Analysis: {s_id} Shaft", fontsize=14, fontweight='bold')
        
        res_configs = [
            (V_res, 'purple', 'Resultant Shear (N)'),
            (M_res, 'blue', 'Resultant Moment (N-mm)'),
            (T_res, 'green', 'Resultant Slope (rad)'),
            (Y_res, 'black', 'Resultant Deflection (mm)')
        ]
        
        for idx, (data, color, label) in enumerate(res_configs):
            axs[idx].plot(x_plot, data, color=color, linewidth=2)
            axs[idx].set_ylabel(label, fontweight='bold')
            axs[idx].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # --- SEGMENT ANALYSIS (RESULTANT) ---
        boundaries = [0] + sorted(s_list) + [row['L_val']]
        for j in range(len(d_list)):
            x_start, x_end = boundaries[j], boundaries[j+1]
            x_seg = np.linspace(x_start, x_end, 100)
            
            # Use Resultant Moment for all reporting
            mv_seg, mh_seg = mv_f(x_seg), mh_f(x_seg)
            m_res_seg = np.sqrt(mv_seg**2 + mh_seg**2)
            
            result_row = row.copy()
            result_row['Segment_D'] = d_list[j]
            result_row['Segment_Range'] = f"{x_start}-{x_end}mm"
            result_row['Max_Moment_Nmm'] = round(np.max(m_res_seg), 2)
            result_row['Moment_at_Step_Start'] = round(np.sqrt(mv_f(x_start)**2 + mh_f(x_start)**2), 2)
            result_row['Moment_at_Step_End'] = round(np.sqrt(mv_f(x_end)**2 + mh_f(x_end)**2), 2)
            result_row['Torque_Nm'] = round(current_torque / 1000, 2)
            all_results.append(result_row)

        # --- REACTION RESULTANTS FOR PLOTTING ---
        R1_sym, R2_sym = sp.symbols('R1 R2')
        res_reacts = {
            R1_sym: np.sqrt(float(reacts_v[R1_sym])**2 + float(reacts_h[R1_sym])**2),
            R2_sym: np.sqrt(float(reacts_v[R2_sym])**2 + float(reacts_h[R2_sym])**2)
        }
        
        # Resultant load magnitudes for the FBD
        loads_res = [(pos, np.sqrt(fv**2 + fh**2)) for (pos, fv), (_, fh) in zip(loads_v, loads_h)]

        # Plot Vertical (Radial) Plane
        plot_fbd_dynamic(s_id, row['L_val'], d_list, s_list, loads_v, reacts_v, 
                         row['Brg1_Pos'], row['Brg2_Pos'], plane_name="Vertical")
        
        # Plot Horizontal (Tangential) Plane
        plot_fbd_dynamic(s_id, row['L_val'], d_list, s_list, loads_h, reacts_h, 
                         row['Brg1_Pos'], row['Brg2_Pos'], plane_name="Horizontal")
        
        # Data collection for summary
        all_data.append(pd.DataFrame({'Shaft': s_id, 'Pos': x_plot, 'V': V_res, 'M': M_res, 'T': T_res, 'Y': Y_res}))
        summary_data.append({'Shaft': s_id, 'Max_V': np.max(V_res), 'Max_M': np.max(M_res), 
                             'Max_T': np.max(T_res), 'Max_Y': np.max(Y_res)})
        
        results_map[s_id] = round(np.max(M_res), 2)
        torque_map[s_id] = round(current_torque, 2)
        
        # --- CAPTURE INDIVIDUAL REACTION COMPONENTS ---
        R1_sym, R2_sym = sp.symbols('R1 R2')
        
        # Vertical plane reactions (y-components)
        r1y_map[s_id] = round(float(reacts_v.get(R1_sym, 0)), 2)
        r2y_map[s_id] = round(float(reacts_v.get(R2_sym, 0)), 2)
        
        # Horizontal plane reactions (z-components)
        r1z_map[s_id] = round(float(reacts_h.get(R1_sym, 0)), 2)
        r2z_map[s_id] = round(float(reacts_h.get(R2_sym, 0)), 2)

    # --- EXPORT LOGIC ---
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
        f.write("\n--- DISTRIBUTED RESULTANT DATA ---\n")
        pd.concat(all_data).to_csv(f, index=False)

    # Map all reaction components to the summary dataframe
    df_specs['R1_y_N'] = df_specs['Shaft_ID'].map(r1y_map)
    df_specs['R2_y_N'] = df_specs['Shaft_ID'].map(r2y_map)
    df_specs['R1_z_N'] = df_specs['Shaft_ID'].map(r1z_map)
    df_specs['R2_z_N'] = df_specs['Shaft_ID'].map(r2z_map)
    
    df_specs['Max_Moment_Nmm'] = df_specs['Shaft_ID'].map(results_map)
    df_specs['Torque_Nm'] = df_specs['Shaft_ID'].map(torque_map)
    
    df_segmented = pd.DataFrame(all_results)

    output_path = "Gear_Train_Results.xlsx"
    try:
        # Check if the file exists to decide whether to append or create new
        if os.path.exists(output_path):
            # 'a' mode appends to existing file; 'replace' overwrites the specific sheet if it exists
            with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df_specs.to_excel(writer, sheet_name='Shaft_Summary', index=False)
                df_segmented.to_excel(writer, sheet_name='Segment_Details', index=False)
        else:
            # Create a new file if it doesn't exist yet
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df_specs.to_excel(writer, sheet_name='Shaft_Summary', index=False)
                df_segmented.to_excel(writer, sheet_name='Segment_Details', index=False)
                
        print(f"Success! Updated Excel: {output_path}")
        print(f"CSV Report: {fname}")
        
    except PermissionError:
        print(f"Error: The file '{output_path}' is open. Please close it and retry.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    return df_specs.set_index('Shaft_ID')['Torque_Nm'], df_specs, df_segmented