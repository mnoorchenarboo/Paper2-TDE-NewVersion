"""
SHAP & Seaborn Heatmap PDF Generator - FIXED
Generate both SHAP and Seaborn heatmaps from XAI results
"""

import sqlite3
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import shap

# ============================================================================
# CONFIGURATION
# ============================================================================
XAI_DB = "databases/xai_results.db"
RESULTS_DIR = "results/visualization/heatmap"
DEFAULT_FONT_SIZE = 18
DEFAULT_SCALE_FACTOR = 10000
DEFAULT_HEATMAP_TYPE = 'seaborn'
DEFAULT_SHOW_NUMBERS = True
DEFAULT_CELL_FONT_SIZE = 12
DEFAULT_SEPARATE_BAR_PLOT = True
DEFAULT_HEATMAP_YLABEL = ''
DEFAULT_BAR_PLOT_TITLE = ''
DEFAULT_DECIMAL_PLACES = 1


def get_available_primary_uses():
    """Get all unique primary uses"""
    conn = sqlite3.connect(XAI_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT primary_use FROM xai_results ORDER BY primary_use")
    uses = [row[0] for row in cursor.fetchall()]
    conn.close()
    return uses


def get_available_options(primary_use):
    """Get available option numbers for primary use"""
    conn = sqlite3.connect(XAI_DB)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT option_number FROM xai_results 
        WHERE primary_use = ? ORDER BY option_number
    """, (primary_use,))
    options = [row[0] for row in cursor.fetchall()]
    conn.close()
    return options


def get_available_models(primary_use, option_number):
    """Get available models for dataset"""
    conn = sqlite3.connect(XAI_DB)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT model_name FROM xai_results 
        WHERE primary_use = ? AND option_number = ?
        ORDER BY model_name
    """, (primary_use, option_number))
    models = [row[0] for row in cursor.fetchall()]
    conn.close()
    return models


def get_available_methods(primary_use, option_number, model_name):
    """Get available XAI methods"""
    conn = sqlite3.connect(XAI_DB)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT xai_method FROM xai_results 
        WHERE primary_use = ? AND option_number = ? AND model_name = ?
        ORDER BY xai_method
    """, (primary_use, option_number, model_name))
    methods = [row[0] for row in cursor.fetchall()]
    conn.close()
    return methods


def get_available_samples(primary_use, option_number, model_name, xai_method):
    """Get available sample indices"""
    conn = sqlite3.connect(XAI_DB)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT sample_idx FROM xai_results 
        WHERE primary_use = ? AND option_number = ? 
        AND model_name = ? AND xai_method = ?
        ORDER BY sample_idx
    """, (primary_use, option_number, model_name, xai_method))
    samples = [row[0] for row in cursor.fetchall()]
    conn.close()
    return samples


def set_plot_font_size(font_size):
    """Set global matplotlib font sizes"""
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': font_size,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'figure.titlesize': font_size,
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'axes.titlepad': 20
    })


def generate_shap_heatmap(shap_vals, sample_data, feature_names, output_path, font_size, show_numbers, scale_factor):
    """Generate SHAP heatmap"""
    n_features = shap_vals.shape[1]
    
    set_plot_font_size(font_size)
    
    shap_exp = shap.Explanation(
        values=shap_vals,
        base_values=np.zeros(shap_vals.shape[0]),
        data=sample_data,
        feature_names=feature_names
    )
    
    fig = plt.figure(figsize=(16, 8))
    shap.plots.heatmap(shap_exp, show=False, max_display=n_features)
    
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    
    scale_label = f'SHAP Value (×{scale_factor:.0f})' if scale_factor != 1 else 'SHAP Value'
    for child_ax in fig.get_axes():
        if child_ax != ax:
            child_ax.set_ylabel(scale_label, fontsize=font_size)
            child_ax.tick_params(labelsize=font_size)
    
    if show_numbers:
        for i in range(shap_vals.shape[0]):
            for j in range(n_features):
                value = shap_vals[i, j]
                ax.text(i + 0.15, n_features - j - 0.85, f'{value:.2f}',
                       ha='center', va='center',
                       color='black',
                       fontsize=max(4, font_size - 11),
                       rotation=90)
    
    plt.tight_layout()
    plt.savefig(str(output_path), format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def save_values_log(shap_vals, feature_names, feature_importance, output_path, xai_method, scale_factor):
    """Save SHAP values and feature importance to log file"""
    n_features = shap_vals.shape[1]
    n_timesteps = shap_vals.shape[0]
    
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importance = feature_importance[sorted_indices]
    sorted_data = shap_vals[:, sorted_indices]
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"SHAP VALUES LOG - {xai_method.upper()}\n")
        f.write(f"Scale Factor: {scale_factor}\n")
        f.write("="*80 + "\n\n")
        
        f.write("HEATMAP VALUES (Features × Time Steps)\n")
        f.write("-"*80 + "\n")
        f.write(f"Shape: {n_features} features × {n_timesteps} time steps\n")
        f.write(f"Features sorted by importance (high to low)\n\n")
        
        f.write(f"{'Feature':<20}")
        for t in range(n_timesteps):
            f.write(f"T{t+1:>3} ")
        f.write("\n" + "-"*80 + "\n")
        
        for j, feature in enumerate(sorted_features):
            f.write(f"{feature:<20}")
            for i in range(n_timesteps):
                f.write(f"{sorted_data[i, j]:>{6+DEFAULT_DECIMAL_PLACES}.{DEFAULT_DECIMAL_PLACES}f} ")
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        f.write("CUMULATIVE FEATURE IMPORTANCE (Sum |SHAP| across all time steps)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Rank':<6}{'Feature':<20}{'Importance':>15}\n")
        f.write("-"*80 + "\n")
        
        for rank, (feature, importance) in enumerate(zip(sorted_features, sorted_importance), 1):
            f.write(f"{rank:<6}{feature:<20}{importance:>15.{DEFAULT_DECIMAL_PLACES}f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"Total: {np.sum(sorted_importance):.{DEFAULT_DECIMAL_PLACES}f}\n")
        f.write(f"Mean: {np.mean(sorted_importance):.{DEFAULT_DECIMAL_PLACES}f}\n")
        f.write(f"Std: {np.std(sorted_importance):.{DEFAULT_DECIMAL_PLACES}f}\n")
        f.write("="*80 + "\n")


def generate_feature_importance_bar(feature_importance, feature_names, output_path, font_size, scale_factor):
    """Generate standalone feature importance bar plot"""
    n_features = len(feature_names)
    
    set_plot_font_size(font_size)
    
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importance = feature_importance[sorted_indices]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.barh(range(n_features), sorted_importance, color='steelblue', edgecolor='black')
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(sorted_features, fontsize=font_size)
    ax.set_xlabel(f'Sum |SHAP| (×{scale_factor:.0f})' if scale_factor != 1 else 'Sum |SHAP|', fontsize=font_size)
    
    if DEFAULT_BAR_PLOT_TITLE:
        ax.set_title(DEFAULT_BAR_PLOT_TITLE, fontsize=font_size)
    
    ax.tick_params(axis='x', labelsize=font_size)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(output_path), format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def generate_seaborn_heatmap(shap_vals, feature_names, output_path, font_size, show_numbers, scale_factor, cell_font_size):
    """Generate Seaborn heatmap"""
    n_features = shap_vals.shape[1]
    n_timesteps = shap_vals.shape[0]
    
    set_plot_font_size(font_size)
    
    feature_importance = np.abs(shap_vals).sum(axis=0)
    
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importance = feature_importance[sorted_indices]
    sorted_data = shap_vals[:, sorted_indices].T
    
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ["#2c95ff", "#63b1ff", '#b4d7ff', '#ffffff', "#ef4e6e88", "#ef4e6ebd", "#ef4e6e"]
    cmap = LinearSegmentedColormap.from_list('shap', colors_list)
    
    labels_to_show = [1] + list(range(6, n_timesteps + 1, 6))
    tick_positions = [label - 1 for label in labels_to_show]
    tick_labels = [str(label) for label in labels_to_show]
    
    if DEFAULT_SEPARATE_BAR_PLOT:
        fig, ax_heatmap = plt.subplots(figsize=(16, 8))
    else:
        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.3)
        ax_heatmap = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])
    
    sns.heatmap(sorted_data, 
                cmap=cmap,
                center=0,
                annot=False,
                cbar_kws={'label': f'SHAP Value (×{scale_factor:.0f})' if scale_factor != 1 else 'SHAP Value'},
                yticklabels=sorted_features,
                xticklabels=False,
                linewidths=0.5,
                linecolor='lightgray',
                ax=ax_heatmap)
    
    for spine in ax_heatmap.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    
    ax_heatmap.set_xticks([p + 0.5 for p in tick_positions])
    ax_heatmap.set_xticklabels(tick_labels)
    
    if show_numbers:
        for i in range(n_timesteps):
            for j in range(n_features):
                value = sorted_data[j, i]
                ax_heatmap.text(i + 0.5, j + 0.5, f'{value:.{DEFAULT_DECIMAL_PLACES}f}',
                               ha='center', va='center',
                               color='black',
                               fontsize=cell_font_size,
                               rotation=90)
    
    ax_heatmap.set_xlabel('Time Steps', fontsize=font_size)
    
    if DEFAULT_HEATMAP_YLABEL:
        ax_heatmap.set_ylabel(DEFAULT_HEATMAP_YLABEL, fontsize=font_size)
    
    ax_heatmap.tick_params(axis='both', labelsize=font_size)
    
    cbar = ax_heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label(f'SHAP Value (×{scale_factor:.0f})' if scale_factor != 1 else 'SHAP Value', fontsize=font_size)
    
    if not DEFAULT_SEPARATE_BAR_PLOT:
        ax_bar.barh(range(n_features), sorted_importance, color='steelblue', edgecolor='black')
        ax_bar.set_yticks(range(n_features))
        ax_bar.set_yticklabels(sorted_features, fontsize=font_size)
        ax_bar.set_xlabel(f'Sum |SHAP| (×{scale_factor:.0f})' if scale_factor != 1 else 'Sum |SHAP|', fontsize=font_size)
        
        if DEFAULT_BAR_PLOT_TITLE:
            ax_bar.set_title(DEFAULT_BAR_PLOT_TITLE, fontsize=font_size)
        
        ax_bar.tick_params(axis='x', labelsize=font_size)
        ax_bar.invert_yaxis()
        ax_bar.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(output_path), format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
    return feature_importance


def generate_heatmaps(primary_use, option_number, model_name, xai_method, sample_idx, 
                      font_size, show_numbers, heatmap_type, scale_factor, cell_font_size):
    """Generate heatmap(s) - SHAP, Seaborn, or both"""
    
    output_dir = Path(RESULTS_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / f"sample_{sample_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(XAI_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT shap_values_original_json FROM xai_results 
        WHERE primary_use = ? AND option_number = ? 
        AND model_name = ? AND sample_idx = ? AND xai_method = ?
    """, (primary_use, option_number, model_name, sample_idx, xai_method))
    
    result = cursor.fetchone()
    if not result:
        conn.close()
        return False, f"No SHAP values found"
    
    cursor.execute("""
        SELECT original_sample_json FROM test_samples 
        WHERE primary_use = ? AND option_number = ? AND sample_idx = ?
    """, (primary_use, option_number, sample_idx))
    
    sample_result = cursor.fetchone()
    conn.close()
    
    if not sample_result:
        return False, f"No sample data found"
    
    shap_vals = np.array(json.loads(result[0]), dtype=float) * scale_factor
    sample_data = np.array(json.loads(sample_result[0]), dtype=float)
    
    try:
        from Functions import preprocess
        container = preprocess.load_and_preprocess_data_with_sequences(
            db_path="databases/energy_data.db",
            primary_use=primary_use,
            option_number=option_number,
            scaled=True,
            scale_type="both"
        )
        feature_names = container.feature_names
    except:
        n_features = shap_vals.shape[1]
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    n_features = shap_vals.shape[1]
    if len(feature_names) < n_features:
        feature_names += [f"Feature_{i}" for i in range(len(feature_names), n_features)]
    elif len(feature_names) > n_features:
        feature_names = feature_names[:n_features]
    
    results = []
    
    if heatmap_type in ['shap', 'both']:
        shap_path = output_dir / f"{xai_method}_shap.pdf"
        try:
            generate_shap_heatmap(shap_vals, sample_data, feature_names, shap_path, font_size, show_numbers, scale_factor)
            results.append(('SHAP', str(shap_path)))
        except Exception as e:
            results.append(('SHAP', f"Error: {e}"))
    
    if heatmap_type in ['seaborn', 'both']:
        seaborn_path = output_dir / f"{xai_method}_seaborn.pdf"
        try:
            feature_importance = generate_seaborn_heatmap(shap_vals, feature_names, seaborn_path, font_size, show_numbers, scale_factor, cell_font_size)
            results.append(('Seaborn', str(seaborn_path)))
            
            log_path = output_dir / f"{xai_method}_values.log"
            save_values_log(shap_vals, feature_names, feature_importance, log_path, xai_method, scale_factor)
            results.append(('Log File', str(log_path)))
            
            # FIX: Always generate separate bar plot
            if DEFAULT_SEPARATE_BAR_PLOT:
                bar_path = output_dir / f"{xai_method}_importance.pdf"
                try:
                    generate_feature_importance_bar(feature_importance, feature_names, bar_path, font_size, scale_factor)
                    results.append(('Importance Bar', str(bar_path)))
                except Exception as e:
                    results.append(('Importance Bar', f"Error: {e}"))
        except Exception as e:
            results.append(('Seaborn', f"Error: {e}"))
    
    return True, results


def main():
    """Interactive user interface"""
    print("\n" + "="*70)
    print("📊 SHAP & SEABORN HEATMAP GENERATOR")
    print("="*70)
    
    scale_input = input(f"\n🔢 Scale factor for SHAP values (Enter={DEFAULT_SCALE_FACTOR}): ").strip()
    scale_factor = float(scale_input) if scale_input else DEFAULT_SCALE_FACTOR
    print(f"✅ Scale factor: {scale_factor}")
    
    print(f"\n🎨 Heatmap type:")
    print(f"  0: Both (SHAP + Seaborn)")
    print(f"  1: SHAP only")
    print(f"  2: Seaborn only")
    
    type_input = input(f"\n👉 Select type (Enter={DEFAULT_HEATMAP_TYPE}): ").strip()
    heatmap_type = ['both', 'shap', 'seaborn'][int(type_input) if type_input in ['0', '1', '2'] else (['both', 'shap', 'seaborn'].index(DEFAULT_HEATMAP_TYPE) if DEFAULT_HEATMAP_TYPE in ['both', 'shap', 'seaborn'] else 0)]
    print(f"✅ Type: {heatmap_type}")
    
    font_input = input(f"\n🎨 Font size (Enter={DEFAULT_FONT_SIZE}): ").strip()
    font_size = int(font_input) if font_input.isdigit() else DEFAULT_FONT_SIZE
    print(f"✅ Font size: {font_size}")
    
    show_numbers_input = input(f"\n🔢 Show numbers in cells? (y/n, Enter={'y' if DEFAULT_SHOW_NUMBERS else 'n'}): ").strip().lower()
    show_numbers = show_numbers_input != 'n' if DEFAULT_SHOW_NUMBERS else show_numbers_input == 'y'
    print(f"✅ Show numbers: {'Yes' if show_numbers else 'No'}")
    
    if show_numbers:
        cell_font_input = input(f"\n🔤 Cell numbers font size (Enter={DEFAULT_CELL_FONT_SIZE}): ").strip()
        cell_font_size = int(cell_font_input) if cell_font_input.isdigit() else DEFAULT_CELL_FONT_SIZE
        print(f"✅ Cell font size: {cell_font_size}")
    else:
        cell_font_size = DEFAULT_CELL_FONT_SIZE
    
    available_uses = get_available_primary_uses()
    print(f"\n📁 Available primary uses:")
    for i, use in enumerate(available_uses):
        print(f"  {i}: {use}")
    print(f"  Enter: all")
    
    primary_input = input("\n👉 Select primary use(s): ").strip()
    
    if primary_input == '':
        selected_uses = available_uses
        print(f"✅ Selected all")
    elif primary_input.isdigit() and int(primary_input) < len(available_uses):
        selected_uses = [available_uses[int(primary_input)]]
        print(f"✅ Selected: {selected_uses[0]}")
    else:
        try:
            indices = [int(x.strip()) for x in primary_input.split(',')]
            selected_uses = [available_uses[i] for i in indices if i < len(available_uses)]
            print(f"✅ Selected: {', '.join(selected_uses)}")
        except:
            selected_uses = available_uses
            print(f"✅ Selected all")
    
    total_generated = 0
    
    for primary_use in selected_uses:
        print(f"\n{'='*70}")
        print(f"🔄 Processing: {primary_use.upper()}")
        print(f"{'='*70}")
        
        available_options = get_available_options(primary_use)
        if not available_options:
            print(f"  ⚠️  No options available")
            continue
        
        print(f"\n🔢 Available options:")
        for i, opt in enumerate(available_options):
            print(f"  {i}: option_{opt}")
        print(f"  Enter: {available_options[0]}")
        
        option_input = input("\n👉 Select option: ").strip()
        
        if option_input == '':
            option_number = available_options[0]
        elif option_input.isdigit() and int(option_input) < len(available_options):
            option_number = available_options[int(option_input)]
        else:
            option_number = available_options[0]
        
        print(f"✅ Selected: option_{option_number}")
        
        available_models = get_available_models(primary_use, option_number)
        if not available_models:
            print(f"  ⚠️  No models available")
            continue
        
        print(f"\n🤖 Available models:")
        for i, model in enumerate(available_models):
            print(f"  {i}: {model}")
        print(f"  Enter: all")
        
        model_input = input("\n👉 Select model(s): ").strip()
        
        if model_input == '':
            selected_models = available_models
            print(f"✅ Selected all")
        elif model_input.isdigit() and int(model_input) < len(available_models):
            selected_models = [available_models[int(model_input)]]
            print(f"✅ Selected: {selected_models[0]}")
        else:
            try:
                indices = [int(x.strip()) for x in model_input.split(',')]
                selected_models = [available_models[i] for i in indices if i < len(available_models)]
                print(f"✅ Selected: {', '.join(selected_models)}")
            except:
                selected_models = available_models
                print(f"✅ Selected all")
        
        n_samples_input = input("\n🎯 Samples per model? (Enter=1): ").strip()
        n_samples = int(n_samples_input) if n_samples_input.isdigit() else 1
        print(f"✅ Generating {n_samples} sample(s)")
        
        for model in selected_models:
            print(f"\n  🔧 [{model}]")
            methods = get_available_methods(primary_use, option_number, model)
            
            if methods:
                available_samples = get_available_samples(primary_use, option_number, model, methods[0])
                samples_to_process = available_samples[:n_samples]
                
                for sample_idx in samples_to_process:
                    print(f"    📝 Sample {sample_idx}:")
                    
                    for method in methods:
                        success, results = generate_heatmaps(primary_use, option_number, model, method, 
                                                            sample_idx, font_size, show_numbers, heatmap_type, 
                                                            scale_factor, cell_font_size)
                        
                        if success:
                            for plot_type, path in results:
                                print(f"      ✅ {method} ({plot_type})")
                                total_generated += 1
                        else:
                            print(f"      ❌ {method}: {results}")
    
    print(f"\n{'='*70}")
    print(f"🎉 Generated {total_generated} heatmaps")
    print(f"📂 Location: {RESULTS_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()