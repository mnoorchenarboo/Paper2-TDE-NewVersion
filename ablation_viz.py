"""
================================================================================
FILENAME: ablation_viz.py (UPDATED FOR CONSOLIDATED TABLES)
PURPOSE: Generate publication-ready tables and figures from ablation studies
CHANGES:
  ✅ Works with 2-table structure (no ablation_id joins)
  ✅ Uses variant names directly
  ✅ Renamed efficiency → completeness
  ✅ Simplified queries
================================================================================
"""

import sqlite3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Configuration
ABLATION_DB = Path("databases/ablation_studies.db")
OUTPUT_DIR = Path("ablation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Plot styling
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# ============================
# DATA LOADING (SIMPLIFIED - NO JOINS)
# ============================

def load_ablation_results(primary_use=None, option_number=None, model_name=None, category=None):
    """Load ablation results from CONSOLIDATED table"""
    conn = sqlite3.connect(ABLATION_DB)
    
    query = '''
        SELECT 
            primary_use,
            option_number,
            model_name,
            ablation_category,
            variant_name,
            config_json,
            best_validation_loss,
            final_training_loss,
            training_time,
            n_parameters,
            mean_fidelity,
            std_fidelity,
            mean_sparsity,
            std_sparsity,
            mean_complexity,
            std_complexity,
            mean_reliability_corr,
            std_reliability_corr,
            mean_reliability_mse,
            std_reliability_mse,
            mean_completeness_error,
            std_completeness_error,
            mean_computation_time,
            std_computation_time,
            sample_results_json
        FROM ablation_results
        WHERE 1=1
    '''
    
    params = []
    if primary_use:
        query += ' AND primary_use = ?'
        params.append(primary_use)
    if option_number is not None:
        query += ' AND option_number = ?'
        params.append(option_number)
    if model_name:
        query += ' AND model_name = ?'
        params.append(model_name)
    if category:
        query += ' AND ablation_category = ?'
        params.append(category)
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    return df


def get_available_configurations():
    """Get all unique configurations in the database"""
    conn = sqlite3.connect(ABLATION_DB)
    
    query = '''
        SELECT DISTINCT primary_use, option_number, model_name
        FROM ablation_results
        ORDER BY primary_use, option_number, model_name
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df


# ============================
# TABLE GENERATION
# ============================

def generate_latex_table(df, category, output_path):
    """Generate LaTeX table for a specific ablation category"""
    
    df_cat = df[df['ablation_category'] == category].copy()
    
    if len(df_cat) == 0:
        print(f"  ⚠️  No results for category: {category}")
        return
    
    # Define metrics with direction (RENAMED: efficiency → completeness)
    metrics = {
        'best_validation_loss': {'name': 'Val Loss', 'format': '.4f', 'lower_better': True},
        'mean_fidelity': {'name': 'Fidelity', 'format': '.4f', 'lower_better': False},
        'mean_sparsity': {'name': 'Sparsity (\%)', 'format': '.1f', 'lower_better': False},
        'mean_complexity': {'name': 'Complexity', 'format': '.4f', 'lower_better': True},
        'mean_reliability_corr': {'name': 'Reliability', 'format': '.4f', 'lower_better': False},
        'mean_completeness_error': {'name': 'Completeness', 'format': '.4f', 'lower_better': True},  # RENAMED
        'training_time': {'name': 'Time (s)', 'format': '.1f', 'lower_better': True}
    }
    
    # Find baseline
    baseline_variants = {
        'architecture': 'full_tde',
        'loss_terms': 'full_loss',
        'masking': 'window_paired'
    }
    baseline_name = baseline_variants.get(category)
    
    baseline_row = df_cat[df_cat['variant_name'] == baseline_name]
    
    # Start LaTeX table
    latex = []
    latex.append(r'\begin{table}[htbp]')
    latex.append(r'\centering')
    latex.append(r'\caption{Ablation Study: ' + category.replace('_', ' ').title() + r'}')
    latex.append(r'\label{tab:ablation_' + category + r'}')
    latex.append(r'\small')
    
    latex.append(r'\begin{tabular}{l' + 'c' * len(metrics) + r'}')
    latex.append(r'\toprule')
    
    # Header
    header = ['Variant'] + [m['name'] for m in metrics.values()]
    latex.append(' & '.join(header) + r' \\')
    latex.append(r'\midrule')
    
    # Sort variants (baseline first)
    df_cat = df_cat.sort_values('variant_name', key=lambda x: x != baseline_name)
    
    # Generate rows
    for _, row in df_cat.iterrows():
        variant = row['variant_name'].replace('_', r'\_')
        
        cells = [variant]
        
        for metric_key, metric_info in metrics.items():
            value = row[metric_key]
            std_key = metric_key.replace('mean_', 'std_')
            std_value = row.get(std_key)
            
            if pd.isna(value):
                cells.append('--')
                continue
            
            # Format value
            formatted = f"{value:{metric_info['format']}}"
            
            # Add std if available
            if std_value is not None and not pd.isna(std_value):
                formatted += f" $\\pm$ {std_value:{metric_info['format']}}"
            
            # Check if best
            if baseline_row.empty:
                is_best = False
            else:
                baseline_val = baseline_row[metric_key].values[0]
                if metric_info['lower_better']:
                    is_best = value < baseline_val
                else:
                    is_best = value > baseline_val
            
            # Bold if best or baseline
            if row['variant_name'] == baseline_name or is_best:
                formatted = r'\textbf{' + formatted + r'}'
            
            # Add percentage change if not baseline
            if row['variant_name'] != baseline_name and not baseline_row.empty:
                baseline_val = baseline_row[metric_key].values[0]
                if not pd.isna(baseline_val) and baseline_val != 0:
                    pct_change = ((value - baseline_val) / baseline_val) * 100
                    sign = '+' if pct_change > 0 else ''
                    formatted += f" ({sign}{pct_change:.1f}\\%)"
            
            cells.append(formatted)
        
        latex.append(' & '.join(cells) + r' \\')
    
    latex.append(r'\bottomrule')
    latex.append(r'\end{tabular}')
    latex.append(r'\end{table}')
    
    # Write to file
    output_file = output_path / f'table_ablation_{category}.tex'
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"  ✅ LaTeX table saved: {output_file}")


def generate_all_latex_tables(df, output_dir):
    """Generate LaTeX tables for all categories"""
    print("\n📊 Generating LaTeX tables...")
    
    categories = df['ablation_category'].unique()
    
    for category in categories:
        generate_latex_table(df, category, output_dir)


# ============================
# FIGURE GENERATION (RENAMED: efficiency → completeness)
# ============================

def plot_architectural_ablation(df, output_path):
    """Plot architectural component ablation results"""
    df_arch = df[df['ablation_category'] == 'architecture'].copy()
    
    if len(df_arch) == 0:
        print("  ⚠️  No architecture ablation results")
        return
    
    # Metrics to plot (RENAMED: efficiency → completeness)
    metrics = [
        ('mean_fidelity', 'Fidelity ↑'),
        ('mean_sparsity', 'Sparsity (%) ↑'),
        ('mean_complexity', 'Complexity ↓'),
        ('mean_reliability_corr', 'Reliability ↑'),
        ('mean_completeness_error', 'Completeness Error ↓')  # RENAMED
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Variant order
    variant_order = ['full_tde', 'no_attention', 'no_direct_input', 'no_soft_threshold', 'tcn_baseline']
    variant_labels = ['Full TDE', 'No Attention', 'No Direct\nInput', 'No Soft\nThreshold', 'TCN\nBaseline']
    
    df_arch['variant_order'] = pd.Categorical(
        df_arch['variant_name'],
        categories=variant_order,
        ordered=True
    )
    df_arch = df_arch.sort_values('variant_order')
    
    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx]
        
        values = df_arch[metric].values
        std_metric = metric.replace('mean_', 'std_')
        errors = df_arch[std_metric].values if std_metric in df_arch.columns else None
        
        bars = ax.bar(range(len(values)), values, yerr=errors, capsize=5,
                      color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#95a5a6'],
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        
        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(2.5)
        
        ax.set_xticks(range(len(variant_labels)))
        ax.set_xticklabels(variant_labels, rotation=0, ha='center', fontsize=9)
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Training time
    fig.delaxes(axes[5])
    ax = fig.add_subplot(2, 3, 6)
    
    times = df_arch['training_time'].values
    bars = ax.bar(range(len(times)), times, 
                  color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#95a5a6'],
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    bars[0].set_linewidth(2.5)
    
    ax.set_xticks(range(len(variant_labels)))
    ax.set_xticklabels(variant_labels, rotation=0, ha='center', fontsize=9)
    ax.set_ylabel('Training Time (s) ↓', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    for bar, val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
               f'{val:.1f}s', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.suptitle('Architectural Component Ablation Study', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_file = output_path / 'fig_ablation_architecture.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Architecture ablation figure saved: {output_file}")


def plot_loss_term_ablation(df, output_path):
    """Plot loss term ablation results (RENAMED: efficiency → completeness)"""
    df_loss = df[df['ablation_category'] == 'loss_terms'].copy()
    
    if len(df_loss) == 0:
        print("  ⚠️  No loss term ablation results")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Variant order (RENAMED: no_efficiency → no_completeness)
    variant_order = ['full_loss', 'no_l1', 'no_l2', 'no_smoothness', 'no_completeness', 'no_sparsity', 'only_coalition']
    variant_labels = ['Full Loss', 'No L1', 'No L2', 'No\nSmoothness', 'No\nCompleteness', 'No\nSparsity', 'Only\nCoalition']
    
    df_loss['variant_order'] = pd.Categorical(
        df_loss['variant_name'],
        categories=variant_order,
        ordered=True
    )
    df_loss = df_loss.sort_values('variant_order')
    
    # Plot multiple metrics as grouped bars
    metrics = ['mean_fidelity', 'mean_sparsity', 'mean_reliability_corr']
    metric_labels = ['Fidelity', 'Sparsity (%)', 'Reliability']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    x = np.arange(len(df_loss))
    width = 0.25
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = df_loss[metric].values
        
        if metric == 'mean_sparsity':
            values_norm = values / 100
        else:
            values_norm = values
        
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values_norm, width, label=label, 
                     color=color, edgecolor='black', linewidth=1, alpha=0.8)
    
    ax.set_xlabel('Loss Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Metric Value', fontsize=12, fontweight='bold')
    ax.set_title('Loss Term Ablation Study', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_file = output_path / 'fig_ablation_loss_terms.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Loss term ablation figure saved: {output_file}")


def plot_masking_ablation(df, output_path):
    """Plot masking strategy ablation results"""
    df_mask = df[df['ablation_category'] == 'masking'].copy()
    
    if len(df_mask) == 0:
        print("  ⚠️  No masking ablation results")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Window vs Feature
    ax = axes[0]
    window_feature = df_mask[df_mask['variant_name'].isin(['window_paired', 'feature_paired'])]
    
    if len(window_feature) > 0:
        labels = ['Window', 'Feature']
        fidelity = window_feature['mean_fidelity'].values
        sparsity = window_feature['mean_sparsity'].values / 100
        reliability = window_feature['mean_reliability_corr'].values
        
        x = np.arange(len(labels))
        width = 0.25
        
        ax.bar(x - width, fidelity, width, label='Fidelity', color='#2ecc71', edgecolor='black', alpha=0.8)
        ax.bar(x, sparsity, width, label='Sparsity (norm)', color='#3498db', edgecolor='black', alpha=0.8)
        ax.bar(x + width, reliability, width, label='Reliability', color='#e74c3c', edgecolor='black', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Metric Value', fontsize=11, fontweight='bold')
        ax.set_title('Window vs Feature Masking', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
    
    # Right: Paired vs Unpaired
    ax = axes[1]
    paired_unpaired = df_mask[df_mask['variant_name'].isin(['window_paired', 'window_unpaired'])]
    
    if len(paired_unpaired) > 0:
        labels = ['Paired', 'Unpaired']
        fidelity = paired_unpaired['mean_fidelity'].values
        time = paired_unpaired['training_time'].values
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax1 = ax
        ax2 = ax1.twinx()
        
        bars1 = ax1.bar(x - width/2, fidelity, width, label='Fidelity', 
                       color='#2ecc71', edgecolor='black', alpha=0.8)
        bars2 = ax2.bar(x + width/2, time, width, label='Training Time (s)', 
                       color='#f39c12', edgecolor='black', alpha=0.8)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=11)
        ax1.set_ylabel('Fidelity', fontsize=11, fontweight='bold', color='#2ecc71')
        ax2.set_ylabel('Training Time (s)', fontsize=11, fontweight='bold', color='#f39c12')
        ax1.set_title('Paired vs Unpaired Sampling', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#2ecc71')
        ax2.tick_params(axis='y', labelcolor='#f39c12')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_file = output_path / 'fig_ablation_masking.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Masking ablation figure saved: {output_file}")


def generate_all_figures(df, output_dir):
    """Generate all ablation study figures"""
    print("\n📈 Generating figures...")
    
    plot_architectural_ablation(df, output_dir)
    plot_loss_term_ablation(df, output_dir)
    plot_masking_ablation(df, output_dir)


# ============================
# STATISTICAL ANALYSIS
# ============================

def statistical_significance_test(df):
    """Perform statistical significance tests"""
    print("\n📊 Statistical Significance Tests")
    print("="*80)
    
    categories = df['ablation_category'].unique()
    
    for category in categories:
        print(f"\n{category.upper()}:")
        print("-" * 40)
        
        df_cat = df[df['ablation_category'] == category]
        
        baseline_variants = {
            'architecture': 'full_tde',
            'loss_terms': 'full_loss',
            'masking': 'window_paired'
        }
        baseline_name = baseline_variants.get(category)
        baseline_row = df_cat[df_cat['variant_name'] == baseline_name]
        
        if baseline_row.empty:
            print("  No baseline found")
            continue
        
        baseline_samples = json.loads(baseline_row['sample_results_json'].values[0])
        baseline_fidelity = [s['fidelity'] for s in baseline_samples if s.get('fidelity') is not None]
        
        for _, row in df_cat.iterrows():
            if row['variant_name'] == baseline_name:
                continue
            
            variant_samples = json.loads(row['sample_results_json'])
            variant_fidelity = [s['fidelity'] for s in variant_samples if s.get('fidelity') is not None]
            
            if len(baseline_fidelity) > 0 and len(variant_fidelity) > 0:
                if len(baseline_fidelity) == len(variant_fidelity):
                    t_stat, p_value = stats.ttest_rel(baseline_fidelity, variant_fidelity)
                else:
                    t_stat, p_value = stats.ttest_ind(baseline_fidelity, variant_fidelity)
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                mean_diff = row['mean_fidelity'] - baseline_row['mean_fidelity'].values[0]
                pct_change = (mean_diff / baseline_row['mean_fidelity'].values[0]) * 100
                
                print(f"  {row['variant_name']:<20} | Δ={mean_diff:+.4f} ({pct_change:+.1f}%) | p={p_value:.4f} {significance}")


# ============================
# SUMMARY REPORT
# ============================

def generate_summary_report(df, output_path):
    """Generate summary report"""
    report = []
    report.append("="*80)
    report.append("ABLATION STUDY SUMMARY REPORT")
    report.append("="*80)
    report.append("")
    
    report.append(f"Total ablation variants tested: {len(df)}")
    report.append(f"Categories: {', '.join(df['ablation_category'].unique())}")
    report.append(f"Primary uses: {', '.join(df['primary_use'].unique())}")
    report.append(f"Models tested: {', '.join(df['model_name'].unique())}")
    report.append("")
    
    report.append("BEST PERFORMING VARIANTS (by Fidelity):")
    report.append("-" * 80)
    
    for category in df['ablation_category'].unique():
        df_cat = df[df['ablation_category'] == category]
        best = df_cat.loc[df_cat['mean_fidelity'].idxmax()]
        
        report.append(f"\n{category.upper()}:")
        report.append(f"  Variant: {best['variant_name']}")
        report.append(f"  Fidelity: {best['mean_fidelity']:.4f} ± {best['std_fidelity']:.4f}")
        report.append(f"  Sparsity: {best['mean_sparsity']:.1f}% ± {best['std_sparsity']:.1f}%")
        report.append(f"  Reliability: {best['mean_reliability_corr']:.4f} ± {best['std_reliability_corr']:.4f}")
        report.append(f"  Completeness Error: {best['mean_completeness_error']:.4f} ± {best['std_completeness_error']:.4f}")
        report.append(f"  Training time: {best['training_time']:.1f}s")
    
    report.append("")
    report.append("="*80)
    
    output_file = output_path / 'ablation_summary_report.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\n✅ Summary report saved: {output_file}")
    
    print('\n'.join(report))


# ============================
# MAIN EXECUTION
# ============================

def main():
    """Main visualization pipeline"""
    print("\n" + "="*80)
    print("🔬 ABLATION STUDY VISUALIZATION")
    print("="*80)
    print("💡 DEFAULT: Press Enter to visualize ALL results")
    print("="*80)
    
    if not ABLATION_DB.exists():
        print("\n❌ Ablation database not found!")
        print(f"   Expected location: {ABLATION_DB}")
        print(f"   Run tde_ablation.py first to generate results.")
        return
    
    configs_df = get_available_configurations()
    
    if len(configs_df) == 0:
        print("\n❌ No ablation results found in database!")
        print(f"   Run tde_ablation.py first to generate results.")
        return
    
    print(f"\n📂 Found {len(configs_df)} configurations in database:")
    for _, row in configs_df.iterrows():
        print(f"   • {row['primary_use']} - Option {row['option_number']} - {row['model_name']}")
    
    print(f"\n👉 Visualize which results?")
    print(f"   0: ALL configurations (recommended)")
    print(f"   1: Select specific configuration")
    
    choice = input("\n   Select [0-1, default=0]: ").strip()
    
    if choice == '1':
        print("\n   Available configurations:")
        for i, (_, row) in enumerate(configs_df.iterrows()):
            print(f"   {i}: {row['primary_use']} - Option {row['option_number']} - {row['model_name']}")
        
        cfg_input = input(f"\n   Select configuration [0-{len(configs_df)-1}]: ").strip()
        try:
            cfg_idx = int(cfg_input)
            selected_cfg = configs_df.iloc[cfg_idx]
            df = load_ablation_results(
                primary_use=selected_cfg['primary_use'],
                option_number=selected_cfg['option_number'],
                model_name=selected_cfg['model_name']
            )
            print(f"\n   ✅ Selected: {selected_cfg['primary_use']} - Option {selected_cfg['option_number']} - {selected_cfg['model_name']}")
        except:
            print("\n   ⚠️  Invalid selection, using ALL")
            df = load_ablation_results()
    else:
        df = load_ablation_results()
        print(f"\n   ✅ Selected: ALL ({len(df)} ablation variants)")
    
    if len(df) == 0:
        print("\n❌ No results to visualize!")
        return
    
    print(f"\n   Loaded {len(df)} ablation variants")
    print(f"   Categories: {', '.join(df['ablation_category'].unique())}")
    
    generate_all_latex_tables(df, OUTPUT_DIR)
    generate_all_figures(df, OUTPUT_DIR)
    statistical_significance_test(df)
    generate_summary_report(df, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("\n   Generated files:")
    for file in sorted(OUTPUT_DIR.iterdir()):
        print(f"   • {file.name}")
    print("="*80)


if __name__ == "__main__":
    main()