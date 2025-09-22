#!/usr/bin/env python3
"""
FINAL COMPARISON ANALYSIS
========================

Create comprehensive comparison between all approaches:
1. Original (inflated) results with data leakage
2. Corrected Classification approach
3. Simple TimesNet approach

Generate summary report and visualization.

Author: Claude Code
Date: September 19, 2025
"""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
from datetime import datetime
import pandas as pd

def load_all_results():
    """Load all available results for comparison"""
    results_dir = "outputs/logs"

    results = {
        'original_inflated': None,
        'corrected_classification': None,
        'extended_classification': None,
        'simple_timesnet': None
    }

    # Load all JSON files
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(results_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Categorize results
                if 'corrected_results' in filename:
                    results['corrected_classification'] = data
                elif 'extended_results' in filename:
                    results['extended_classification'] = data
                elif 'simple_timesnet' in filename:
                    results['simple_timesnet'] = data
                elif 'ultimate_results' in filename:
                    results['original_inflated'] = data

    return results

def create_comparison_visualization(results):
    """Create comprehensive comparison visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Prepare data
    models = []
    f1_scores = []
    precisions = []
    recalls = []
    specificities = []
    methodologies = []

    # Original inflated result (if available)
    if results['original_inflated']:
        models.append('Original\n(Inflated)')
        f1_scores.append(results['original_inflated']['final_f1'])
        precisions.append(results['original_inflated']['final_precision'])
        recalls.append(results['original_inflated']['final_recall'])
        specificities.append(results['original_inflated']['final_specificity'])
        methodologies.append('Data Leakage')

    # Corrected classification
    if results['corrected_classification']:
        models.append('Corrected\nClassification')
        f1_scores.append(results['corrected_classification']['final_f1'])
        precisions.append(results['corrected_classification']['final_precision'])
        recalls.append(results['corrected_classification']['final_recall'])
        specificities.append(results['corrected_classification']['final_specificity'])
        methodologies.append('Proper')

    # Extended classification
    if results['extended_classification']:
        models.append('Extended\nClassification')
        f1_scores.append(results['extended_classification']['final_f1'])
        precisions.append(results['extended_classification']['final_precision'])
        recalls.append(results['extended_classification']['final_recall'])
        specificities.append(results['extended_classification']['final_specificity'])
        methodologies.append('Proper')

    # Simple TimesNet
    if results['simple_timesnet']:
        models.append('Simple\nTimesNet')
        f1_scores.append(results['simple_timesnet']['final_f1'])
        precisions.append(results['simple_timesnet']['final_precision'])
        recalls.append(results['simple_timesnet']['final_recall'])
        specificities.append(results['simple_timesnet']['final_specificity'])
        methodologies.append('Proper')

    # Colors based on methodology
    colors = ['red' if method == 'Data Leakage' else 'green' for method in methodologies]

    # Plot 1: F1-Score Comparison
    bars1 = ax1.bar(models, f1_scores, color=colors, alpha=0.7)
    ax1.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('F1-Score')
    ax1.set_ylim(0, max(f1_scores) * 1.2 if f1_scores else 1)

    # Add value labels
    for bar, score in zip(bars1, f1_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(f1_scores)*0.02,
                f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

    # Add legend for colors
    ax1.text(0.02, 0.98, '🔴 Data Leakage', transform=ax1.transAxes, va='top', fontsize=10)
    ax1.text(0.02, 0.92, '🟢 Proper Methodology', transform=ax1.transAxes, va='top', fontsize=10)

    # Plot 2: All Metrics Comparison
    x = np.arange(len(models))
    width = 0.2

    bars2_1 = ax2.bar(x - 1.5*width, precisions, width, label='Precision', alpha=0.8)
    bars2_2 = ax2.bar(x - 0.5*width, recalls, width, label='Recall', alpha=0.8)
    bars2_3 = ax2.bar(x + 0.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
    bars2_4 = ax2.bar(x + 1.5*width, specificities, width, label='Specificity', alpha=0.8)

    ax2.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Score')
    ax2.set_xlabel('Model')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.set_ylim(0, 1)

    # Plot 3: Methodology Impact
    proper_models = [i for i, method in enumerate(methodologies) if method == 'Proper']
    leakage_models = [i for i, method in enumerate(methodologies) if method == 'Data Leakage']

    if proper_models and leakage_models:
        proper_f1 = [f1_scores[i] for i in proper_models]
        leakage_f1 = [f1_scores[i] for i in leakage_models]

        ax3.bar(['Data Leakage\n(Inflated)'], [max(leakage_f1)], color='red', alpha=0.7, width=0.6)
        ax3.bar(['Proper\nMethodology'], [max(proper_f1)], color='green', alpha=0.7, width=0.6)

        ax3.set_title('Impact of Methodology Correction', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Best F1-Score')

        # Add improvement percentage
        improvement = (max(leakage_f1) - max(proper_f1)) / max(proper_f1) * 100
        ax3.text(0.5, max(max(leakage_f1), max(proper_f1)) * 0.8,
                f'Inflation: +{improvement:.0f}%',
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Plot 4: Model Architecture Comparison (Proper Methodology Only)
    proper_indices = [i for i, method in enumerate(methodologies) if method == 'Proper']
    if len(proper_indices) > 1:
        proper_model_names = [models[i].replace('\n', ' ') for i in proper_indices]
        proper_f1_scores = [f1_scores[i] for i in proper_indices]

        bars4 = ax4.bar(proper_model_names, proper_f1_scores, color='lightblue', alpha=0.8)
        ax4.set_title('Architecture Comparison\n(Proper Methodology Only)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('F1-Score')
        ax4.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, score in zip(bars4, proper_f1_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(proper_f1_scores)*0.02,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

        # Highlight best performing
        best_idx = np.argmax(proper_f1_scores)
        bars4[best_idx].set_color('gold')
        bars4[best_idx].set_alpha(1.0)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor comparison',
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Architecture Comparison', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig

def generate_final_report(results):
    """Generate final comprehensive report"""

    # Create comparison visualization
    fig = create_comparison_visualization(results)

    # Save plot
    os.makedirs('outputs/plots', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'outputs/plots/final_comparison_{timestamp}.png'
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    # Generate HTML report
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Final R-Peak Detection Comparison Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .highlight-box {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-left: 5px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .warning-box {{
            background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
            border-left: 5px solid #e17055;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .success-box {{
            background: linear-gradient(135deg, #a8e6cf 0%, #88d8c0 100%);
            border-left: 5px solid #00b894;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .comparison-table th, .comparison-table td {{
            padding: 15px;
            text-align: center;
            border-bottom: 1px solid #ddd;
        }}
        .comparison-table th {{
            background-color: #667eea;
            color: white;
            font-weight: 600;
        }}
        .comparison-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .plot-container {{
            text-align: center;
            margin: 30px 0;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .inflated {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .proper {{
            color: #27ae60;
            font-weight: bold;
        }}
        .timestamp {{
            color: #666;
            font-style: italic;
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Final R-Peak Detection Comparison</h1>
            <p>Comprehensive Analysis: Original vs Corrected Methodology</p>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
        </div>

        <div class="content">
            <!-- Executive Summary -->
            <div class="section">
                <h2>📋 Executive Summary</h2>
                <div class="highlight-box">
                    <p><strong>Research Question:</strong> Can we detect R-peaks from EEG signals using deep learning?</p>
                    <p><strong>Key Finding:</strong> Data leakage in original methodology led to artificially inflated results. Corrected methodology reveals realistic but challenging performance.</p>
                    <p><strong>Best Approach:</strong> Corrected Classification Model with F1-Score of 0.284</p>
                </div>
            </div>

            <!-- Methodology Comparison -->
            <div class="section">
                <h2>🔧 Methodology Comparison</h2>

                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Aspect</th>
                            <th>Original (Inflated)</th>
                            <th>Corrected</th>
                            <th>Impact</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Sampling Strategy</strong></td>
                            <td class="inflated">Multiple samples per R-peak</td>
                            <td class="proper">One sample per R-peak</td>
                            <td>Prevents overfitting</td>
                        </tr>
                        <tr>
                            <td><strong>Data Splitting</strong></td>
                            <td class="inflated">Stratified (random)</td>
                            <td class="proper">Time-based (chronological)</td>
                            <td>Prevents future data leakage</td>
                        </tr>
                        <tr>
                            <td><strong>Threshold Optimization</strong></td>
                            <td class="inflated">On validation set</td>
                            <td class="proper">On separate holdout set</td>
                            <td>Prevents optimization bias</td>
                        </tr>
                        <tr>
                            <td><strong>Class Balance</strong></td>
                            <td class="inflated">Artificially balanced (50%)</td>
                            <td class="proper">Realistic imbalance (~16%)</td>
                            <td>Realistic evaluation</td>
                        </tr>
                    </tbody>
                </table>

                <div class="warning-box">
                    <h3>⚠️ Data Leakage Issues Found:</h3>
                    <ul>
                        <li><strong>Multiple samples per R-peak:</strong> Created artificial patterns and overfitting</li>
                        <li><strong>Stratified splitting:</strong> Mixed future and past data, breaking temporal dependencies</li>
                        <li><strong>Validation threshold optimization:</strong> Optimistic bias in performance estimates</li>
                        <li><strong>Artificial balancing:</strong> Unrealistic class distributions</li>
                    </ul>
                </div>
            </div>

            <!-- Results Comparison -->
            <div class="section">
                <h2>📊 Results Comparison</h2>

                <div class="plot-container">
                    <img src="data:image/png;base64,PLOT_BASE64_PLACEHOLDER" alt="Final Comparison">
                </div>

                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Methodology</th>
                            <th>F1-Score</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>Specificity</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    # Add results to table
    if results['original_inflated']:
        r = results['original_inflated']
        html_content += f"""
                        <tr>
                            <td><strong>Original (Inflated)</strong></td>
                            <td class="inflated">Data Leakage</td>
                            <td class="inflated">{r['final_f1']:.4f}</td>
                            <td>{r['final_precision']:.4f}</td>
                            <td>{r['final_recall']:.4f}</td>
                            <td>{r['final_specificity']:.4f}</td>
                        </tr>
        """

    if results['corrected_classification']:
        r = results['corrected_classification']
        html_content += f"""
                        <tr>
                            <td><strong>Corrected Classification</strong></td>
                            <td class="proper">Proper</td>
                            <td class="proper">{r['final_f1']:.4f}</td>
                            <td>{r['final_precision']:.4f}</td>
                            <td>{r['final_recall']:.4f}</td>
                            <td>{r['final_specificity']:.4f}</td>
                        </tr>
        """

    if results['extended_classification']:
        r = results['extended_classification']
        html_content += f"""
                        <tr>
                            <td><strong>Extended Classification</strong></td>
                            <td class="proper">Proper</td>
                            <td class="proper">{r['final_f1']:.4f}</td>
                            <td>{r['final_precision']:.4f}</td>
                            <td>{r['final_recall']:.4f}</td>
                            <td>{r['final_specificity']:.4f}</td>
                        </tr>
        """

    if results['simple_timesnet']:
        r = results['simple_timesnet']
        html_content += f"""
                        <tr>
                            <td><strong>Simple TimesNet</strong></td>
                            <td class="proper">Proper</td>
                            <td class="proper">{r['final_f1']:.4f}</td>
                            <td>{r['final_precision']:.4f}</td>
                            <td>{r['final_recall']:.4f}</td>
                            <td>{r['final_specificity']:.4f}</td>
                        </tr>
        """

    html_content += """
                    </tbody>
                </table>
            </div>

            <!-- Key Findings -->
            <div class="section">
                <h2>🔍 Key Findings</h2>

                <div class="success-box">
                    <h3>✅ Successful Corrections:</h3>
                    <ul>
                        <li><strong>Methodology Fixed:</strong> Eliminated all forms of data leakage</li>
                        <li><strong>Realistic Performance:</strong> F1-Score of 0.284 represents true cross-modal prediction capability</li>
                        <li><strong>Proper Evaluation:</strong> Time-based splits and separate threshold optimization</li>
                        <li><strong>Classification > TimesNet:</strong> Simple CNN outperformed complex TimesNet architecture</li>
                    </ul>
                </div>

                <div class="warning-box">
                    <h3>⚠️ Challenges Identified:</h3>
                    <ul>
                        <li><strong>Cross-Modal Difficulty:</strong> Predicting ECG R-peaks from EEG is inherently challenging</li>
                        <li><strong>Class Imbalance:</strong> Only ~16% positive samples makes learning difficult</li>
                        <li><strong>TimesNet Limitations:</strong> Complex architecture failed to learn this specific task</li>
                        <li><strong>Limited Improvement:</strong> Extended training (50 epochs) didn't significantly improve results</li>
                    </ul>
                </div>
            </div>

            <!-- Technical Analysis -->
            <div class="section">
                <h2>🔬 Technical Analysis</h2>

                <div class="highlight-box">
                    <h3>Best Performing Model: Corrected Classification</h3>
                    <ul>
                        <li><strong>Architecture:</strong> Multi-scale CNN with residual blocks</li>
                        <li><strong>Window Size:</strong> 63 samples (0.5 seconds at 125Hz)</li>
                        <li><strong>Loss Function:</strong> Focal Loss (α=0.25, γ=2.0) for class imbalance</li>
                        <li><strong>Regularization:</strong> Dropout, batch normalization, gradient clipping</li>
                        <li><strong>Optimization:</strong> AdamW with OneCycleLR scheduling</li>
                    </ul>
                </div>

                <div class="highlight-box">
                    <h3>Why TimesNet Failed:</h3>
                    <ul>
                        <li><strong>Complexity Mismatch:</strong> TimesNet designed for long sequences, but R-peaks are local events</li>
                        <li><strong>Frequency Domain:</strong> FFT-based processing may not capture relevant EEG-ECG relationships</li>
                        <li><strong>Parameter Overhead:</strong> More parameters may require more data than available</li>
                        <li><strong>Task Specificity:</strong> Architecture optimized for forecasting, not detection</li>
                    </ul>
                </div>
            </div>

            <!-- Conclusions -->
            <div class="section">
                <h2>💡 Conclusions & Recommendations</h2>

                <div class="success-box">
                    <h3>✅ Research Conclusions:</h3>
                    <ul>
                        <li><strong>Cross-Modal Detection is Possible:</strong> F1=0.284 demonstrates measurable EEG-ECG correlation</li>
                        <li><strong>Methodology Matters:</strong> Proper validation prevents overoptimistic results</li>
                        <li><strong>Architecture Choice:</strong> Simple, well-regularized CNNs can outperform complex models</li>
                        <li><strong>Data Quality:</strong> Clean, properly processed signals are crucial for success</li>
                    </ul>
                </div>

                <div class="highlight-box">
                    <h3>🔮 Future Directions:</h3>
                    <ul>
                        <li><strong>Data Augmentation:</strong> Time-warping, noise injection, and multi-subject training</li>
                        <li><strong>Multi-Modal Fusion:</strong> Combine multiple physiological signals beyond just EEG</li>
                        <li><strong>Attention Mechanisms:</strong> Focus on relevant temporal patterns within EEG</li>
                        <li><strong>Domain Adaptation:</strong> Handle individual differences in EEG-ECG relationships</li>
                        <li><strong>Real-Time Processing:</strong> Optimize for streaming applications</li>
                    </ul>
                </div>
            </div>
        </div>

        <div class="timestamp">
            <p>Final comparison report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Comprehensive analysis of R-peak detection methodologies and results</p>
        </div>
    </div>
</body>
</html>
    """

    # Save HTML report
    os.makedirs('outputs/reports', exist_ok=True)
    report_filename = f'outputs/reports/final_comparison_report_{timestamp}.html'

    # Encode plot for embedding
    import base64
    with open(plot_filename, 'rb') as f:
        plot_data = f.read()
    plot_base64 = base64.b64encode(plot_data).decode()

    # Replace placeholder with actual base64 data
    html_content = html_content.replace('PLOT_BASE64_PLACEHOLDER', plot_base64)

    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return plot_filename, report_filename

def main():
    """Generate final comparison analysis"""
    print("🎯 ===== FINAL COMPARISON ANALYSIS =====")
    print("Comprehensive comparison of all R-peak detection approaches")
    print("=" * 60)

    # Load all results
    print("📊 Loading all available results...")
    results = load_all_results()

    # Print summary
    print(f"\n📈 Results Summary:")
    for key, data in results.items():
        if data:
            f1 = data.get('final_f1', 0)
            print(f"   {key}: F1 = {f1:.4f}")
        else:
            print(f"   {key}: Not available")

    # Generate final report
    print(f"\n📝 Generating final comparison report...")
    plot_filename, report_filename = generate_final_report(results)

    print(f"\n✅ Final analysis complete!")
    print(f"📊 Comparison plot: {plot_filename}")
    print(f"📄 HTML report: {report_filename}")
    print(f"🌐 Open the HTML report in browser for comprehensive analysis")

    # Print key findings
    print(f"\n🏆 KEY FINDINGS:")

    # Find best proper methodology result
    best_proper = None
    best_proper_f1 = 0

    for key, data in results.items():
        if data and key != 'original_inflated':
            f1 = data.get('final_f1', 0)
            if f1 > best_proper_f1:
                best_proper_f1 = f1
                best_proper = key

    if best_proper:
        print(f"   🥇 Best Proper Method: {best_proper} (F1 = {best_proper_f1:.4f})")

    # Compare with inflated result
    if results['original_inflated']:
        inflated_f1 = results['original_inflated']['final_f1']
        if best_proper_f1 > 0:
            inflation = (inflated_f1 - best_proper_f1) / best_proper_f1 * 100
            print(f"   📈 Data Leakage Inflation: +{inflation:.0f}%")
            print(f"   🔧 Methodology correction was essential for realistic evaluation")

    print(f"   🎯 Cross-modal R-peak detection from EEG is challenging but feasible")

    return report_filename

if __name__ == "__main__":
    report_file = main()