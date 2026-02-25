"""
Visualization Dashboard for Supply Chain Risk
Creates risk heatmaps, trend charts, and portfolio views
"""

import pandas as pd
import numpy as np

class RiskVisualizer:
    """Generate risk visualizations (text-based for reliability)"""
    
    @staticmethod
    def risk_heatmap_by_class(data, risk_scores):
        """Create ASCII heatmap of risk by SKU class"""
        df = pd.DataFrame({
            'sku_class': data['sku_class'],
            'risk_score': risk_scores
        })
        
        heatmap = df.groupby('sku_class')['risk_score'].agg(['mean', 'count']).round(3)
        heatmap = heatmap.sort_values('mean', ascending=False)
        
        print("\n" + "="*60)
        print("RISK HEATMAP BY SKU CLASS")
        print("="*60)
        print(f"{'SKU Class':<12} {'Avg Risk':>10} {'Count':>8} {'Heat':>15}")
        print("-"*60)
        
        for sku, row in heatmap.iterrows():
            risk = row['mean']
            # ASCII heat bar
            heat_bar = "█" * int(risk * 10) + "░" * (10 - int(risk * 10))
            print(f"{sku:<12} {risk:>10.3f} {int(row['count']):>8} {heat_bar}")
        
        return heatmap
    
    @staticmethod
    def risk_by_region(data, risk_scores):
        """Regional risk analysis"""
        df = pd.DataFrame({
            'region': data['region'],
            'geopolitical_risk': data['geopolitical_risk'],
            'predicted_risk': risk_scores
        })
        
        regional = df.groupby('region').agg({
            'geopolitical_risk': 'mean',
            'predicted_risk': 'mean',
            'region': 'count'
        }).rename(columns={'region': 'count'}).round(3)
        
        print("\n" + "="*60)
        print("RISK BY GEOGRAPHIC REGION")
        print("="*60)
        print(f"{'Region':<10} {'Suppliers':>10} {'Geo Risk':>12} {'Predicted':>12}")
        print("-"*60)
        
        for region, row in regional.iterrows():
            print(f"{region:<10} {int(row['count']):>10} {row['geopolitical_risk']:>12.3f} {row['predicted_risk']:>12.3f}")
        
        return regional
    
    @staticmethod
    def risk_distribution(risk_scores):
        """Histogram of risk scores"""
        print("\n" + "="*60)
        print("RISK SCORE DISTRIBUTION")
        print("="*60)
        
        bins = [0, 0.25, 0.45, 0.65, 1.0]
        labels = ["LOW (0-0.25)", "MEDIUM (0.25-0.45)", "HIGH (0.45-0.65)", "CRITICAL (0.65-1.0)"]
        
        counts = []
        for i in range(len(bins)-1):
            mask = (np.array(risk_scores) >= bins[i]) & (np.array(risk_scores) < bins[i+1])
            count = mask.sum()
            counts.append(count)
        
        total = len(risk_scores)
        print(f"{'Risk Level':<25} {'Count':>8} {'Percentage':>12} {'Bar':>15}")
        print("-"*60)
        
        for label, count in zip(labels, counts):
            pct = count / total * 100
            bar = "█" * int(pct / 5)
            print(f"{label:<25} {count:>8} {pct:>11.1f}% {bar}")
        
        print(f"\nTotal: {total} SKUs")
        print(f"Mean Risk Score: {np.mean(risk_scores):.3f}")
    
    @staticmethod
    def top_risk_items(data, risk_scores, top_n=10):
        """Show highest risk items"""
        df = pd.DataFrame({
            'sku_class': data['sku_class'],
            'supplier_reliability': data['supplier_reliability'],
            'is_sole_source': data['is_sole_source'],
            'current_inventory': data['current_inventory'],
            'risk_score': risk_scores
        })
        
        top_risk = df.nlargest(top_n, 'risk_score')
        
        print("\n" + "="*80)
        print(f"TOP {top_n} HIGHEST RISK ITEMS")
        print("="*80)
        print(f"{'Rank':<6} {'SKU':<8} {'Risk':>8} {'Sole':>6} {'Reliability':>12} {'Inventory':>12}")
        print("-"*80)
        
        for i, (idx, row) in enumerate(top_risk.iterrows(), 1):
            sole = "YES" if row['is_sole_source'] else "NO"
            print(f"{i:<6} {row['sku_class']:<8} {row['risk_score']:>8.3f} {sole:>6} {row['supplier_reliability']:>12.3f} {int(row['current_inventory']):>12}")
        
        return top_risk
    
    @staticmethod
    def action_summary(assessments):
        """Summary of recommended actions"""
        actions = [a['action'] for a in assessments]
        action_counts = pd.Series(actions).value_counts()
        
        print("\n" + "="*60)
        print("RECOMMENDED ACTIONS SUMMARY")
        print("="*60)
        print(f"{'Action':<40} {'Count':>10} {'Percent':>10}")
        print("-"*60)
        
        total = len(actions)
        for action, count in action_counts.items():
            pct = count / total * 100
            print(f"{action[:38]:<40} {count:>10} {pct:>9.1f}%")
        
        critical = sum(1 for a in action_counts.index if 'Switch' in a or 'Emergency' in a or 'Production' in a)
        print(f"\nCritical actions required: {critical} ({critical/total*100:.1f}%)")
        
        return action_counts

def generate_full_dashboard(engine, data, X_sample, n_samples=100):
    """Generate complete risk dashboard"""
    print("\n" + "="*80)
    print("  SUPPLY CHAIN RESILIENCE DASHBOARD")
    print("  Real-Time Risk Monitoring & Decision Support")
    print("="*80)
    
    # Get risk assessments
    assessments = []
    risk_scores = []
    
    sample_data = X_sample.sample(min(n_samples, len(X_sample)), random_state=42)
    
    for idx, row in sample_data.iterrows():
        sample = pd.DataFrame([row])
        try:
            assessment = engine.assess_disruption_risk(sample)
            assessments.append(assessment)
            risk_scores.append(assessment['composite_risk_score'])
        except:
            pass
    
    # Generate visualizations
    viz = RiskVisualizer()
    
    viz.risk_distribution(risk_scores)
    viz.risk_heatmap_by_class(data.loc[sample_data.index], risk_scores)
    viz.risk_by_region(data.loc[sample_data.index], risk_scores)
    viz.top_risk_items(data.loc[sample_data.index], risk_scores, top_n=10)
    viz.action_summary(assessments)
    
    print("\n" + "="*80)
    print("Dashboard Generation Complete")
    print("="*80)
    
    return {
        'risk_scores': risk_scores,
        'assessments': assessments,
        'sample_data': sample_data
    }

if __name__ == "__main__":
    print("Risk visualization utilities ready")
    print("Use: generate_full_dashboard(engine, data, X, n_samples=100)")
