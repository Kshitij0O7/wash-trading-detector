import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
import warnings
warnings.filterwarnings('ignore')

class WashTradingDetector:
    def __init__(self, df):
        self.df = df.copy()
        self.prepare_data()
        self.suspicious_patterns = {}
        
    def prepare_data(self):
        """Prepare and clean the data for analysis"""
        # Convert time to datetime if it's a string
        if 'Block.Time' in self.df.columns:
            self.df['Block.Time'] = pd.to_datetime(self.df['Block.Time'])
        
        # Create simplified column names for easier access
        self.df['buyer'] = self.df.get('Trade.Buy.Account.Address', '')
        self.df['seller'] = self.df.get('Trade.Sell.Account.Address', '')
        self.df['token_mint'] = self.df.get('Trade.Buy.Currency.MintAddress', '')
        self.df['tx_signature'] = self.df.get('Transaction.Signature', '')
        self.df['buy_amount'] = pd.to_numeric(self.df.get('Trade.Buy.Amount', 0), errors='coerce')
        self.df['sell_amount'] = pd.to_numeric(self.df.get('Trade.Sell.Amount', 0), errors='coerce')
        self.df['buy_price_usd'] = pd.to_numeric(self.df.get('Trade.Buy.PriceInUSD', 0), errors='coerce')
        self.df['sell_price_usd'] = pd.to_numeric(self.df.get('Trade.Sell.PriceInUSD', 0), errors='coerce')
        
        # Remove rows with missing critical data
        self.df = self.df.dropna(subset=['buyer', 'seller', 'token_mint'])
        
    def detect_self_trades(self):
        """Detect direct self-trading (same wallet buying and selling)"""
        self_trades = self.df[self.df['buyer'] == self.df['seller']]
        return {
            'pattern': 'self_trades',
            'count': len(self_trades),
            'transactions': self_trades['tx_signature'].tolist(),
            'wallets': self_trades['buyer'].unique().tolist(),
            'tokens': self_trades['token_mint'].unique().tolist(),
            'severity': 'HIGH' if len(self_trades) > 0 else 'LOW'
        }
    
    def detect_repeated_pairs(self, threshold=5):
        """Detect wallets that trade with each other repeatedly"""
        # Create trading pairs
        pairs = self.df.groupby(['buyer', 'seller']).agg({
            'tx_signature': 'count',
            'token_mint': 'nunique',
            'buy_amount': 'sum'
        }).reset_index()
        
        pairs.columns = ['buyer', 'seller', 'trade_count', 'token_count', 'total_volume']
        repeated_pairs = pairs[pairs['trade_count'] > threshold]
        
        # Get suspicious transactions
        suspicious_txs = []
        for _, row in repeated_pairs.iterrows():
            txs = self.df[
                (self.df['buyer'] == row['buyer']) & 
                (self.df['seller'] == row['seller'])
            ]['tx_signature'].tolist()
            suspicious_txs.extend(txs)
            
        return {
            'pattern': 'repeated_pairs',
            'count': len(repeated_pairs),
            'transactions': suspicious_txs,
            'pairs': repeated_pairs.to_dict('records'),
            'severity': 'HIGH' if len(repeated_pairs) > 0 else 'LOW'
        }
    
    def detect_circular_trading(self):
        """Detect circular trading patterns (A->B->A or A->B->C->A)"""
        circular_trades = []
        
        # Group by wallet to find their trading partners
        wallet_trades = defaultdict(list)
        for _, row in self.df.iterrows():
            wallet_trades[row['buyer']].append({
                'counterparty': row['seller'],
                'tx': row['tx_signature'],
                'token': row['token_mint'],
                'time': row.get('Block.Time', None)
            })
            wallet_trades[row['seller']].append({
                'counterparty': row['buyer'],
                'tx': row['tx_signature'],
                'token': row['token_mint'],
                'time': row.get('Block.Time', None)
            })
        
        # Look for circular patterns
        for wallet, trades in wallet_trades.items():
            counterparties = [t['counterparty'] for t in trades]
            counterparty_counts = Counter(counterparties)
            
            # If a wallet trades with the same counterparty multiple times
            for counterparty, count in counterparty_counts.items():
                if count >= 4:  # At least 2 round trips
                    circular_trades.append({
                        'wallet_a': wallet,
                        'wallet_b': counterparty,
                        'round_trips': count // 2,
                        'transactions': [t['tx'] for t in trades if t['counterparty'] == counterparty]
                    })
        
        return {
            'pattern': 'circular_trading',
            'count': len(circular_trades),
            'transactions': [tx for trade in circular_trades for tx in trade['transactions']],
            'circles': circular_trades,
            'severity': 'HIGH' if len(circular_trades) > 0 else 'LOW'
        }
    
    def detect_timing_patterns(self, time_threshold_minutes=5):
        """Detect suspiciously regular trading patterns"""
        if 'Block.Time' not in self.df.columns:
            return {'pattern': 'timing_patterns', 'count': 0, 'severity': 'LOW'}
        
        timing_anomalies = []
        
        # Group by token and analyze timing
        for token in self.df['token_mint'].unique():
            token_trades = self.df[self.df['token_mint'] == token].sort_values('Block.Time')
            
            if len(token_trades) < 5:
                continue
                
            # Calculate time differences between consecutive trades
            time_diffs = token_trades['Block.Time'].diff().dt.total_seconds() / 60  # in minutes
            
            # Look for suspiciously regular patterns
            if len(time_diffs) > 1:
                # Check if many trades happen at very regular intervals
                regular_intervals = time_diffs[(time_diffs > 0) & (time_diffs < time_threshold_minutes)]
                
                if len(regular_intervals) > len(token_trades) * 0.3:  # More than 30% of trades
                    timing_anomalies.append({
                        'token': token,
                        'regular_trades': len(regular_intervals),
                        'total_trades': len(token_trades),
                        'avg_interval_minutes': regular_intervals.mean(),
                        'transactions': token_trades['tx_signature'].tolist()
                    })
        
        return {
            'pattern': 'timing_patterns',
            'count': len(timing_anomalies),
            'transactions': [tx for anomaly in timing_anomalies for tx in anomaly['transactions']],
            'anomalies': timing_anomalies,
            'severity': 'MEDIUM' if len(timing_anomalies) > 0 else 'LOW'
        }
    
    def detect_volume_anomalies(self, volume_threshold=0.8):
        """Detect tokens where most volume comes from few wallets"""
        volume_concentration = []
        
        for token in self.df['token_mint'].unique():
            token_trades = self.df[self.df['token_mint'] == token]
            
            if len(token_trades) < 10:  # Skip tokens with too few trades
                continue
            
            # Calculate volume per wallet
            wallet_volume = token_trades.groupby('buyer')['buy_amount'].sum()
            total_volume = wallet_volume.sum()
            
            if total_volume == 0:
                continue
            
            # Check if top wallets dominate volume
            top_wallets_volume = wallet_volume.nlargest(3).sum()
            concentration_ratio = top_wallets_volume / total_volume
            
            if concentration_ratio > volume_threshold:
                volume_concentration.append({
                    'token': token,
                    'concentration_ratio': concentration_ratio,
                    'top_wallets': wallet_volume.nlargest(3).index.tolist(),
                    'total_volume': total_volume,
                    'transactions': token_trades['tx_signature'].tolist()
                })
        
        return {
            'pattern': 'volume_concentration',
            'count': len(volume_concentration),
            'transactions': [tx for conc in volume_concentration for tx in conc['transactions']],
            'concentrations': volume_concentration,
            'severity': 'MEDIUM' if len(volume_concentration) > 0 else 'LOW'
        }
    
    def detect_price_manipulation(self, price_deviation_threshold=0.5):
        """Detect unusual price movements that might indicate manipulation"""
        price_anomalies = []
        
        for token in self.df['token_mint'].unique():
            token_trades = self.df[
                (self.df['token_mint'] == token) & 
                (self.df['buy_price_usd'] > 0)
            ].sort_values('Block.Time')
            
            if len(token_trades) < 5:
                continue
            
            # Calculate price volatility
            prices = token_trades['buy_price_usd']
            price_changes = prices.pct_change().abs()
            
            # Look for extreme price swings
            extreme_changes = price_changes[price_changes > price_deviation_threshold]
            
            if len(extreme_changes) > len(token_trades) * 0.2:  # More than 20% extreme changes
                price_anomalies.append({
                    'token': token,
                    'extreme_changes': len(extreme_changes),
                    'avg_change': extreme_changes.mean(),
                    'max_change': extreme_changes.max(),
                    'transactions': token_trades['tx_signature'].tolist()
                })
        
        return {
            'pattern': 'price_manipulation',
            'count': len(price_anomalies),
            'transactions': [tx for anomaly in price_anomalies for tx in anomaly['transactions']],
            'anomalies': price_anomalies,
            'severity': 'HIGH' if len(price_anomalies) > 0 else 'LOW'
        }
    
    def detect_new_wallet_patterns(self):
        """Detect patterns involving newly created wallets"""
        # This is a simplified version - in reality you'd need wallet creation timestamps
        wallet_activity = self.df.groupby('buyer').agg({
            'tx_signature': 'count',
            'token_mint': 'nunique',
            'Block.Time': ['min', 'max'] if 'Block.Time' in self.df.columns else 'count'
        }).reset_index()
        
        # Flatten column names
        wallet_activity.columns = ['wallet', 'trade_count', 'token_count', 'first_trade', 'last_trade']
        
        # Look for wallets that trade only specific tokens and have limited activity
        suspicious_new_wallets = wallet_activity[
            (wallet_activity['trade_count'] <= 10) & 
            (wallet_activity['token_count'] <= 2)
        ]
        
        if len(suspicious_new_wallets) == 0:
            return {'pattern': 'new_wallet_patterns', 'count': 0, 'severity': 'LOW'}
        
        suspicious_txs = []
        for wallet in suspicious_new_wallets['wallet']:
            txs = self.df[self.df['buyer'] == wallet]['tx_signature'].tolist()
            suspicious_txs.extend(txs)
        
        return {
            'pattern': 'new_wallet_patterns',
            'count': len(suspicious_new_wallets),
            'transactions': suspicious_txs,
            'wallets': suspicious_new_wallets['wallet'].tolist(),
            'severity': 'MEDIUM' if len(suspicious_new_wallets) > 5 else 'LOW'
        }
    
    def analyze_all_patterns(self):
        """Run all detection methods and compile results"""
        print("🔍 Running comprehensive wash trading analysis...")
        
        # Run all detection methods
        self.suspicious_patterns['self_trades'] = self.detect_self_trades()
        self.suspicious_patterns['repeated_pairs'] = self.detect_repeated_pairs()
        self.suspicious_patterns['circular_trading'] = self.detect_circular_trading()
        self.suspicious_patterns['timing_patterns'] = self.detect_timing_patterns()
        self.suspicious_patterns['volume_concentration'] = self.detect_volume_anomalies()
        self.suspicious_patterns['price_manipulation'] = self.detect_price_manipulation()
        self.suspicious_patterns['new_wallet_patterns'] = self.detect_new_wallet_patterns()
        
        return self.suspicious_patterns
    
    def get_summary_report(self):
        """Generate a comprehensive summary report"""
        if not self.suspicious_patterns:
            self.analyze_all_patterns()
        
        # Collect all suspicious elements
        all_suspicious_txs = set()
        all_suspicious_wallets = set()
        all_suspicious_tokens = set()
        
        high_severity_patterns = []
        medium_severity_patterns = []
        
        for pattern_name, pattern_data in self.suspicious_patterns.items():
            if pattern_data['severity'] == 'HIGH':
                high_severity_patterns.append(pattern_name)
            elif pattern_data['severity'] == 'MEDIUM':
                medium_severity_patterns.append(pattern_name)
            
            # Collect suspicious elements
            if 'transactions' in pattern_data:
                all_suspicious_txs.update(pattern_data['transactions'])
            if 'wallets' in pattern_data:
                all_suspicious_wallets.update(pattern_data['wallets'])
            if 'tokens' in pattern_data:
                all_suspicious_tokens.update(pattern_data['tokens'])
        
        # Calculate risk score
        risk_score = len(high_severity_patterns) * 3 + len(medium_severity_patterns) * 1
        
        return {
            'total_trades_analyzed': len(self.df),
            'suspicious_transactions': len(all_suspicious_txs),
            'suspicious_wallets': len(all_suspicious_wallets),
            'suspicious_tokens': len(all_suspicious_tokens),
            'high_severity_patterns': high_severity_patterns,
            'medium_severity_patterns': medium_severity_patterns,
            'risk_score': risk_score,
            'risk_level': 'HIGH' if risk_score >= 6 else 'MEDIUM' if risk_score >= 3 else 'LOW',
            'suspicious_tx_list': list(all_suspicious_txs)[:20],  # Top 20 for display
            'suspicious_wallet_list': list(all_suspicious_wallets)[:10],  # Top 10 for display
            'suspicious_token_list': list(all_suspicious_tokens)[:5],  # Top 5 for display
            'detailed_patterns': self.suspicious_patterns
        }
    
    def export_report(self, filename='wash_trading_report.json'):
        """Export detailed report to JSON file"""
        report = self.get_summary_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Detailed report exported to {filename}")
        return report