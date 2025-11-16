"""
Nifty Options Scalping Strategy
Author: Your Name
Version: 1.0
Description: Automated scalping bot for NSE Nifty 50 options using Fyers API
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as time_class
import time as time_module
import pytz
import math
from fyers_apiv3 import fyersModel
from collections import deque, defaultdict
import platform
import warnings
warnings.filterwarnings('ignore')

# Import credentials from config file
try:
    from config import FYERS_CLIENT_ID, FYERS_ACCESS_TOKEN
except ImportError:
    print("❌ ERROR: config.py not found!")
    print("📝 Please copy config.example.py to config.py and add your credentials")
    print("   cp config.example.py config.py")
    exit(1)

try:
    import winsound
except ImportError:
    winsound = None

# ANSI Color Codes
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
WHITE = '\033[97m'
RESET = '\033[0m'


class ImprovedNiftyScalper:
    """
    Simplified Nifty Scalping Strategy - Secure Version
    """
    
    def __init__(self):
        # ============================================
        # FYERS API CREDENTIALS - LOADED FROM CONFIG
        # ============================================
        self.client_id = FYERS_CLIENT_ID
        self.access_token = FYERS_ACCESS_TOKEN
        
        # Validate credentials
        if not self.client_id or not self.access_token:
            print(f"{RED}❌ Invalid credentials in config.py{RESET}")
            print(f"{YELLOW}Please add your Fyers API credentials to config.py{RESET}")
            self.test_mode = True
        else:
            # Test mode (set True to use synthetic data, False for live)
            self.test_mode = True
        
        # Initialize Fyers API
        if not self.test_mode:
            try:
                self.fyers = fyersModel.FyersModel(
                    client_id=self.client_id, 
                    token=self.access_token, 
                    log_path=""
                )
                print(f"{GREEN}✅ Connected to Fyers API{RESET}")
            except Exception as e:
                print(f"{YELLOW}⚠️  API connection failed: {e}. Using test mode.{RESET}")
                self.test_mode = True
                self.fyers = None
        else:
            self.fyers = None
            print(f"{BLUE}📊 Running in TEST MODE (synthetic data){RESET}")
        
        # Core config
        self.nifty_symbol = "NSE:NIFTY50-INDEX"
        self.option_base = "NSE:NIFTY"
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        
        # ============================================
        # CAPITAL MANAGEMENT
        # ============================================
        self.initial_balance = 3000.0
        self.balance = self.initial_balance
        self.target_return = 50.0
        self.target_balance = self.initial_balance * (1 + self.target_return / 100)
        
        # Position sizing
        self.max_positions = 2
        self.risk_per_trade = 15.0  # Percentage
        self.max_position_value = 1200.0
        
        # ============================================
        # SCALPING PARAMETERS - FIXED RISK-REWARD
        # ============================================
        self.profit_target = 15.0      # 15% profit target
        self.stop_loss = 7.0           # 7% stop loss
        self.time_exit_seconds = 600   # 10 minutes
        self.trailing_stop_trigger = 8.0   # Activate trailing at 8%
        self.trailing_stop_distance = 4.0  # Trail by 4%
        
        # Validate risk-reward ratio
        self.risk_reward_ratio = self.profit_target / self.stop_loss
        if self.risk_reward_ratio < 1.5:
            print(f"{YELLOW}⚠️  Warning: Risk-reward ratio {self.risk_reward_ratio:.2f} is below 1.5{RESET}")
        else:
            print(f"{GREEN}✅ Risk-Reward Ratio: 1:{self.risk_reward_ratio:.2f}{RESET}")
        
        # ============================================
        # STRIKE SELECTION
        # ============================================
        self.strike_selection_mode = "OTM"  # ATM, OTM, ITM
        self.min_option_price = 10.0
        self.max_option_price = 200.0
        self.min_oi = 5000
        self.min_volume = 200
        
        # ============================================
        # TECHNICAL INDICATORS
        # ============================================
        self.ema_fast = 3
        self.ema_medium = 8
        self.ema_slow = 21
        self.rsi_period = 14
        self.rsi_oversold = 30.0
        self.rsi_overbought = 70.0
        
        # Signal threshold
        self.signal_threshold = 50  # Minimum score needed
        
        # ============================================
        # RISK MANAGEMENT
        # ============================================
        self.max_daily_loss = 600.0
        self.max_daily_profit = 1500.0
        self.max_daily_trades = 20
        self.max_consecutive_losses = 4
        self.cooldown_seconds = 30
        
        # Transaction costs
        self.brokerage_per_order = 20.0
        self.stt_rate = 0.0005
        self.exchange_charges = 0.00035
        self.slippage = 0.005  # 0.5% slippage
        
        # ============================================
        # DATA STORAGE
        # ============================================
        self.nifty_data = pd.DataFrame()
        self.option_chain = {}
        self.positions = {}
        self.trades = []
        self.recent_trades = deque(maxlen=15)
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        
        # Control flags
        self.is_running = True
        self.trading_enabled = True
        self.target_reached = False
        self.circuit_breaker_active = False
        self.data_ready = False
        
        # Daily reset
        self.daily_reset_date = None
        self.last_nifty_price = None
        
        # Debug
        self.debug_mode = True
        self.iteration_count = 0
        self.signal_history = deque(maxlen=20)
        self.last_signal_time = {}
        
        self._print_initialization()
    
    def _print_initialization(self):
        """Print startup configuration"""
        print(f"\n{MAGENTA}{'='*80}{RESET}")
        print(f"{MAGENTA}🚀 IMPROVED NIFTY SCALPING STRATEGY{RESET}")
        print(f"{MAGENTA}{'='*80}{RESET}")
        
        print(f"\n{CYAN}💰 CAPITAL SETTINGS:{RESET}")
        print(f"{CYAN}   Initial Balance: ₹{self.initial_balance:,.2f}{RESET}")
        print(f"{CYAN}   Target Return: {self.target_return}% (₹{self.target_balance:,.2f}){RESET}")
        print(f"{CYAN}   Risk per Trade: {self.risk_per_trade}%{RESET}")
        print(f"{CYAN}   Max Position Value: ₹{self.max_position_value:,.2f}{RESET}")
        
        print(f"\n{CYAN}⚡ SCALPING PARAMETERS:{RESET}")
        print(f"{GREEN}   Profit Target: {self.profit_target}%{RESET}")
        print(f"{RED}   Stop Loss: {self.stop_loss}%{RESET}")
        print(f"{BLUE}   Risk-Reward Ratio: 1:{self.risk_reward_ratio:.2f}{RESET}")
        print(f"{CYAN}   Time Exit: {self.time_exit_seconds}s ({self.time_exit_seconds//60} min){RESET}")
        print(f"{CYAN}   Trailing: Trigger {self.trailing_stop_trigger}%, Distance {self.trailing_stop_distance}%{RESET}")
        
        print(f"\n{CYAN}📊 TECHNICAL INDICATORS:{RESET}")
        print(f"{CYAN}   EMAs: {self.ema_fast}/{self.ema_medium}/{self.ema_slow}{RESET}")
        print(f"{CYAN}   RSI: Period {self.rsi_period} (OS:{self.rsi_oversold}, OB:{self.rsi_overbought}){RESET}")
        print(f"{CYAN}   Signal Threshold: {self.signal_threshold} points{RESET}")
        
        print(f"\n{CYAN}🛡️ RISK CONTROLS:{RESET}")
        print(f"{CYAN}   Max Daily Loss: ₹{self.max_daily_loss:,.2f}{RESET}")
        print(f"{CYAN}   Max Daily Profit: ₹{self.max_daily_profit:,.2f}{RESET}")
        print(f"{CYAN}   Max Daily Trades: {self.max_daily_trades}{RESET}")
        print(f"{CYAN}   Max Consecutive Losses: {self.max_consecutive_losses}{RESET}")
        
        print(f"\n{CYAN}🎯 OPTION SELECTION:{RESET}")
        print(f"{CYAN}   Mode: {self.strike_selection_mode}{RESET}")
        print(f"{CYAN}   Price Range: ₹{self.min_option_price} - ₹{self.max_option_price}{RESET}")
        print(f"{CYAN}   Min OI: {self.min_oi:,} | Min Volume: {self.min_volume:,}{RESET}")
        
        print(f"\n{GREEN}✅ FEATURES:{RESET}")
        print(f"{GREEN}   ✓ Robust signal generation{RESET}")
        print(f"{GREEN}   ✓ Multi-level risk management{RESET}")
        print(f"{GREEN}   ✓ Trailing stop loss{RESET}")
        print(f"{GREEN}   ✓ Circuit breaker protection{RESET}")
        print(f"{GREEN}   ✓ Transaction cost accounting{RESET}")
        
        print(f"{MAGENTA}{'='*80}{RESET}\n")
    
    def debug_log(self, message, color=BLUE):
        """Print debug messages"""
        if self.debug_mode:
            timestamp = datetime.now(self.ist_tz).strftime('%H:%M:%S')
            print(f"{color}[DEBUG {timestamp}] {message}{RESET}")
    
    # ============================================
    # TIME & MARKET CHECKS
    # ============================================
    
    def is_market_open(self):
        """Check if market is open"""
        if self.test_mode:
            return True  # Always open in test mode
        
        now = datetime.now(self.ist_tz)
        current_time = now.time()
        
        # Weekend check
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Market hours: 9:15 AM to 3:30 PM
        market_open = time_class(9, 15)
        market_close = time_class(15, 30)
        
        return market_open <= current_time <= market_close
    
    def is_scalping_hours(self):
        """Check if within scalping hours"""
        if self.test_mode:
            return True
        
        now = datetime.now(self.ist_tz).time()
        
        # Scalping hours: 9:30 AM to 3:15 PM
        start = time_class(9, 30)
        end = time_class(15, 15)
        
        return start <= now <= end
    
    def reset_daily_metrics(self):
        """Reset daily tracking"""
        today = datetime.now(self.ist_tz).date()
        
        if self.daily_reset_date != today:
            if self.daily_pnl != 0:
                print(f"\n{BLUE}{'='*70}{RESET}")
                print(f"{BLUE}📊 PREVIOUS DAY SUMMARY{RESET}")
                print(f"{BLUE}   Date: {self.daily_reset_date}{RESET}")
                print(f"{BLUE}   P&L: ₹{self.daily_pnl:,.2f}{RESET}")
                print(f"{BLUE}   Trades: {self.daily_trades}{RESET}")
                print(f"{BLUE}{'='*70}{RESET}\n")
            
            # Reset metrics
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.consecutive_losses = 0
            self.consecutive_wins = 0
            self.circuit_breaker_active = False
            self.daily_reset_date = today
            
            print(f"{GREEN}🔄 New Trading Day: {today.strftime('%Y-%m-%d')}{RESET}\n")
    
    def get_expiry_date(self):
        """Get next Thursday expiry (Nifty weekly)"""
        today = datetime.now(self.ist_tz).date()
        
        # Days until Thursday (3 = Thursday)
        days_ahead = 3 - today.weekday()
        
        if days_ahead <= 0:  # If today is Fri/Sat/Sun/Thu after 3:30
            days_ahead += 7
        
        expiry = today + timedelta(days=days_ahead)
        return expiry.strftime('%y%m%d')
    
    # ============================================
    # DATA GENERATION & FETCHING
    # ============================================
    
    def create_synthetic_trending_data(self, trend_type=None):
        """Create realistic trending data for testing"""
        print(f"{BLUE}📊 Generating synthetic market data...{RESET}")
        
        now = datetime.now(self.ist_tz)
        start = now - timedelta(hours=2)
        timestamps = pd.date_range(start=start, end=now, freq='1min')
        
        # Random trend if not specified
        if trend_type is None:
            trend_type = np.random.choice(['BULLISH', 'BEARISH', 'SIDEWAYS'], p=[0.4, 0.4, 0.2])
        
        print(f"{YELLOW}📈 Trend Type: {trend_type}{RESET}")
        
        base_price = 25000
        prices = [base_price]
        
        # Generate price movement
        for i in range(1, len(timestamps)):
            if trend_type == 'BULLISH':
                trend_component = np.random.uniform(0.5, 1.5)
                noise = np.random.normal(0, 2.0)
            elif trend_type == 'BEARISH':
                trend_component = -np.random.uniform(0.5, 1.5)
                noise = np.random.normal(0, 2.0)
            else:  # SIDEWAYS
                trend_component = np.random.uniform(-0.3, 0.3)
                noise = np.random.normal(0, 2.5)
            
            change = trend_component + noise
            new_price = prices[-1] + change
            
            # Keep within bounds
            new_price = max(new_price, base_price * 0.97)
            new_price = min(new_price, base_price * 1.03)
            
            prices.append(new_price)
        
        # Create OHLC data
        data = []
        for i, close_price in enumerate(prices):
            volatility = np.random.uniform(2, 6)
            open_price = prices[i-1] if i > 0 else close_price
            
            high_price = max(open_price, close_price) + volatility
            low_price = min(open_price, close_price) - volatility
            volume = np.random.randint(400000, 900000)
            
            data.append([open_price, high_price, low_price, close_price, volume])
        
        # Create DataFrame
        df = pd.DataFrame(
            data, 
            columns=['open', 'high', 'low', 'close', 'volume'], 
            index=timestamps
        )
        
        # Clean any NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        self.nifty_data = df
        self.data_ready = True
        
        # Print statistics
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        change = end_price - start_price
        change_pct = (change / start_price) * 100
        
        print(f"{GREEN}✅ Generated {len(df)} candles{RESET}")
        print(f"{CYAN}   Start: {start_price:.2f} | End: {end_price:.2f}{RESET}")
        print(f"{CYAN}   Change: {change:+.2f} ({change_pct:+.2f}%){RESET}")
        print(f"{CYAN}   High: {df['high'].max():.2f} | Low: {df['low'].min():.2f}{RESET}\n")
        
        return True
    
    def fetch_nifty_data(self):
        """Fetch Nifty data - try real API, fallback to synthetic"""
        if self.test_mode or self.fyers is None:
            return self.create_synthetic_trending_data()
        
        try:
            end_date = datetime.now(self.ist_tz)
            start_date = end_date - timedelta(days=5)
            
            data_request = {
                "symbol": self.nifty_symbol,
                "resolution": "1",
                "date_format": "1",
                "range_from": start_date.strftime('%Y-%m-%d'),
                "range_to": end_date.strftime('%Y-%m-%d'),
                "cont_flag": "1"
            }
            
            self.debug_log("Fetching real market data from Fyers...")
            response = self.fyers.history(data_request)
            
            if response and response.get('s') == 'ok' and 'candles' in response:
                candles = response['candles']
                
                if len(candles) > 50:
                    df = pd.DataFrame(
                        candles, 
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(self.ist_tz)
                    df.set_index('timestamp', inplace=True)
                    
                    # Clean data
                    df = df.replace([np.inf, -np.inf], np.nan)
                    df = df.fillna(method='ffill').fillna(method='bfill')
                    
                    self.nifty_data = df.tail(200)
                    self.data_ready = True
                    
                    print(f"{GREEN}✅ Loaded {len(self.nifty_data)} real candles from Fyers{RESET}")
                    return True
            
            self.debug_log("Real data not available, using synthetic", YELLOW)
            return self.create_synthetic_trending_data()
            
        except Exception as e:
            self.debug_log(f"API error: {e}. Using synthetic data.", YELLOW)
            return self.create_synthetic_trending_data()
    
    def get_current_nifty_price(self):
        """Get current Nifty price"""
        if not self.test_mode and self.fyers:
            try:
                response = self.fyers.quotes({"symbols": self.nifty_symbol})
                
                if response and response.get('s') == 'ok' and 'd' in response:
                    ltp = response['d'][0]['v'].get('lp', 0)
                    if ltp > 0:
                        self.last_nifty_price = ltp
                        return ltp
            except:
                pass
        
        # Fallback to cached price
        if self.last_nifty_price and self.last_nifty_price > 0:
            return self.last_nifty_price
        
        # Fallback to data
        if not self.nifty_data.empty:
            return float(self.nifty_data['close'].iloc[-1])
        
        return 25000.0
    
    # ============================================
    # OPTION CHAIN
    # ============================================
    
    def get_atm_strike(self, price):
        """Get ATM strike (rounded to nearest 50)"""
        return round(price / 50) * 50
    
    def calculate_realistic_option_price(self, spot, strike, option_type):
        """Calculate realistic option price"""
        # Intrinsic value
        if option_type == 'CE':
            intrinsic = max(0, spot - strike)
        else:  # PE
            intrinsic = max(0, strike - spot)
        
        # Time value based on moneyness
        moneyness = abs(spot - strike)
        
        if moneyness == 0:  # ATM
            time_value = np.random.uniform(35, 55)
        elif moneyness <= 50:
            time_value = np.random.uniform(25, 40)
        elif moneyness <= 100:
            time_value = np.random.uniform(15, 30)
        elif moneyness <= 150:
            time_value = np.random.uniform(8, 20)
        else:
            time_value = np.random.uniform(3, 12)
        
        # Add small random variation
        noise = np.random.uniform(-2, 2)
        
        price = intrinsic + time_value + noise
        
        return max(5.0, round(price, 2))
    
    def create_synthetic_option_chain(self, nifty_price):
        """Create synthetic option chain"""
        atm = self.get_atm_strike(nifty_price)
        expiry = self.get_expiry_date()
        
        self.debug_log(f"Creating option chain | Spot: {nifty_price:.2f} | ATM: {atm}")
        
        option_chain = {}
        
        # Generate strikes (-5 to +5 around ATM)
        for i in range(-5, 6):
            strike = atm + (i * 50)
            
            # Call option
            call_price = self.calculate_realistic_option_price(nifty_price, strike, 'CE')
            call_symbol = f"{self.option_base}{expiry}{strike}CE"
            
            # Put option
            put_price = self.calculate_realistic_option_price(nifty_price, strike, 'PE')
            put_symbol = f"{self.option_base}{expiry}{strike}PE"
            
            # Greeks calculation (simplified)
            if i == 0:  # ATM
                call_delta, put_delta = 0.50, -0.50
                oi_multiplier = 1.5
            elif i > 0:  # OTM calls, ITM puts
                call_delta = max(0.05, 0.50 - (i * 0.12))
                put_delta = min(-0.95, -0.50 - (i * 0.12))
                oi_multiplier = max(0.5, 1.0 - (i * 0.1))
            else:  # ITM calls, OTM puts
                call_delta = min(0.95, 0.50 + (abs(i) * 0.12))
                put_delta = max(-0.05, -0.50 + (abs(i) * 0.12))
                oi_multiplier = max(0.5, 1.0 - (abs(i) * 0.1))
            
            base_oi = np.random.randint(12000, 45000)
            base_volume = np.random.randint(1000, 6000)
            
            # Call option data
            option_chain[call_symbol] = {
                'symbol': call_symbol,
                'strike': strike,
                'type': 'CE',
                'ltp': call_price,
                'bid': call_price * 0.997,
                'ask': call_price * 1.003,
                'oi': int(base_oi * oi_multiplier),
                'volume': int(base_volume * oi_multiplier),
                'delta': round(call_delta, 3),
                'gamma': 0.02,
                'theta': -0.02,
                'vega': 0.15
            }
            
            # Put option data
            option_chain[put_symbol] = {
                'symbol': put_symbol,
                'strike': strike,
                'type': 'PE',
                'ltp': put_price,
                'bid': put_price * 0.997,
                'ask': put_price * 1.003,
                'oi': int(base_oi * oi_multiplier),
                'volume': int(base_volume * oi_multiplier),
                'delta': round(put_delta, 3),
                'gamma': 0.02,
                'theta': -0.02,
                'vega': 0.15
            }
        
        self.option_chain = option_chain
        self.debug_log(f"Created {len(option_chain)} option contracts")
        
        return True
    
    def fetch_option_chain(self, nifty_price):
        """Fetch option chain - real or synthetic"""
        if self.test_mode or self.fyers is None:
            return self.create_synthetic_option_chain(nifty_price)
        
        try:
            data = {
                "symbol": self.nifty_symbol,
                "strikecount": 11,
                "timestamp": ""
            }
            response = self.fyers.optionchain(data=data)
            
            if response and response.get('s') == 'ok' and 'data' in response:
                self.option_chain = {}
                for opt in response['data'].get('options', []):
                    symbol = opt['symbol']
                    self.option_chain[symbol] = {
                        'symbol': symbol,
                        'strike': opt.get('strike_price', 0),
                        'type': opt.get('option_type', ''),
                        'ltp': opt.get('ltp', 0.0),
                        'bid': opt.get('bid_price', 0.0),
                        'ask': opt.get('ask_price', 0.0),
                        'oi': opt.get('open_interest', 0),
                        'volume': opt.get('volume', 0),
                        'delta': opt.get('delta', 0),
                        'gamma': opt.get('gamma', 0),
                        'theta': opt.get('theta', 0),
                        'vega': opt.get('vega', 0)
                    }
                
                self.debug_log(f"Fetched {len(self.option_chain)} real options")
                return True
            else:
                return self.create_synthetic_option_chain(nifty_price)
                
        except Exception as e:
            self.debug_log(f"Option chain fetch error: {e}", YELLOW)
            return self.create_synthetic_option_chain(nifty_price)
    
    # ============================================
    # STRIKE SELECTION
    # ============================================
    
    def select_best_strike(self, signal, nifty_price):
        """Select best strike based on signal and mode"""
        atm = self.get_atm_strike(nifty_price)
        option_type = 'CE' if signal == 'BUY_CALL' else 'PE'
        
        self.debug_log(f"Selecting {option_type} strike | ATM: {atm} | Mode: {self.strike_selection_mode}")
        
        # Filter options by type
        candidates = [
            (symbol, data) for symbol, data in self.option_chain.items()
            if data['type'] == option_type
        ]
        
        if not candidates:
            self.debug_log("No candidates found!", RED)
            return None
        
        # Score each option
        scored = []
        
        for symbol, data in candidates:
            strike = data['strike']
            ltp = data['ltp']
            oi = data['oi']
            volume = data['volume']
            
            # Price filter
            if not (self.min_option_price <= ltp <= self.max_option_price):
                continue
            
            # Liquidity filter
            if oi < self.min_oi or volume < self.min_volume:
                continue
            
            # Distance from ATM
            distance = strike - atm
            
            # Score based on selection mode
            if self.strike_selection_mode == "OTM":
                # Prefer OTM options
                if signal == 'BUY_CALL' and option_type == 'CE':
                    if distance > 0:  # OTM CE
                        distance_score = 100 - max(0, (distance - 50) / 50 * 20)
                    elif distance == 0:  # ATM
                        distance_score = 60
                    else:  # ITM
                        distance_score = 30
                elif signal == 'BUY_PUT' and option_type == 'PE':
                    if distance < 0:  # OTM PE
                        distance_score = 100 - max(0, (abs(distance) - 50) / 50 * 20)
                    elif distance == 0:  # ATM
                        distance_score = 60
                    else:  # ITM
                        distance_score = 30
                else:
                    distance_score = 0
                    
            elif self.strike_selection_mode == "ATM":
                # Prefer ATM options
                abs_distance = abs(distance)
                if abs_distance == 0:
                    distance_score = 100
                elif abs_distance == 50:
                    distance_score = 75
                else:
                    distance_score = max(0, 50 - abs_distance / 10)
                    
            else:  # ITM
                # Prefer ITM options
                if signal == 'BUY_CALL' and option_type == 'CE':
                    if distance < 0:  # ITM CE
                        distance_score = 100
                    else:
                        distance_score = 50
                elif signal == 'BUY_PUT' and option_type == 'PE':
                    if distance > 0:  # ITM PE
                        distance_score = 100
                    else:
                        distance_score = 50
                else:
                    distance_score = 0
            
            # Liquidity bonus
            liquidity_score = min(30, (oi / 20000) * 15 + (volume / 3000) * 15)
            
            # Price preference (sweet spot)
            if 40 <= ltp <= 120:
                price_score = 20
            elif 20 <= ltp <= 150:
                price_score = 10
            else:
                price_score = 0
            
            total_score = distance_score + liquidity_score + price_score
            
            scored.append({
                'symbol': symbol,
                'data': data,
                'score': total_score,
                'distance': distance
            })
        
        if not scored:
            self.debug_log("No suitable options after filtering", YELLOW)
            return None
        
        # Select best option
        scored.sort(key=lambda x: (-x['score'], abs(x['distance'])))
        best = scored[0]
        
        print(f"\n{GREEN}✅ SELECTED STRIKE:{RESET}")
        print(f"{GREEN}   Symbol: {best['symbol']}{RESET}")
        print(f"{GREEN}   Strike: {best['data']['strike']} | Type: {best['data']['type']}{RESET}")
        print(f"{GREEN}   LTP: ₹{best['data']['ltp']:.2f}{RESET}")
        print(f"{GREEN}   OI: {best['data']['oi']:,} | Volume: {best['data']['volume']:,}{RESET}")
        print(f"{GREEN}   Score: {best['score']:.1f}{RESET}\n")
        
        return best['symbol'], best['data']
    
    # ============================================
    # TECHNICAL INDICATORS
    # ============================================
    
    def calculate_indicators(self):
        """Calculate technical indicators with error handling"""
        required_length = max(self.ema_slow, self.rsi_period) + 5
        
        if len(self.nifty_data) < required_length:
            self.debug_log(f"Insufficient data: {len(self.nifty_data)} < {required_length}", YELLOW)
            return None
        
        try:
            close = self.nifty_data['close'].copy()
            
            # Handle NaN
            if close.isna().any():
                self.debug_log("NaN values detected, cleaning...", YELLOW)
                close = close.fillna(method='ffill').fillna(method='bfill')
            
            # Calculate EMAs
            ema_fast = close.ewm(span=self.ema_fast, adjust=False).mean().iloc[-1]
            ema_medium = close.ewm(span=self.ema_medium, adjust=False).mean().iloc[-1]
            ema_slow = close.ewm(span=self.ema_slow, adjust=False).mean().iloc[-1]
            
            # Calculate RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            
            rs = gain / loss.replace(0, 0.001)
            rsi_series = 100 - (100 / (1 + rs))
            rsi = rsi_series.iloc[-1]
            
            # Handle NaN RSI
            if pd.isna(rsi) or not np.isfinite(rsi):
                rsi = 50.0
                self.debug_log("RSI invalid, using neutral 50", YELLOW)
            
            # Calculate Momentum
            lookback = min(10, len(close) - 1)
            momentum = ((close.iloc[-1] - close.iloc[-lookback]) / close.iloc[-lookback]) * 100
            
            # Volume analysis
            avg_volume = self.nifty_data['volume'].tail(20).mean()
            current_volume = self.nifty_data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            current_price = float(close.iloc[-1])
            
            indicators = {
                'ema_fast': float(ema_fast),
                'ema_medium': float(ema_medium),
                'ema_slow': float(ema_slow),
                'rsi': float(rsi),
                'momentum': float(momentum),
                'current_price': current_price,
                'volume_ratio': float(volume_ratio)
            }
            
            # Validate all values
            for key, value in indicators.items():
                if pd.isna(value) or not np.isfinite(value):
                    self.debug_log(f"Invalid {key}: {value}, using fallback", YELLOW)
                    if key == 'rsi':
                        indicators[key] = 50.0
                    elif 'ema' in key or key == 'current_price':
                        indicators[key] = current_price
                    else:
                        indicators[key] = 0.0
            
            return indicators
            
        except Exception as e:
            self.debug_log(f"Indicator calculation error: {e}", RED)
            import traceback
            traceback.print_exc()
            return None
    
    # ============================================
    # SIGNAL GENERATION
    # ============================================
    
    def generate_signal(self, indicators):
        """Generate trading signal based on indicators"""
        
        if not indicators:
            return "HOLD"
        
        if not self.trading_enabled or self.circuit_breaker_active:
            return "HOLD"
        
        if not self.is_scalping_hours():
            return "HOLD"
        
        # Extract indicators
        ema_fast = indicators['ema_fast']
        ema_medium = indicators['ema_medium']
        ema_slow = indicators['ema_slow']
        rsi = indicators['rsi']
        momentum = indicators['momentum']
        current = indicators['current_price']
        volume_ratio = indicators.get('volume_ratio', 1.0)
        
        # Initialize scores
        bullish_score = 0
        bearish_score = 0
        
        print(f"\n{CYAN}{'='*70}{RESET}")
        print(f"{CYAN}🔍 SIGNAL ANALYSIS{RESET}")
        print(f"{CYAN}{'='*70}{RESET}")
        print(f"{BLUE}Price: {current:.2f}{RESET}")
        print(f"{BLUE}EMA: Fast={ema_fast:.2f} | Med={ema_medium:.2f} | Slow={ema_slow:.2f}{RESET}")
        print(f"{BLUE}RSI: {rsi:.1f} | Momentum: {momentum:+.2f}% | Volume: {volume_ratio:.2f}x{RESET}\n")
        
        # 1. EMA Alignment (40 points)
        print(f"{WHITE}1. EMA ALIGNMENT (40 pts):{RESET}")
        if ema_fast > ema_medium > ema_slow:
            bullish_score += 40
            print(f"{GREEN}   ✓ Strong Bullish (+40){RESET}")
        elif ema_fast < ema_medium < ema_slow:
            bearish_score += 40
            print(f"{RED}   ✓ Strong Bearish (+40){RESET}")
        elif ema_fast > ema_slow:
            bullish_score += 20
            print(f"{GREEN}   ✓ Partial Bullish (+20){RESET}")
        elif ema_fast < ema_slow:
            bearish_score += 20
            print(f"{RED}   ✓ Partial Bearish (+20){RESET}")
        else:
            print(f"{YELLOW}   - Neutral (0){RESET}")
        
        # 2. RSI (30 points)
        print(f"\n{WHITE}2. RSI ANALYSIS (30 pts):{RESET}")
        if rsi < self.rsi_oversold:
            bullish_score += 30
            print(f"{GREEN}   ✓ Oversold: {rsi:.1f} (+30){RESET}")
        elif rsi < 45:
            bullish_score += 20
            print(f"{GREEN}   ✓ Bullish Zone: {rsi:.1f} (+20){RESET}")
        elif rsi > self.rsi_overbought:
            bearish_score += 30
            print(f"{RED}   ✓ Overbought: {rsi:.1f} (+30){RESET}")
        elif rsi > 55:
            bearish_score += 20
            print(f"{RED}   ✓ Bearish Zone: {rsi:.1f} (+20){RESET}")
        else:
            print(f"{YELLOW}   - Neutral: {rsi:.1f} (0){RESET}")
        
        # 3. Momentum (20 points)
        print(f"\n{WHITE}3. MOMENTUM (20 pts):{RESET}")
        if momentum > 0.1:
            bullish_score += 20
            print(f"{GREEN}   ✓ Strong Positive: {momentum:+.2f}% (+20){RESET}")
        elif momentum > 0:
            bullish_score += 10
            print(f"{GREEN}   ✓ Positive: {momentum:+.2f}% (+10){RESET}")
        elif momentum < -0.1:
            bearish_score += 20
            print(f"{RED}   ✓ Strong Negative: {momentum:+.2f}% (+20){RESET}")
        elif momentum < 0:
            bearish_score += 10
            print(f"{RED}   ✓ Negative: {momentum:+.2f}% (+10){RESET}")
        else:
            print(f"{YELLOW}   - Flat: {momentum:+.2f}% (0){RESET}")
        
        # 4. Volume (10 points)
        print(f"\n{WHITE}4. VOLUME CONFIRMATION (10 pts):{RESET}")
        if volume_ratio > 1.5:
            if bullish_score > bearish_score:
                bullish_score += 10
                print(f"{GREEN}   ✓ High Volume - Bullish (+10){RESET}")
            else:
                bearish_score += 10
                print(f"{RED}   ✓ High Volume - Bearish (+10){RESET}")
        else:
            print(f"{YELLOW}   - Normal Volume (0){RESET}")
        
        # Final Decision
        print(f"\n{CYAN}{'='*70}{RESET}")
        print(f"{CYAN}📊 FINAL SCORES:{RESET}")
        print(f"{GREEN}   BULLISH: {bullish_score}/100{RESET}")
        print(f"{RED}   BEARISH: {bearish_score}/100{RESET}")
        print(f"{BLUE}   THRESHOLD: {self.signal_threshold}{RESET}")
        
        signal = "HOLD"
        
        if bullish_score >= self.signal_threshold and bullish_score > bearish_score:
            signal = "BUY_CALL"
            print(f"\n{GREEN}{'='*70}{RESET}")
            print(f"{GREEN}🚀 SIGNAL: BUY CALL (Score: {bullish_score}){RESET}")
            print(f"{GREEN}{'='*70}{RESET}\n")
            
        elif bearish_score >= self.signal_threshold and bearish_score > bullish_score:
            signal = "BUY_PUT"
            print(f"\n{RED}{'='*70}{RESET}")
            print(f"{RED}🔻 SIGNAL: BUY PUT (Score: {bearish_score}){RESET}")
            print(f"{RED}{'='*70}{RESET}\n")
            
        else:
            print(f"\n{YELLOW}{'='*70}{RESET}")
            print(f"{YELLOW}⏸️  SIGNAL: HOLD (Insufficient score){RESET}")
            print(f"{YELLOW}{'='*70}{RESET}\n")
        
        # Store in history
        self.signal_history.append({
            'time': datetime.now(self.ist_tz),
            'signal': signal,
            'bullish': bullish_score,
            'bearish': bearish_score,
            'rsi': rsi,
            'momentum': momentum
        })
        
        return signal
    
    # ============================================
    # POSITION MANAGEMENT
    # ============================================
    
    def calculate_transaction_costs(self, trade_value, is_entry=True):
        """Calculate all transaction costs"""
        costs = self.brokerage_per_order
        
        if not is_entry:
            costs += trade_value * self.stt_rate
        
        costs += trade_value * self.exchange_charges
        costs += trade_value * self.slippage
        
        return round(costs, 2)
    
    def calculate_position_size(self, option_price):
        """Calculate position size in lots"""
        lot_size = 50
        
        risk_amount = min(
            self.balance * (self.risk_per_trade / 100),
            self.max_position_value
        )
        
        max_lots = int(risk_amount / (option_price * lot_size * 1.02))
        
        return max(1, min(max_lots, 3))
    
    def execute_trade(self, signal, nifty_price):
        """Execute a trade"""
        if not self.trading_enabled or self.circuit_breaker_active:
            return False
        
        if len(self.positions) >= self.max_positions:
            self.debug_log(f"Max positions reached: {len(self.positions)}/{self.max_positions}", YELLOW)
            return False
        
        # Select strike
        selection = self.select_best_strike(signal, nifty_price)
        
        if not selection:
            return False
        
        symbol, option_data = selection
        
        entry_price = option_data['ltp']
        strike = option_data['strike']
        quantity = self.calculate_position_size(entry_price)
        
        trade_value = entry_price * quantity * 50
        entry_costs = self.calculate_transaction_costs(trade_value, True)
        
        total_required = trade_value + entry_costs
        
        if total_required > self.balance:
            self.debug_log(f"Insufficient balance: need ₹{total_required:.2f}, have ₹{self.balance:.2f}", YELLOW)
            return False
        
        # Deduct costs from balance
        self.balance -= entry_costs
        
        # Create position
        position = {
            'symbol': symbol,
            'signal': signal,
            'strike': strike,
            'option_type': option_data['type'],
            'entry_price': entry_price,
            'current_price': entry_price,
            'quantity': quantity,
            'entry_time': datetime.now(self.ist_tz),
            'nifty_entry': nifty_price,
            'trade_value': trade_value,
            'entry_costs': entry_costs,
            'status': 'OPEN',
            'highest_price': entry_price,
            'trailing_stop_active': False,
            'delta': option_data.get('delta', 0)
        }
        
        self.positions[symbol] = position
        self.trades.append(position.copy())
        self.daily_trades += 1
        
        # Recent trades
        self.recent_trades.append({
            'time': datetime.now(self.ist_tz).strftime('%H:%M:%S'),
            'signal': signal,
            'strike': strike,
            'type': option_data['type'],
            'entry': entry_price,
            'qty': quantity,
            'status': 'OPENED'
        })
        
        # Play alert sound
        self.play_sound(signal)
        
        # Print trade details
        print(f"\n{MAGENTA}{'='*80}{RESET}")
        print(f"{MAGENTA}🚨 NEW TRADE OPENED 🚨{RESET}")
        print(f"{MAGENTA}{'='*80}{RESET}")
        print(f"{MAGENTA}Signal: {signal} | Symbol: {symbol}{RESET}")
        print(f"{MAGENTA}Strike: {strike} | Type: {option_data['type']} | Entry: ₹{entry_price:.2f}{RESET}")
        print(f"{MAGENTA}Quantity: {quantity} lots ({quantity*50} units) | Value: ₹{trade_value:,.2f}{RESET}")
        print(f"{MAGENTA}Entry Costs: ₹{entry_costs:.2f} | Balance: ₹{self.balance:,.2f}{RESET}")
        print(f"{MAGENTA}{'='*80}{RESET}")
        print(f"{GREEN}Targets:{RESET}")
        profit_target_value = trade_value * (self.profit_target / 100)
        stop_loss_value = trade_value * (self.stop_loss / 100)
        print(f"{GREEN}  Profit: {self.profit_target}% = ₹{profit_target_value:.0f}{RESET}")
        print(f"{RED}  Stop: {self.stop_loss}% = ₹{stop_loss_value:.0f}{RESET}")
        print(f"{BLUE}  Time Exit: {self.time_exit_seconds}s ({self.time_exit_seconds//60} min){RESET}")
        print(f"{MAGENTA}{'='*80}{RESET}\n")
        
        return True
    
    def check_exits(self):
        """Check and execute exit conditions"""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol not in self.option_chain:
                continue
            
            current_price = self.option_chain[symbol]['ltp']
            entry_price = position['entry_price']
            
            position['current_price'] = current_price
            
            # Calculate P&L
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            pnl_value = (pnl_pct / 100) * position['trade_value']
            
            # Update highest price
            if current_price > position['highest_price']:
                position['highest_price'] = current_price
            
            # Activate trailing stop
            if pnl_pct >= self.trailing_stop_trigger and not position['trailing_stop_active']:
                position['trailing_stop_active'] = True
                print(f"{GREEN}✅ Trailing stop ACTIVATED for {symbol} @ ₹{current_price:.2f} ({pnl_pct:+.1f}%){RESET}")
            
            should_exit = False
            reason = ""
            
            # 1. Profit target
            if pnl_pct >= self.profit_target:
                should_exit = True
                reason = "PROFIT_TARGET"
            
            # 2. Trailing stop
            elif position['trailing_stop_active']:
                drawdown_from_peak = ((position['highest_price'] - current_price) / position['highest_price']) * 100
                if drawdown_from_peak >= self.trailing_stop_distance:
                    should_exit = True
                    reason = "TRAILING_STOP"
            
            # 3. Stop loss
            elif pnl_pct <= -self.stop_loss:
                should_exit = True
                reason = "STOP_LOSS"
            
            # 4. Time exit
            hold_time = (datetime.now(self.ist_tz) - position['entry_time']).seconds
            if hold_time >= self.time_exit_seconds:
                should_exit = True
                reason = "TIME_EXIT"
            
            if should_exit:
                positions_to_close.append((symbol, current_price, pnl_pct, pnl_value, reason))
        
        # Close positions
        for symbol, exit_price, pnl_pct, pnl_value, reason in positions_to_close:
            self.close_position(symbol, exit_price, pnl_pct, pnl_value, reason)
    
    def close_position(self, symbol, exit_price, pnl_pct, pnl_value, reason):
        """Close a position"""
        position = self.positions[symbol]
        
        exit_trade_value = exit_price * position['quantity'] * 50
        exit_costs = self.calculate_transaction_costs(exit_trade_value, False)
        
        net_pnl = pnl_value - exit_costs
        
        # Update balance
        self.balance += exit_trade_value - exit_costs
        self.daily_pnl += net_pnl
        
        # Update streaks
        if net_pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        # Update position data
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now(self.ist_tz)
        position['exit_reason'] = reason
        position['exit_costs'] = exit_costs
        position['pnl'] = net_pnl
        position['pnl_pct'] = pnl_pct
        position['status'] = 'CLOSED'
        position['hold_time'] = (position['exit_time'] - position['entry_time']).seconds
        
        # Add to recent trades
        self.recent_trades.append({
            'time': datetime.now(self.ist_tz).strftime('%H:%M:%S'),
            'signal': position['signal'],
            'strike': position['strike'],
            'type': position['option_type'],
            'entry': position['entry_price'],
            'exit': exit_price,
            'qty': position['quantity'],
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'hold_time': position['hold_time'],
            'status': f'CLOSED-{reason}'
        })
        
        # Print closure details
        color = GREEN if net_pnl > 0 else RED
        emoji = "✅" if net_pnl > 0 else "❌"
        
        print(f"\n{color}{'='*80}{RESET}")
        print(f"{color}{emoji} POSITION CLOSED - {reason}{RESET}")
        print(f"{color}{'='*80}{RESET}")
        print(f"{color}Symbol: {symbol} | Strike: {position['strike']} | Type: {position['option_type']}{RESET}")
        print(f"{color}Entry: ₹{position['entry_price']:.2f} → Exit: ₹{exit_price:.2f} ({pnl_pct:+.2f}%){RESET}")
        print(f"{color}Hold Time: {position['hold_time']}s | Gross P&L: ₹{pnl_value:.2f}{RESET}")
        print(f"{color}Exit Costs: ₹{exit_costs:.2f} | Net P&L: ₹{net_pnl:,.2f}{RESET}")
        print(f"{color}Balance: ₹{self.balance:,.2f} | Daily P&L: ₹{self.daily_pnl:,.2f}{RESET}")
        print(f"{color}{'='*80}{RESET}\n")
        
        if reason == "PROFIT_TARGET":
            self.play_sound("PROFIT")
        
        # Remove from active positions
        del self.positions[symbol]
        
        # Update drawdown
        self.update_drawdown()
    
    def close_all_positions(self, reason="MANUAL"):
        """Close all open positions"""
        print(f"{YELLOW}Closing all positions: {reason}{RESET}")
        
        for symbol in list(self.positions.keys()):
            if symbol in self.option_chain:
                current_price = self.option_chain[symbol]['ltp']
                position = self.positions[symbol]
                
                pnl_pct = ((current_price - position['entry_price']) / position['entry_price']) * 100
                pnl_value = (pnl_pct / 100) * position['trade_value']
                
                self.close_position(symbol, current_price, pnl_pct, pnl_value, reason)
    
    # ============================================
    # RISK MANAGEMENT
    # ============================================
    
    def check_circuit_breaker(self):
        """Check if circuit breaker should be triggered"""
        # Daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            self.circuit_breaker_active = True
            print(f"\n{RED}{'='*80}{RESET}")
            print(f"{RED}🚨 CIRCUIT BREAKER - DAILY LOSS LIMIT{RESET}")
            print(f"{RED}Daily P&L: ₹{self.daily_pnl:.2f} / Limit: ₹{-self.max_daily_loss:.2f}{RESET}")
            print(f"{RED}{'='*80}{RESET}\n")
            self.close_all_positions("CIRCUIT_BREAKER")
            return True
        
        # Daily profit target
        if self.daily_pnl >= self.max_daily_profit:
            print(f"\n{GREEN}{'='*80}{RESET}")
            print(f"{GREEN}🎯 DAILY PROFIT TARGET ACHIEVED!{RESET}")
            print(f"{GREEN}Daily P&L: ₹{self.daily_pnl:.2f}{RESET}")
            print(f"{GREEN}{'='*80}{RESET}\n")
            self.close_all_positions("DAILY_TARGET")
            self.trading_enabled = False
            return True
        
        # Consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.circuit_breaker_active = True
            print(f"\n{RED}{'='*80}{RESET}")
            print(f"{RED}🚨 CIRCUIT BREAKER - CONSECUTIVE LOSSES{RESET}")
            print(f"{RED}Losses in a row: {self.consecutive_losses}{RESET}")
            print(f"{RED}{'='*80}{RESET}\n")
            return True
        
        # Daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            print(f"{YELLOW}⚠️  Daily trade limit reached: {self.daily_trades}/{self.max_daily_trades}{RESET}")
            return True
        
        return False
    
    def check_target_return(self):
        """Check if overall target return achieved"""
        current_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        if current_return >= self.target_return and not self.target_reached:
            self.target_reached = True
            self.trading_enabled = False
            
            print(f"\n{MAGENTA}{'='*80}{RESET}")
            print(f"{MAGENTA}🎉🎉🎉 TARGET RETURN ACHIEVED! 🎉🎉🎉{RESET}")
            print(f"{MAGENTA}{'='*80}{RESET}")
            print(f"{MAGENTA}Initial Capital: ₹{self.initial_balance:,.2f}{RESET}")
            print(f"{MAGENTA}Current Balance: ₹{self.balance:,.2f}{RESET}")
            print(f"{MAGENTA}Profit: ₹{self.balance - self.initial_balance:,.2f}{RESET}")
            print(f"{MAGENTA}Return: {current_return:.2f}% (Target: {self.target_return}%){RESET}")
            print(f"{MAGENTA}{'='*80}{RESET}\n")
            
            self.play_sound("TARGET_ACHIEVED")
            self.close_all_positions("TARGET_ACHIEVED")
            
            return True
        
        return False
    
    def update_drawdown(self):
        """Update maximum drawdown"""
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        if self.peak_balance > 0:
            dd = (self.peak_balance - self.balance) / self.peak_balance
            if dd > self.max_drawdown:
                self.max_drawdown = dd
    
    # ============================================
    # DASHBOARD
    # ============================================
    
    def print_dashboard(self, nifty_price, signal, indicators):
        """Print comprehensive dashboard"""
        closed_trades = [t for t in self.trades if t.get('status') == 'CLOSED']
        total_pnl = sum([t.get('pnl', 0) for t in closed_trades])
        wins = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losses = [t for t in closed_trades if t.get('pnl', 0) < 0]
        win_rate = len(wins) / len(closed_trades) if closed_trades else 0
        current_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        print(f"\n{CYAN}{'='*90}{RESET}")
        print(f"{CYAN}📊 SCALPING DASHBOARD - {datetime.now(self.ist_tz).strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
        print(f"{CYAN}{'='*90}{RESET}")
        
        # Status
        if self.target_reached:
            print(f"{MAGENTA}🏆 TARGET ACHIEVED - {current_return:.2f}% Return!{RESET}")
        elif self.circuit_breaker_active:
            print(f"{RED}🚨 CIRCUIT BREAKER ACTIVE{RESET}")
        else:
            progress = (self.balance / self.target_balance) * 100 if self.target_balance > 0 else 0
            needed = self.target_balance - self.balance
            print(f"{BLUE}🎯 Progress: {current_return:.2f}%/{self.target_return}% ({progress:.1f}% complete) | Need: ₹{needed:,.2f}{RESET}")
        
        # Market
        print(f"\n{WHITE}📈 MARKET:{RESET}")
        print(f"{BLUE}   Nifty: {nifty_price:.2f} | ATM Strike: {self.get_atm_strike(nifty_price)}{RESET}")
        
        if indicators:
            print(f"{BLUE}   EMA: {indicators['ema_fast']:.1f} / {indicators['ema_medium']:.1f} / {indicators['ema_slow']:.1f}{RESET}")
            print(f"{BLUE}   RSI: {indicators['rsi']:.1f} | Momentum: {indicators['momentum']:+.2f}%{RESET}")
        
        sig_color = GREEN if signal == "BUY_CALL" else RED if signal == "BUY_PUT" else YELLOW
        print(f"{sig_color}   Current Signal: {signal}{RESET}")
        
        # Positions
        print(f"\n{WHITE}💼 OPEN POSITIONS ({len(self.positions)}/{self.max_positions}):{RESET}")
        
        if self.positions:
            total_exposure = 0
            for symbol, pos in self.positions.items():
                if symbol in self.option_chain:
                    curr = self.option_chain[symbol]['ltp']
                    pnl_pct = ((curr - pos['entry_price']) / pos['entry_price']) * 100
                    pnl_val = (pnl_pct / 100) * pos['trade_value']
                    hold = (datetime.now(self.ist_tz) - pos['entry_time']).seconds
                    trail = "🔒" if pos['trailing_stop_active'] else ""
                    
                    color = GREEN if pnl_val > 0 else RED
                    total_exposure += pos['trade_value']
                    
                    print(f"{color}   [{pos['option_type']}] {symbol} {trail}{RESET}")
                    print(f"{color}      Entry: ₹{pos['entry_price']:.2f} → Current: ₹{curr:.2f}{RESET}")
                    print(f"{color}      P&L: ₹{pnl_val:,.0f} ({pnl_pct:+.2f}%) | Hold: {hold}s / {self.time_exit_seconds}s{RESET}")
            
            print(f"{BLUE}   Total Exposure: ₹{total_exposure:,.2f}{RESET}")
        else:
            print(f"{YELLOW}   No open positions{RESET}")
        
        # Recent trades
        print(f"\n{WHITE}📋 RECENT TRADES (Last 5):{RESET}")
        
        for trade in list(self.recent_trades)[-5:]:
            if 'exit' in trade:
                pnl = trade.get('pnl', 0)
                color = GREEN if pnl > 0 else RED
                print(f"{color}   {trade['time']} | {trade['signal']} {trade['strike']} {trade['type']}{RESET}")
                print(f"{color}      ₹{trade['entry']:.1f} → ₹{trade['exit']:.1f} | P&L: ₹{pnl:,.0f} | {trade['reason']}{RESET}")
            else:
                print(f"{BLUE}   {trade['time']} | {trade['signal']} {trade['strike']} {trade['type']} | OPENED @ ₹{trade['entry']:.1f}{RESET}")
        
        # Performance
        print(f"\n{WHITE}💰 PERFORMANCE:{RESET}")
        print(f"{BLUE}   Balance: ₹{self.balance:,.2f} | Total P&L: ₹{total_pnl:,.2f} ({current_return:+.2f}%){RESET}")
        print(f"{BLUE}   Daily P&L: ₹{self.daily_pnl:,.2f} | Trades Today: {self.daily_trades}/{self.max_daily_trades}{RESET}")
        print(f"{BLUE}   Streak: {self.consecutive_wins}W / {self.consecutive_losses}L | Max DD: {self.max_drawdown:.2%}{RESET}")
        
        if closed_trades:
            avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
            avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
            max_win = max([t['pnl'] for t in wins]) if wins else 0
            max_loss = min([t['pnl'] for t in losses]) if losses else 0
            
            print(f"\n{WHITE}📊 STATISTICS:{RESET}")
            print(f"{BLUE}   Total Trades: {len(closed_trades)} | Win Rate: {win_rate:.1%} ({len(wins)}W / {len(losses)}L){RESET}")
            print(f"{GREEN}   Avg Win: ₹{avg_win:.2f} | Max Win: ₹{max_win:.2f}{RESET}")
            print(f"{RED}   Avg Loss: ₹{avg_loss:.2f} | Max Loss: ₹{max_loss:.2f}{RESET}")
            
            if avg_loss != 0:
                profit_factor = abs(avg_win * len(wins) / (avg_loss * len(losses)))
                print(f"{BLUE}   Profit Factor: {profit_factor:.2f}{RESET}")
        
        print(f"{CYAN}{'='*90}{RESET}\n")
    
    def play_sound(self, alert_type):
        """Play sound alert (Windows only)"""
        try:
            if platform.system() == "Windows" and winsound:
                if alert_type == "TARGET_ACHIEVED":
                    for _ in range(5):
                        winsound.Beep(1500, 200)
                        time_module.sleep(0.1)
                elif alert_type == "PROFIT":
                    for _ in range(2):
                        winsound.Beep(1200, 300)
                        time_module.sleep(0.1)
                elif "BUY" in alert_type:
                    winsound.Beep(1000, 500)
        except:
            pass
    
    # ============================================
    # MAIN EXECUTION LOOP
    # ============================================
    
    def run(self):
        """Main execution loop"""
        print(f"{GREEN}🚀 Starting Nifty Scalper...{RESET}\n")
        
        # Initial data load
        if not self.fetch_nifty_data():
            print(f"{RED}❌ Failed to load initial data{RESET}")
            return
        
        try:
            while self.is_running:
                self.iteration_count += 1
                
                # Daily reset
                self.reset_daily_metrics()
                
                # Check target return
                if self.check_target_return():
                    if len(self.positions) == 0:
                        break
                
                # Check circuit breaker
                if self.check_circuit_breaker():
                    time_module.sleep(60)
                    continue
                
                # Check market hours
                if not self.is_market_open():
                    self.debug_log("Market closed", YELLOW)
                    time_module.sleep(60)
                    continue
                
                # Get current Nifty price
                nifty_price = self.get_current_nifty_price()
                
                # Refresh data periodically
                if self.iteration_count % 5 == 0:
                    self.fetch_nifty_data()
                
                # Fetch option chain
                if not self.fetch_option_chain(nifty_price):
                    self.debug_log("Failed to fetch option chain", RED)
                    time_module.sleep(10)
                    continue
                
                # Calculate indicators
                indicators = self.calculate_indicators()
                
                if not indicators:
                    self.debug_log("Indicators not ready", YELLOW)
                    time_module.sleep(10)
                    continue
                
                # Check exits for open positions
                self.check_exits()
                
                # Generate signal
                signal = self.generate_signal(indicators)
                
                # Execute trades
                if signal in ["BUY_CALL", "BUY_PUT"]:
                    if self.trading_enabled and not self.circuit_breaker_active:
                        current_time = time_module.time()
                        last_time = self.last_signal_time.get(signal, 0)
                        
                        if current_time - last_time > self.cooldown_seconds:
                            print(f"{MAGENTA}🎯 Attempting to execute {signal} trade...{RESET}")
                            if self.execute_trade(signal, nifty_price):
                                self.last_signal_time[signal] = current_time
                        else:
                            remaining = self.cooldown_seconds - (current_time - last_time)
                            self.debug_log(f"Cooldown: {remaining:.0f}s remaining for {signal}", YELLOW)
                
                # Print dashboard
                self.print_dashboard(nifty_price, signal, indicators)
                
                # Exit if target reached
                if self.target_reached and len(self.positions) == 0:
                    print(f"{GREEN}✅ Target achieved and all positions closed. Exiting...{RESET}")
                    break
                
                # Sleep
                time_module.sleep(20)
        
        except KeyboardInterrupt:
            print(f"\n{YELLOW}⚠️  Keyboard interrupt. Shutting down...{RESET}")
        except Exception as e:
            print(f"\n{RED}❌ Error in main loop: {e}{RESET}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup and final report"""
        print(f"\n{BLUE}🔄 Cleaning up...{RESET}")
        
        # Close remaining positions
        if self.positions:
            print(f"{YELLOW}Closing {len(self.positions)} open positions...{RESET}")
            self.close_all_positions("CLEANUP")
        
        # Final statistics
        closed_trades = [t for t in self.trades if t.get('status') == 'CLOSED']
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        wins = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losses = [t for t in closed_trades if t.get('pnl', 0) < 0]
        
        print(f"\n{MAGENTA}{'='*90}{RESET}")
        print(f"{MAGENTA}🏁 FINAL RESULTS{RESET}")
        print(f"{MAGENTA}{'='*90}{RESET}")
        
        print(f"\n{CYAN}💰 CAPITAL SUMMARY:{RESET}")
        print(f"{BLUE}   Initial Balance: ₹{self.initial_balance:,.2f}{RESET}")
        print(f"{BLUE}   Final Balance: ₹{self.balance:,.2f}{RESET}")
        print(f"{GREEN if total_return > 0 else RED}   Total Return: {total_return:+.2f}%{RESET}")
        print(f"{BLUE}   Target Return: {self.target_return}%{RESET}")
        
        if self.target_reached:
            print(f"{GREEN}   Status: ✅ TARGET ACHIEVED!{RESET}")
        else:
            print(f"{YELLOW}   Status: ❌ Target Not Achieved{RESET}")
        
        if closed_trades:
            win_rate = len(wins) / len(closed_trades)
            total_pnl = sum([t.get('pnl', 0) for t in closed_trades])
            
            print(f"\n{CYAN}📊 TRADING STATISTICS:{RESET}")
            print(f"{BLUE}   Total Trades: {len(closed_trades)}{RESET}")
            print(f"{BLUE}   Wins: {len(wins)} | Losses: {len(losses)}{RESET}")
            print(f"{BLUE}   Win Rate: {win_rate:.1%}{RESET}")
            print(f"{BLUE}   Total P&L: ₹{total_pnl:,.2f}{RESET}")
            
            if wins:
                avg_win = np.mean([t['pnl'] for t in wins])
                max_win = max([t['pnl'] for t in wins])
                print(f"{GREEN}   Avg Win: ₹{avg_win:.2f} | Max Win: ₹{max_win:.2f}{RESET}")
            
            if losses:
                avg_loss = np.mean([t['pnl'] for t in losses])
                max_loss = min([t['pnl'] for t in losses])
                print(f"{RED}   Avg Loss: ₹{avg_loss:.2f} | Max Loss: ₹{max_loss:.2f}{RESET}")
            
            print(f"{BLUE}   Max Drawdown: {self.max_drawdown:.2%}{RESET}")
            
            if wins and losses:
                profit_factor = abs(sum([t['pnl'] for t in wins]) / sum([t['pnl'] for t in losses]))
                print(f"{BLUE}   Profit Factor: {profit_factor:.2f}{RESET}")
        
        print(f"\n{MAGENTA}{'='*90}{RESET}\n")
        print(f"{GREEN}✅ Strategy execution completed.{RESET}\n")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    try:
        print(f"{CYAN}Initializing Improved Nifty Scalper...{RESET}\n")
        
        # Create and run strategy
        strategy = ImprovedNiftyScalper()
        strategy.run()
        
    except Exception as e:
        print(f"{RED}❌ Fatal error: {e}{RESET}")
        import traceback
        traceback.print_exc()
