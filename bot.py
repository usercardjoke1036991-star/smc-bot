import os
import ccxt
import pandas as pd
import pandas_ta as ta

# Cargar claves de forma segura desde los secretos de GitHub
api_key = os.environ.get('BINANCE_API_KEY')
secret_key = os.environ.get('BINANCE_SECRET')

# Inicializar conexión con Binance Futures (Testnet o Real según prefieras)
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret_key,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

def ejecutar_estrategia():
    simbolo = 'BTC/USDT'
    timeframe = '1h'
    
    # Descargar datos históricos
    velas = exchange.fetch_ohlcv(simbolo, timeframe, limit=100)
    df = pd.DataFrame(velas, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Calcular indicadores de prueba (ejemplo: EMA)
    df['ema_rapida'] = ta.ema(df['close'], length=9)
    df['ema_lenta'] = ta.ema(df['close'], length=21)
    
    ultima_vela = df.iloc[-2]
    print(f"[{ultima_vela['timestamp']}] Análisis ejecutado con éxito en la nube. Cierre actual: {ultima_vela['close']}")

if __name__ == "__main__":
    ejecutar_estrategia()

