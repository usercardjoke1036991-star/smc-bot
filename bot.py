import os
import ccxt
import pandas as pd

# Cargar claves de forma segura desde los secretos de GitHub
api_key = os.environ.get('BINANCE_API_KEY')
secret_key = os.environ.get('BINANCE_SECRET')

# Inicializar conexión con Binance Futures
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
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Calcular medias móviles nativas con Pandas (sin errores de instalación)
    df['ema_rapida'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_lenta'] = df['close'].ewm(span=21, adjust=False).mean()
    
    ultima_vela = df.iloc[-2]
    print(f"[{ultima_vela['timestamp']}] Análisis ejecutado con éxito en la nube. Cierre actual: {ultima_vela['close']}")

if __name__ == "__main__":
    ejecutar_estrategia()
    
