import ccxt
import pandas as pd

# Cambiamos a MEXC ya que no bloquea las IPs de los servidores en la nube de GitHub
exchange = ccxt.mexc({
    'enableRateLimit': True,
})

def ejecutar_estrategia():
    simbolo = 'BTC/USDT'
    timeframe = '1h'
    
    # Descargar datos históricos públicos
    velas = exchange.fetch_ohlcv(simbolo, timeframe, limit=100)
    df = pd.DataFrame(velas, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Calcular medias móviles con Pandas
    df['ema_rapida'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_lenta'] = df['close'].ewm(span=21, adjust=False).mean()
    
    ultima_vela = df.iloc[-2]
    print(f"[{ultima_vela['timestamp']}] Análisis ejecutado con éxito en la nube (MEXC). Cierre actual: {ultima_vela['close']}")

if __name__ == "__main__":
    ejecutar_estrategia()
    
