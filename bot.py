import os
import ccxt
import pandas as pd
import requests

# Configuración de Telegram utilizando variables de entorno del sistema
TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# Cambiamos a MEXC ya que no bloquea las IPs de los servidores en la nube de GitHub
exchange = ccxt.mexc({
    'enableRateLimit': True,
})


def enviar_alerta_telegram(mensaje):
  """Envía la notificación push directamente a tu Telegram."""
  if not TOKEN or not CHAT_ID:
    print(
        '⚠️ Credenciales de Telegram no encontradas en las variables de'
        ' entorno.'
    )
    return

  url = f'https://api.telegram.org/bot{TOKEN}/sendMessage'
  payload = {"chat_id": CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
  try:
    response = requests.post(url, json=payload)
    if response.status_code == 200:
      print('📲 ¡Alerta enviada a Telegram con éxito!')
    else:
      print(f'❌ Error al enviar Telegram: {response.text}')
  except Exception as e:
    print(f'❌ Excepción al conectar con Telegram: {e}')


def ejecutar_estrategia():
  simbolo = 'BTC/USDT'
  timeframe = '1h'

  # Descargar datos históricos públicos
  velas = exchange.fetch_ohlcv(simbolo, timeframe, limit=100)
  df = pd.DataFrame(
      velas, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
  )
  df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

  # Calcular medias móviles con Pandas
  df['ema_rapida'] = df['close'].ewm(span=9, adjust=False).mean()
  df['ema_lenta'] = df['close'].ewm(span=21, adjust=False).mean()

  ultima_vela = df.iloc[-2]
  precio_actual = ultima_vela['close']

  # --- SISTEMA PROFESIONAL DE GESTIÓN DE RIESGO (TP / SL) ---
  if ultima_vela['ema_rapida'] > ultima_vela['ema_lenta']:
    tipo_operacion = '🟢 COMPRA (LONG)'
    # Stop Loss basado en el mínimo estructural de las últimas 10 velas
    stop_loss = df['low'].iloc[-10:].min()
    riesgo = precio_actual - stop_loss
    # Take Profit con relación de beneficio 1:2
    take_profit = precio_actual + (riesgo * 2)
  else:
    tipo_operacion = '🔴 VENTA (SHORT)'
    # Stop Loss basado en el máximo estructural de las últimas 10 velas
    stop_loss = df['high'].iloc[-10:].max()
    riesgo = stop_loss - precio_actual
    # Take Profit con relación de beneficio 1:2
    take_profit = precio_actual - (riesgo * 2)

  print(
      f"[{ultima_vela['timestamp']}] Análisis ejecutado. Operación:"
      f" {tipo_operacion} | Cierre actual: {precio_actual}"
  )

  # Construir el mensaje profesional para Telegram
  mensaje = (
      f"🚨 *SEÑAL PROFESIONAL SMC / EMA* 🚨\n\n"
      f"🔹 *Activo:* {simbolo}\n"
      f"🔹 *Temporalidad:* {timeframe}\n"
      f"📊 *Dirección:* {tipo_operacion}\n\n"
      f"📍 *Precio de Entrada:* `${precio_actual:,.2f}`\n"
      f"🛑 *Stop Loss (SL):* `${stop_loss:,.2f}`\n"
      f"🎯 *Take Profit (TP 1:2):* `${take_profit:,.2f}`\n\n"
      f"📈 *EMA 9:* `${ultima_vela['ema_rapida']:,.2f}`\n"
      f"📉 *EMA 21:* `${ultima_vela['ema_lenta']:,.2f}`\n\n"
      f"🕒 *Timestamp:* `{ultima_vela['timestamp']}`"
  )

  # Enviar la alerta
  enviar_alerta_telegram(mensaje)


if __name__ == '__main__':
  ejecutar_estrategia()
    
