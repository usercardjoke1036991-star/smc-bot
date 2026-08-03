import os
import time
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
  print(
      f"[{ultima_vela['timestamp']}] Análisis ejecutado con éxito en la nube"
      f" (MEXC). Cierre actual: {ultima_vela['close']}"
  )

  # Construir el mensaje para Telegram
  mensaje = (
      f"🚨 *ACTUALIZACIÓN BOT SMC / EMA* 🚨\n\n🔹 *Activo:* {simbolo}\n🔹"
      f" *Temporalidad:* {timeframe}\n🔹 *Cierre Actual:*"
      f" `${ultima_vela['close']:,.2f}`\n🔹 *EMA 9:*"
      f" `${ultima_vela['ema_rapida']:,.2f}`\n🔹 *EMA 21:*"
      f" `${ultima_vela['ema_lenta']:,.2f}`\n\n🕒 *Timestamp:*"
      f" `{ultima_vela['timestamp']}`"
  )

  # Enviar la alerta
  enviar_alerta_telegram(mensaje)


if __name__ == '__main__':
  print('🤖 Bot SMC iniciado en modo continuo (Bucle automático)...')
  while True:
    try:
      ejecutar_estrategia()
    except Exception as e:
      print(f'❌ Error en el ciclo del bot: {e}')

    print(
        '⏳ Esperando 1 hora para el siguiente análisis... (Puedes dejar la'
        ' pestaña abierta)'
    )
    time.sleep(3600)  # Pausa de 3600 segundos (1 hora)
      
