"""
Tunel ngrok para o app Via A2.

Uso:
    python ngrok_tunnel.py

Configure NGROK_TOKEN via variavel de ambiente (preferido):
    export NGROK_AUTHTOKEN=seu_token   (bash/zsh)
    $env:NGROK_AUTHTOKEN="seu_token"   (PowerShell)

Ou passe --token na linha de comando.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request

from pyngrok import ngrok
from pyngrok.exception import PyngrokNgrokError

DEFAULT_PORT = 8095
CHECK_INTERVAL = 15  # segundos


def internet_ok() -> bool:
    try:
        urllib.request.urlopen("https://httpbin.org/status/200", timeout=5)
        return True
    except Exception:
        return False


def start_tunnel(port: int) -> ngrok.NgrokTunnel:
    try:
        ngrok.kill()
    except Exception:
        pass
    tunnel = ngrok.connect(port, "http")
    print(f"\n╔════════════════════════════════════════════════════════════╗")
    print(f"║  TUNEL ATIVO                                                 ║")
    print(f"║  URL publica: {tunnel.public_url:<44} ║")
    print(f"╚════════════════════════════════════════════════════════════╝\n", flush=True)
    return tunnel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--token", type=str, default=os.environ.get("NGROK_AUTHTOKEN", ""))
    args = parser.parse_args()

    if args.token:
        ngrok.set_auth_token(args.token)
    else:
        print("[aviso] NGROK_AUTHTOKEN nao definido. Tentando usar config global.", file=sys.stderr)

    print(f"=== ngrok auto-reconnect (porta {args.port}) ===")
    tunnel = None
    while True:
        try:
            if tunnel is None:
                if internet_ok():
                    tunnel = start_tunnel(args.port)
                else:
                    print(".", end="", flush=True)
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            print("\n[info] encerrando tunel...")
            ngrok.kill()
            break
        except PyngrokNgrokError as e:
            print(f"\n[!] erro ngrok: {e}", flush=True)
            tunnel = None


if __name__ == "__main__":
    main()
