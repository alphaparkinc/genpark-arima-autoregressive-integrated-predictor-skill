import sys
import json
from client import ARIMAPredictor

def handle_request(req):
    method = req.get("method")
    req_id = req.get("id")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "genpark-arima-autoregressive-integrated-predictor-skill", "version": "1.0.0"}
            }
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "forecast_arima",
                        "description": "Fit ARIMA(1, d, 0) model via differencing and autoregression to forecast future values",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "series": {"type": "array", "items": {"type": "number"}},
                                "d": {"type": "integer", "default": 1},
                                "horizon": {"type": "integer", "default": 2}
                            },
                            "required": ["series"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})
        
        if tool_name == "forecast_arima":
            s = args.get("series", [])
            d = args.get("d", 1)
            h = args.get("horizon", 2)
            res = ARIMAPredictor.fit_predict(s, d=d, horizon=h)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(res)}]
                }
            }
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
