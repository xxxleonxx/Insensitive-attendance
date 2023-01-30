import json
from loguru import logger
from sanic import Sanic
from line_manage import LineManage

app = Sanic()


def is_json(string):
    """是否json"""
    try:
        if string:
            json.loads(string)
        else:
            return False
    except TypeError:
        return False
    return True


@logger.catch
def generate_app():
    async def coroutine_method(module, method, _params):
        result = await getattr(module, method)(**_params)
        return result

    def foreground_method(module, method, _params):
        """等待返回"""
        result = getattr(module, method)(**_params)
        return result

    async def background_method(module, method, _params):
        """不等返回"""
        result = getattr(module, method)(**_params)
        return result

    @app.websocket('/line')
    async def line(ws):
        try:
            await ws.accept()
            while True:

                data = await ws.receive_text()  # 消息接受方法 receive_{text/json/bytes}
                if not data:
                    logger.error("error code is 1!")
                    break
                if not is_json(data):
                    logger.error("error code is 2!")
                    break

                data = json.loads(data)
                # methods.debug_log('application.line', f"data: {data}")
                if not data.get('line_id'):
                    logger.error("error code is 3!")
                    break

                # --- save ---
                line_id = data.get('line_id')
                LineManage.line_dict[line_id] = ws

        except Exception as exception:

            if exception.__class__.__name__ == 'WebSocketDisconnect':
                await ws.close()
            else:
                logger.exception(exception)

def app_run():
    LineManage.run_background()
    app.run(host='127.0.0.1', port=10080)


