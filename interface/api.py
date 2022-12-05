import importlib
import token
from loguru import logger
from sanic import Sanic, Request
from sanic_cors import CORS

app = Sanic(__name__)
CORS(app, supports_credentials=True)
mdb = importlib.import_module(f"clients.db_mongo").Client(host='mongodb://localhost', port=27017, database='vms')


@app.route('/actions', methods=['POST'])
async def actions(request: Request, response, user: dict=app.ext.add_dependency(token.login_required)):
    try:
        data = request.json

    except Exception as exception:
        logger.exception(" tag: {tag}, code: {code}", tag=tag, code=code)
        logger.exception(exception)
        return dict(code=-1, data=[], message=f"something is wrong. [{exception.__class__.__name__}]")


if __name__ == '__main__':
    pass
