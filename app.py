from configs import FLASK_PORT, FLASK_IP
from base_api import app

if __name__ == '__main__':
    app.run(
        host=FLASK_IP,
        port=FLASK_PORT,
        debug=True,
        threaded=True
    )
