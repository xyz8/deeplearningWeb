from flask import Flask
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from config import config


def create_app():
    app = Flask(__name__)
    app.config.from_object(config['default'])
    bootstrap = Bootstrap(app)
    moment = Moment(app)

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app




