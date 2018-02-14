from flask import Flask


app = Flask('cryptonalysis', instance_relative_config=True)
app.config.from_object('config')
app.config.from_pyfile('config.py')

import cryptonalysis.views
