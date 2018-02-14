from cryptonalysis import app


@app.route('/')
def index():
    return 'Welcome to Cryptonalysis!'
