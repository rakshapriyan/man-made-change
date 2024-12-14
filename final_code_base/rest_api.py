from flask import Flask, request

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/uploadfiles",methods=["GET","POST"])
def uploadfiles():
    print(request.files.keys())
    return "fucc you"

@app.route("/coords", methods=["GET","POST"])
def get_coords():
    test = request.get_json()
    print(test)
    return test


app.run("0.0.0.0", debug=True)
