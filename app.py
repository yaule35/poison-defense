import os
import torch
import __main__

from main import poison_defense
from flask import Flask, flash, request, redirect, send_from_directory, url_for, after_this_request
from werkzeug.utils import secure_filename

from pSp.pgd_attack import NewNet

setattr(__main__, "NewNet", NewNet)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd() + '/'
app.add_url_rule(
    "/deepfake/04/<name>", endpoint="download_file", build_only=True
)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/deepfake/04', methods=['POST'])
def poison():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(filename)
            print(f'File saving path:{filename}')
            result = poison_defense(filename)
            @after_this_request
            def remove_file(response):
                os.remove(filename)
                return response
            return redirect(url_for("download_file", name=result))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
@app.route('/deepfake/04/<name>')
def download_file(name):
    @after_this_request
    def remove_file(response):
        os.remove(name)
        return response
    return send_from_directory(app.config['UPLOAD_FOLDER'], name)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force='True')
    app.run(host='0.0.0.0', port=5000, debug=True)