from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from detect import main
app = Flask(__name__)
videos_path = os.path.join('static', 'videos')
app.config['UPLOAD_FOLDER'] = videos_path
output_format = '.mp4'


@app.route('/')
def upload_file():
    return render_template('test.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploadfile():
    if request.method == 'POST':
        file = request.files['file']
        file_name = secure_filename(file.filename)
        file.save(os.path.join(videos_path, file_name))

        dataset = request.form['dataset']

        main(file_name, dataset)
    output_file_path = os.path.join(
        videos_path, 'output_{}'.format(file_name))
    return render_template('postUpload.html', data=output_file_path)


if __name__ == '__main__':
    app.run(debug=True)
