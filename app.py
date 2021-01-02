from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from detect import main
from analyze import analyze_statistics
app = Flask(__name__)
videos_path = os.path.join('static', 'videos')
images_path = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = videos_path
output_format = '.mp4'
closest_dists_path = ""
min_closest_dists_path = ""
stats_vs_time_path = ""
two_d_hist_density_vs_avg_dists_path = ""
two_d_hist_density_vs_min_dists_path = ""
two_d_hist_density_vs_violation_path = ""
regression_density_vs_violations_path = ""


@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploadfile():
    if request.method == 'POST':

        os.makedirs(videos_path, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)

        file = request.files['file']
        dataset = request.form['dataset']
        model = request.form['model']

        file_name = secure_filename(file.filename)
        file_base_name = file_name.split('.')[0]

        videos_dir = os.path.join(videos_path, file_base_name)
        images_dir = os.path.join(images_path, file_base_name)
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        file.save(os.path.join(videos_dir, file_name))

        main(file_name, dataset, model)

        (closest_dists_path,
            min_closest_dists_path,
         stats_vs_time_path,
         two_d_hist_density_vs_avg_dists_path,
         two_d_hist_density_vs_min_dists_path,
         two_d_hist_density_vs_violation_path,
         regression_density_vs_violations_path) = analyze_statistics(dataset, file_name)

    output_file_name = 'output_%s%s' % (file_base_name, output_format)

    output_file_path = os.path.join(videos_dir, output_file_name)

    return render_template('results.html',
                           data=output_file_path,
                           closest_dists_path=closest_dists_path,
                           min_closest_dists_path=min_closest_dists_path,
                           stats_vs_time_path=stats_vs_time_path,
                           two_d_hist_density_vs_avg_dists_path=two_d_hist_density_vs_avg_dists_path,
                           two_d_hist_density_vs_min_dists_path=two_d_hist_density_vs_min_dists_path,
                           two_d_hist_density_vs_violation_path=two_d_hist_density_vs_violation_path,
                           regression_density_vs_violations_path=regression_density_vs_violations_path)


if __name__ == '__main__':
    app.run(debug=True)
