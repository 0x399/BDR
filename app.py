import os

import pandas as pd
from flask import Flask, render_template, url_for, redirect, session, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
from flask import request, flash
from werkzeug.utils import secure_filename
from plot_generator import generate_temp_avg_plot, generate_temp_max_plot, generate_temp_min_plot, generate_wind_plot, \
    generate_humidity_plot, generate_pressure_plot, generate_precipitation_plot, \
    generate_linear_regression_plots_all, generate_prophet_forecast_plot

app = Flask(__name__)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'



UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

class File(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    filepath = db.Column(db.String(200), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    user = db.relationship('User', backref=db.backref('files', lazy=True))


class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')


@app.route('/')
def home():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)

@app.route('/dashboard')
@login_required
def dashboard():
    files = File.query.filter_by(user_id=current_user.id).all()
    plot_file_id = request.args.get('plot_file_id', type=int)

    img_avg = img_max = img_min = img_wind = img_humidity = img_pressure = img_precipitation = None
    img_avg_regression = img_max_regression = img_min_regression = img_precipitation_regression = None
    img_humidity_regression = img_wind_regression = img_pressure_regression = None
    selected_file = first_date_html = last_date_html = None
    first_date_str = last_date_str = None

    if plot_file_id:
        file_record = File.query.get_or_404(plot_file_id)
        if file_record.user_id != current_user.id:
            flash("Unauthorized access to file")
            return redirect(url_for('dashboard'))

        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')

        df_dates = pd.read_csv(file_record.filepath, delimiter=';', decimal=',', na_values=[''])
        df_dates['date'] = pd.to_datetime(df_dates['date'], format='%d.%m.%Y', errors='coerce')

        if start_date_str and end_date_str:
            start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
            end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
            df_dates = df_dates[(df_dates['date'] >= start_date) & (df_dates['date'] <= end_date)]

        first_date = df_dates['date'].min()
        last_date = df_dates['date'].max()
        first_date_html = first_date.strftime('%Y-%m-%d') if pd.notnull(first_date) else ''
        last_date_html = last_date.strftime('%Y-%m-%d') if pd.notnull(last_date) else ''
        first_date_str = first_date.strftime('%d.%m.%Y') if pd.notnull(first_date) else None
        last_date_str = last_date.strftime('%d.%m.%Y') if pd.notnull(last_date) else None

        df_dates['date'] = df_dates['date'].dt.strftime('%d.%m.%Y')

        temp_filtered_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_filtered_{file_record.id}.csv')
        df_dates.to_csv(temp_filtered_path, sep=';', decimal=',', index=False)

        img_avg = generate_temp_avg_plot(temp_filtered_path)
        img_max = generate_temp_max_plot(temp_filtered_path)
        img_min = generate_temp_min_plot(temp_filtered_path)
        img_wind = generate_wind_plot(temp_filtered_path)
        img_humidity = generate_humidity_plot(temp_filtered_path)
        img_pressure = generate_pressure_plot(temp_filtered_path)
        img_precipitation = generate_precipitation_plot(temp_filtered_path)

        regression_images = generate_linear_regression_plots_all(temp_filtered_path)
        img_avg_regression = regression_images['temp_avg']
        img_max_regression = regression_images['temp_max']
        img_min_regression = regression_images['temp_min']
        img_precipitation_regression = regression_images['precipitation']
        img_humidity_regression = regression_images['humidity']
        img_wind_regression = regression_images['wind']
        img_pressure_regression = regression_images['pressure']

        selected_file = file_record

        if os.path.exists(temp_filtered_path):
            os.remove(temp_filtered_path)

    return render_template('dashboard.html',
                           files=files,
                           img_avg=img_avg,
                           img_max=img_max,
                           img_min=img_min,
                           img_wind=img_wind,
                           img_humidity=img_humidity,
                           img_pressure=img_pressure,
                           img_precipitation=img_precipitation,
                           first_date=first_date_str,
                           last_date=last_date_str,
                           selected_file=selected_file,
                           first_date_html=first_date_html,
                           last_date_html=last_date_html,

                           img_avg_regression=img_avg_regression,
                           img_max_regression=img_max_regression,
                           img_min_regression=img_min_regression,
                           img_precipitation_regression=img_precipitation_regression,
                           img_humidity_regression=img_humidity_regression,
                           img_wind_regression=img_wind_regression,
                           img_pressure_regression=img_pressure_regression)



@app.route('/plot/<int:file_id>')
@login_required
def show_plot(file_id):
    file_record = File.query.get_or_404(file_id)

    if file_record.user_id != current_user.id:
        flash("Unauthorized access to file")
        return redirect(url_for('dashboard'))

    # Generate multiple plots
    avg_plot = generate_temp_avg_plot(file_record.filepath)
    max_plot = generate_temp_max_plot(file_record.filepath)
    min_plot = generate_temp_min_plot(file_record.filepath)

    return render_template('plot.html',
                           img_avg=avg_plot,
                           img_max=max_plot,
                           img_min=min_plot,
                           selected_file=file_record)

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
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
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Save file record in DB
            new_file = File(filename=filename, filepath=filepath, user_id=current_user.id)
            db.session.add(new_file)
            db.session.commit()

            flash('File uploaded successfully')
            return redirect(url_for('dashboard'))

    return render_template('upload.html')


@app.route('/forecast', methods=['GET', 'POST'])
@login_required
def forecast():
    files = File.query.filter_by(user_id=current_user.id).all()
    plot_data = None
    selected_file = None
    future_days = selected_column = None
    forecast_df = None

    if request.method == 'POST':
        file_id = int(request.form['file_id'])
        future_days = int(request.form['future_days'])
        selected_column = request.form.get('column_name', 'temp_avg')

        file_record = File.query.get_or_404(file_id)
        if file_record.user_id != current_user.id:
            flash("Unauthorized access to file")
            return redirect(url_for('forecast'))

        plot_data, forecast_df = generate_prophet_forecast_plot(file_record.filepath, future_days, selected_column)
        selected_file = file_record

        # Store forecast data temporarily in session
        session['forecast_data'] = forecast_df.to_json(orient='split')
        session['forecast_filename'] = f"{file_record.filename}_forecast_{selected_column}.csv"

    return render_template('forecast.html',
                           files=files,
                           plot_data=plot_data,
                           selected_file=selected_file,
                           selected_column=selected_column,
                           future_days=future_days)


@app.route('/download_forecast')
@login_required
def download_forecast():
    forecast_json = session.get('forecast_data')
    filename = session.get('forecast_filename', 'forecast.csv')
    if not forecast_json:
        flash('No forecast data available to download.')
        return redirect(url_for('forecast'))

    forecast_df = pd.read_json(forecast_json, orient='split')
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds']).dt.strftime('%Y-%m-%d')

    forecast_df = forecast_df.rename(columns={
        'ds': 'date',
        'yhat': 'forecast',
        'yhat_lower': 'lower_bound',
        'yhat_upper': 'upper_bound'
    })

    # Create in-memory buffer
    from io import StringIO
    buffer = StringIO()
    forecast_df.to_csv(buffer, index=False)
    buffer.seek(0)

    return Response(buffer,
                    mimetype='text/csv',
                    headers={"Content-Disposition": f"attachment;filename={filename}"})


@app.route('/myfiles')
@login_required
def myfiles():
    files = File.query.filter_by(user_id=current_user.id).all()
    return render_template('myfiles.html', files=files)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
