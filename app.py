from flask import Flask, render_template, Response,request, redirect, url_for, flash, session,jsonify,g
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import cv2
import numpy as np
import mediapipe as mp
import base64  # Ensure you import base64
from utils import read_landmarks, add_mask, face_points
app = Flask(__name__)
app.secret_key = 'your_secret_key'
# Define face elements and colors
face_elements = [
    "LIP_LOWER",
    "LIP_UPPER",
    "EYEBROW_LEFT",
    "EYEBROW_RIGHT",
    "EYELINER_LEFT",
    "EYELINER_RIGHT",
    "EYESHADOW_LEFT",
    "EYESHADOW_RIGHT",
]

colors_map = {
    "LIP_UPPER": [0, 0, 255],
    "LIP_LOWER": [0, 0, 255],
    "EYELINER_LEFT": [139, 0, 0],
    "EYELINER_RIGHT": [139, 0, 0],
    "EYESHADOW_LEFT": [0, 100, 0],
    "EYESHADOW_RIGHT": [0, 100, 0],
    "EYEBROW_LEFT": [19, 69, 139],
    "EYEBROW_RIGHT": [19, 69, 139],
}

face_connections = [face_points[feature] for feature in face_elements]
colors = [colors_map[feature] for feature in face_elements]


def generate_video():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        mask = np.zeros_like(frame)
        landmarks = read_landmarks(frame)
        if landmarks:
            mask = add_mask(mask, landmarks, face_connections, colors)
            frame = cv2.addWeighted(frame, 1.0, mask, 0.2, 1.0)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/apply_makeup', methods=['POST'])
def apply_makeup():
    data = request.get_json()
    lip_color = data.get('lipColor')
    eyeliner_color = data.get('eyelinerColor')
    eyeshadow_color = data.get('eyeshadowColor')

    # Update the colors map with the selected colors
    colors_map["LIP_UPPER"] = hex_to_rgb(lip_color)
    colors_map["LIP_LOWER"] = hex_to_rgb(lip_color)
    colors_map["EYELINER_LEFT"] = hex_to_rgb(eyeliner_color)
    colors_map["EYELINER_RIGHT"] = hex_to_rgb(eyeliner_color)
    colors_map["EYESHADOW_LEFT"] = hex_to_rgb(eyeshadow_color)
    colors_map["EYESHADOW_RIGHT"] = hex_to_rgb(eyeshadow_color)

    # Rebuild color list for the next frame
    global colors
    colors = [colors_map[feature] for feature in face_elements]

    return 'Makeup applied', 200


def hex_to_rgb(hex_color):
    # Convert hex to RGB
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
DATABASE = 'users.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def create_connection():
    conn = sqlite3.connect(DATABASE)
    return conn

# Create tables for users, profiles, products, and feedback
def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    # Create users and profiles table
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT UNIQUE,
                        password TEXT
                    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS profiles (
                        user_id INTEGER PRIMARY KEY,
                        email TEXT,
                        name TEXT,
                        age INTEGER,
                        address TEXT,
                        phone TEXT,
                        gender TEXT,
                        marital_status TEXT,
                        skin_color TEXT,
                        skin_type TEXT,
                        skin_allergies TEXT,
                        feedback TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE)
                    ''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS skin_products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    product_name TEXT,
    product_cost REAL,
    product_picture TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS body_products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    product_name TEXT,
    product_cost REAL,
    product_picture TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS feedback (
                        user_id INTEGER,
                        message TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE)
                    ''')

    conn.commit()
    conn.close()
mp_face_mesh = mp.solutions.face_mesh

def add_to_cart(user_id, product_name, product_cost, product_picture, product_type):
    conn = create_connection()
    cursor = conn.cursor()

    try:
        if product_type == "skin":
            cursor.execute("INSERT INTO skin_products (user_id, product_name, product_cost, product_picture) VALUES (?, ?, ?, ?)",
                           (user_id, product_name, product_cost, product_picture))
        else:
            cursor.execute("INSERT INTO body_products (user_id, product_name, product_cost, product_picture) VALUES (?, ?, ?, ?)",
                           (user_id, product_name, product_cost, product_picture))
        conn.commit()
    except Exception as e:
        print(f"Error adding to cart: {e}")
    finally:
        cursor.close()
        conn.close()

def get_cart_items(user_id):
    db = get_db()  # Assuming get_db() is a function that returns the database connection
    try:
        # Fetch skin products for the user
        skin_cart = db.execute("SELECT * FROM skin_products WHERE user_id=?", (user_id,)).fetchall()
        # Fetch body products for the user
        body_cart = db.execute("SELECT * FROM body_products WHERE user_id=?", (user_id,)).fetchall()

        # Get column names
        skin_cart_columns = [column[0] for column in db.execute("SELECT * FROM skin_products").description]
        body_cart_columns = [column[0] for column in db.execute("SELECT * FROM body_products").description]

        # Convert rows to dictionaries
        skin_cart_items = [dict(zip(skin_cart_columns, item)) for item in skin_cart]
        body_cart_items = [dict(zip(body_cart_columns, item)) for item in body_cart]

        return skin_cart_items, body_cart_items
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return [], []  # Return empty lists if there's an error

@app.route('/skin_products')
def skin_products():
    return render_template('skin_products.html')

@app.route('/body_products')
def body_products():
    return render_template('body_products.html')


@app.route('/mainhome')
def main_home():
    return render_template('mainhome.html')

@app.route('/add_to_cart', methods=['POST'])
def add_to_cart_route():
    user_id = session['user_id']  # Assuming you're storing user_id in the session
    product_name = request.form.get('product_name')
    product_cost = request.form.get('product_cost')
    product_picture = request.form.get('product_picture')
    product_type = request.form.get('product_type')

    add_to_cart(user_id, product_name, product_cost, product_picture, product_type)

    return redirect(url_for('cart'))

@app.route('/cart')
def cart():
    user_id = session['user_id']  # Assuming you're storing user_id in the session
    skin_cart, body_cart = get_cart_items(user_id)

    return render_template('cart.html', skin_cart=skin_cart, body_cart=body_cart)

def calculate_total(skin_cart, body_cart):
    total = 0
    for item in skin_cart:
        total += item['product_cost']
    for item in body_cart:
        total += item['product_cost']
    return total

@app.route('/payment', methods=['GET', 'POST'])
def payment_page():
    user_id = session.get('user_id')  # Get user_id from session
    if user_id is None:
        flash("You need to log in to access the payment page.")
        return redirect(url_for('login'))  # Redirect to login page if not logged in

    skin_cart, body_cart = get_cart_items(user_id)
    total_cost = calculate_total(skin_cart, body_cart)

    if request.method == 'POST':
        return process_payment()

    return render_template('payment.html', skin_cart=skin_cart, body_cart=body_cart, total_cost=total_cost)

def process_payment():
    card_number = request.form.get('card_number')
    expiry_date = request.form.get('expiry_date')
    cvv = request.form.get('cvv')

    if not card_number or not expiry_date or not cvv:
        flash("Please fill in all payment fields.")
        return redirect(url_for('payment_page'))


    flash("Payment processed successfully!")
    return redirect(url_for('some_success_page'))  # Redirect to a success page


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback_message = request.form.get('feedback_message')
    if 'user_id' in session:
        user_id = session['user_id']
        db = get_db()
        db.execute("INSERT INTO feedback (user_id, message) VALUES (?, ?)", (user_id, feedback_message))
        db.commit()
    return redirect(url_for('skin_products'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/mainhome')
def mainhome():
    return render_template('mainhome.html')


@app.route('/virtual_tryon')
def virtual_tryon():
    return render_template('virtual_tryon.html')

@app.route('/makeup_tutorials')
def makeup_tutorials():
    return render_template('makeup_tutorials.html')

@app.route('/profile')
def profile():
    if 'user_id' in session:
        user_id = session['user_id']
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute('''SELECT email, name, age, address, phone, gender, marital_status, skin_color, skin_type,skin_allergies,feedback
                          FROM profiles WHERE user_id = ?''', (user_id,))
        user = cursor.fetchone()
        conn.close()

        if user is None:
            user = (None, None, None, None, None, None, None, None, None, None, None)

        return render_template('profile.html', user=user)
    else:
        flash('Please sign in to view your profile.', 'warning')
        return redirect(url_for('signin'))

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'user_id' in session:
        if request.method == 'POST':
            email = request.form['email']
            name = request.form['name']
            age = request.form['age']
            gender = request.form['gender']
            marital_status = request.form['marital_status']
            phone = request.form['phone']
            address = request.form['address']
            skin_color = request.form['skin_color']
            skin_type = request.form['skin_type']
            skin_allergies = ', '.join(request.form.getlist('skin_allergies'))
            feedback = request.form['feedback']

            conn = create_connection()
            cursor = conn.cursor()

            cursor.execute('SELECT user_id FROM profiles WHERE user_id = ?', (session['user_id'],))
            profile_exists = cursor.fetchone()

            if profile_exists:
                cursor.execute('''UPDATE profiles SET email = ?, name = ?, age = ?, address = ?, phone = ?, 
                                 gender = ?, marital_status = ?, skin_color = ?, skin_type = ?, 
                                 skin_allergies = ?, feedback = ? 
                                 WHERE user_id = ?''', 
                               (email, name, age, address, phone, gender, marital_status, skin_color, skin_type, skin_allergies, feedback, session['user_id']))
            else:
                cursor.execute('''INSERT INTO profiles (user_id, email, name, age, address, phone, gender, marital_status, 
                                                         skin_color, skin_type, skin_allergies, feedback) 
                                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                               (session['user_id'], email, name, age, address, phone, gender, marital_status, skin_color, skin_type, skin_allergies, feedback))
            
            conn.commit()
            conn.close()

            flash('Profile updated successfully.', 'success')
            return redirect(url_for('profile'))
        else:
            conn = create_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM profiles WHERE user_id = ?', (session['user_id'],))
            user = cursor.fetchone()
            conn.close()

            return render_template('settings.html', user=user)
    else:
        flash('Please sign in to access settings.', 'warning')
        return redirect(url_for('signin'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        flash('Password reset instructions have been sent to your email.', 'info')
        return redirect(url_for('signin'))
    return render_template('forgot_password.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, password FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            flash('Login successful!', 'success')
            return redirect(url_for('main_home'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('signin.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('login'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        conn = create_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, hashed_password))
            conn.commit()
            user_id = cursor.lastrowid
            session['user_id'] = user_id
            flash('Account created successfully!', 'success')
            return redirect(url_for('mainhome'))
        except sqlite3.IntegrityError:
            flash('Email is already registered. Please sign in.', 'danger')
            return redirect(url_for('login'))
        finally:
            conn.close()

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'info')
    return redirect(url_for('home'))

if __name__ == '__main__':
    create_table()  # Create tables if they don't exist
    app.run(debug=True)
