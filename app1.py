from flask import Flask, render_template, request, redirect, url_for, g
import sqlite3

app = Flask(__name__)

DATABASE = 'products.db'

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

def create_tables():
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS skin_products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            product_name TEXT,
            product_cost REAL,
            product_picture TEXT
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS body_products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            product_name TEXT,
            product_cost REAL,
            product_picture TEXT
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT
        );
    ''')

    connection.commit()
    connection.close()

def add_to_cart(table, product_id, product_name, product_cost, product_picture):
    db = get_db()
    db.execute(f"INSERT INTO {table} (product_id, product_name, product_cost, product_picture) VALUES (?, ?, ?, ?)", 
               (product_id, product_name, product_cost, product_picture))
    db.commit()

def get_cart_items():
    db = get_db()
    skin_cart = db.execute("SELECT * FROM skin_products").fetchall()
    body_cart = db.execute("SELECT * FROM body_products").fetchall()
    return skin_cart, body_cart

@app.route('/skin_products')
def skin_products():
    return render_template('skin_products.html')

@app.route('/body_products')
def body_products():
    return render_template('body_products.html')

@app.route('/main_home')
def main_home():
    return render_template('main_home.html')

@app.route('/add_to_cart', methods=['POST'])
def add_to_cart_route():
    product_id = request.form.get('product_id')
    product_name = request.form.get('product_name')
    product_cost = request.form.get('product_cost')
    product_picture = request.form.get('product_picture')
    product_type = request.form.get('product_type')

    if product_type == 'skin':
        add_to_cart('skin_products', product_id, product_name, product_cost, product_picture)
    elif product_type == 'body':
        add_to_cart('body_products', product_id, product_name, product_cost, product_picture)
    
    return redirect(url_for('cart'))

@app.route('/cart')
def cart():
    skin_cart, body_cart = get_cart_items()
    return render_template('cart.html', skin_cart=skin_cart, body_cart=body_cart)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback_message = request.form.get('feedback_message')
    db = get_db()
    db.execute("INSERT INTO feedback (message) VALUES (?)", (feedback_message,))
    db.commit()
    return redirect(url_for('skin_products'))

if __name__ == '__main__':
    create_tables()
    app.run(debug=True)
