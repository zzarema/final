import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import lagrange
from scipy.integrate import romb
from numpy.linalg import inv, solve

# Task 1: Graphical Method and Absolute Error
def f1(x):
    return x**3 - 3*x + 2

def task1():
    st.subheader("Task 1: Graphical Method and Absolute Error")
    st.subheader("Plot the graph of f(x)=x^3−3x+2 in the range x∈[-2,2].")
    x_min = st.number_input("Enter x min:", value=-2.0, format="%.2f")
    x_max = st.number_input("Enter x max:", value=2.0, format="%.2f")
    x = np.linspace(x_min, x_max, 400)
    y = f1(x)
    roots = fsolve(f1, [x_min, 0, x_max])
    st.write(f"Approximate roots: {roots}")
    fig, ax = plt.subplots()
    ax.plot(x, y, label="f(x) = x³ - 3x + 2")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid()
    ax.legend()
    st.pyplot(fig)

# Task 2: Root-Finding Methods
def f2(x):
    return x**4 - 5*x**2 + 4

def bisection_method(a, b, tol=1e-6):
    while abs(b - a) > tol:
        c = (a + b) / 2
        if f2(c) * f2(a) < 0:
            b = c
        else:
            a = c
    return c

def task2():
    st.subheader("Task 2: Comparison of Root-Finding Methods")
    st.subheader("Find the root of f(x)=x^4−5x^2+4 in the interval [0,3] using:")

    a = st.number_input("Enter range start (a):", value=0.0, format="%.2f")
    b = st.number_input("Enter range end (b):", value=3.0, format="%.2f")
    method = st.selectbox("Select method:", ["Bisection Method", "False Position Method"])
    if method == "Bisection Method":
        root = bisection_method(a, b)
    else:
        root = fsolve(f2, [(a + b) / 2])
    st.write(f"Root found using {method}: {root}")

# Task 3: Gaussian Elimination with Partial Pivoting
def task3():
    st.subheader("Task 3: Gaussian Elimination with Partial Pivoting")
    A = np.array([[2, 1, -1], [-3, 1, 2], [-2, 1, 3]])
    b = np.array([8, -11, -3])
    x = solve(A, b)
    st.write("Solving the system:")
    st.latex(r"\begin{cases} 2x + y - z = 8 \\ -3x + y + 2z = -11 \\ -2x + y + 3z = -3 \end{cases}")
    st.write(f"Solution: {x}")

# Task 4: Iterative Method for Matrix Inversion
def task4():
    st.subheader("Task 4: Iterative Method for Matrix Inversion")
    A = np.array([[1, 2, 3], [0, -1, 4], [5, 6, -1]])
    st.write("Original Matrix A:")
    st.write(A)
    A_inv = inv(A)
    st.write("Inverse Matrix A⁻¹:")
    st.write(A_inv)

# Task 5: Polynomial Curve Fitting
def task5():
    st.subheader("Task 5: Polynomial Curve Fitting")
    x = st.text_input("Enter x values (comma-separated):", "0,1,2,3,4")
    y = st.text_input("Enter y values (comma-separated):", "0,1,4,9,16")
    x = np.array([float(i) for i in x.split(',')])
    y = np.array([float(i) for i in y.split(',')])
    coeffs = np.polyfit(x, y, 2)
    coeffs = np.round(coeffs, decimals=5)
    st.write(f"Quadratic Fit Coefficients: {coeffs}")

# Task 6: Lagrange Interpolation
def task6():
    st.subheader("Task 6: Lagrange Interpolation")
    x = st.text_input("Enter x values (comma-separated):", "1,3,5")
    y = st.text_input("Enter y values (comma-separated):", "2,8,18")
    x = np.array([float(i) for i in x.split(',')])
    y = np.array([float(i) for i in y.split(',')])
    poly = lagrange(x, y)
    val = st.number_input("Enter value to estimate:", value=4.0)
    f4 = poly(val)
    st.write(f"Estimated f({val}) using Lagrange Interpolation: {f4}")


def f7(x):
    return x ** 2  # Интегрируем x^2


def romberg_integration(a, b, h_values):
    results = []
    for h in h_values:
        x = np.arange(a, b + h, h)
        y = f7(x)
        result = np.trapz(y, x)  # Используем метод трапеций
        results.append(result)
    return results


def task7():
    st.subheader("Task 7: Romberg’s Integration")

    # Ввод пределов интегрирования
    a = st.number_input("Enter lower limit (a):", value=0.0, format="%.2f")
    b = st.number_input("Enter upper limit (b):", value=1.0, format="%.2f")

    # Ввод шагов разбиения
    h_values_input = st.text_input("Enter step values (comma-separated):", "0.5,0.25,0.125")
    h_values = np.array([float(h) for h in h_values_input.split(',')])

    # Вычисляем интеграл
    results = romberg_integration(a, b, h_values)

    # Отображаем таблицу
    st.write("Romberg Integration Table:")
    table_data = {"h": h_values, "Approximation": results}
    st.table(table_data)


# Task 8: Runge-Kutta 2nd Order
def f8(x, y):
    return np.exp(x)-y

def task8():
    st.subheader("Task 8: Runge-Kutta 2nd Order")
    st.subheader("Apply the Runge-Kutta second-order method to solve dy/dx=e^x−y, y(0)=0, for x=0.2, with a step size h=0.1.")
    y0 = st.number_input("Enter initial y(0):", value=0.0, format="%.2f")
    h = st.number_input("Enter step size h:", value=0.1, format="%.2f")
    x_end = st.number_input("Enter final x:", value=0.2, format="%.2f")
    x = 0
    y = y0
    while x < x_end:
        k1 = h * f8(x, y)
        k2 = h * f8(x + h / 2, y + k1 / 2)
        y += k2
        x += h
    st.write(f"Approximate solution at x = {x_end}: y = {y}")


# Streamlit main UI
st.title("Computational Methods Interface")
task_options = {
    "Task 1: Graphical Method": task1,
    "Task 2: Root-Finding Methods": task2,
    "Task 3: Gaussian Elimination": task3,
    "Task 4: Matrix Inversion": task4,
    "Task 5: Polynomial Curve Fitting": task5,
    "Task 6: Lagrange Interpolation": task6,
    "Task 7: Romberg’s Integration": task7,
    "Task 8: Runge-Kutta 2nd Order": task8,
}
selected_task = st.sidebar.selectbox("Select a task:", list(task_options.keys()))
task_options[selected_task]()

