# Capstone Project FAQs

Standalone markdown version of the Capstone Project FAQs (Black-Box Optimization).

---

## Data and access

### Q: Where are the initial data (.npy) files?

The initial data points are provided in `.npy` file format. You can directly download these files either via the course materials or through the link provided in Mini-lesson 12.8.

To access the data:

1. Extract the zip file containing the `.npy` files.
2. Use `np.load()` in Python to load and work with these files.

The `.npy` files contain the starting data points for each function, which are essential for the optimisation process.

---

### Q: What do I submit? What goes into the portal?

You only submit your **input for each function** (the numeric input string) in the following format:

`x1-x2-x3-...-xn`

You do **not** upload the `.npy` files, code or plots to the portal. For more information on the requirements for the input format, refer to Mini-lesson 12.7.

---

### Q: How many data points should I use? Should I use all the provided data points?

Yes. You should use **all** the data points provided in the initial `.npy` files for each function. If you see more than ten data points for any function, this is expected, and you should use all of them.

Initial data points per function:

| Function | Data points | Dimensions | Shape   |
|----------|-------------|------------|---------|
| 1        | 10          | 2D         | (10, 2) |
| 2        | 10          | 2D         | (10, 2) |
| 3        | 15          | 3D         | (15, 3) |
| 4        | 30          | 4D         | (30, 4) |
| 5        | 20          | 4D         | (20, 4) |
| 6        | 20          | 5D         | (20, 5) |
| 7        | 30          | 6D         | (30, 6) |
| 8        | 40          | 8D         | (40, 8) |

These data points define the initial state of the model and support the Bayesian optimisation process. In the first submission, use them as provided. In future submissions, append new data (feedback from your optimisation) to these initial data sets.

---

### Q: Should inputs be random or based on the initial data?

Inputs should be **based on the initial data** provided. You may use methods such as random search or grid search, but you must begin with the initial data points (from the `.npy` files) as your foundation. From there, you can apply techniques such as Bayesian optimisation that build on these starting points. Your approach should use the provided data as the starting point for further exploration and optimisation, and you should be able to explain your strategy.

---

### Q: How do I read the initial data files?

Example for Function 1:

```python
import numpy as np
# Load initial data
X = np.load(r"C:\initial_data\function_1\initial_inputs.npy")
Y = np.load(r"C:\initial_data\function_1\initial_outputs.npy")
```

(Adjust the path to match your project layout, e.g. `initial_data/function_1/`.)

---

## Appending data

After each submission, the portal returns the **inputs you submitted** and the **corresponding outputs** for each function. You must **append** these new points to your data so that the next run of your optimizer uses the full history (initial data + all previous feedback).

### When to append

- **After every submission**, once you receive the processed results (inputs and outputs) from the portal.
- Append **before** you run the next round of optimisation so your surrogate model sees all available data.

### What to append

- For **each function**: one new row of **inputs** (the `x` you submitted) and one new **output** (the `y` returned).
- Keep inputs and outputs in the same order: the \(i\)-th row in your input array must correspond to the \(i\)-th value in your output array.

### Workflow

1. **Receive** the portal feedback (this submission’s inputs and outputs, or a cumulative download).
2. **Load** your current data for each function (initial data plus any already-appended points).
3. **Append** the new input row(s) and output value(s).
4. **Save** the updated arrays (to a local copy; do not overwrite the original challenge `initial_data` if it is read-only).
5. **Re-run** your optimisation (e.g. fit surrogate, choose next point) using the updated data.

### Example: append a single new point (Function 1)

You submitted one new input for Function 1 and received one output. Append them like this:

```python
import numpy as np
from pathlib import Path

# Paths: use a local folder for updated data (e.g. data/problems/ or a copy of initial_data)
data_dir = Path("initial_data/function_1")  # or your local copy
X = np.load(data_dir / "initial_inputs.npy")
Y = np.load(data_dir / "initial_outputs.npy")

# New point from portal feedback (example)
x_new = np.array([0.472352, 0.625531], dtype=np.float64)   # shape (2,) for 2D
y_new = np.array([-1.802e-144], dtype=np.float64)         # shape (1,)

# Append: new input as one row, new output as one scalar
X_updated = np.vstack((X, x_new))
Y_updated = np.append(Y, y_new)

# Save to a local file (e.g. do not overwrite read-only initial_data)
out_dir = Path("data/problems/function_1")
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / "inputs.npy", X_updated)
np.save(out_dir / "outputs.npy", Y_updated)
```

### Example: append new points for all 8 functions

If you have one new input and one new output per function from the portal:

```python
import numpy as np
from pathlib import Path

def append_feedback(function_id, x_new, y_new, initial_data_dir="initial_data", out_dir="data/problems"):
    """Load current data for one function, append one (x, y), save updated arrays."""
    base = Path(initial_data_dir) / f"function_{function_id}"
    X = np.load(base / "initial_inputs.npy")
    Y = np.load(base / "initial_outputs.npy")
    x_new = np.atleast_2d(x_new)
    X_updated = np.vstack((X, x_new))
    Y_updated = np.append(Y, y_new)
    out = Path(out_dir) / f"function_{function_id}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "inputs.npy", X_updated)
    np.save(out / "outputs.npy", Y_updated)
    return X_updated, Y_updated

# Example: feedback from portal (one new point per function)
feedback = {
    1: (np.array([0.472352, 0.625531]), -1.802e-144),
    2: (np.array([0.977791, 0.261960]), 0.02132),
    # ... 3–8 similarly
}
for fid, (x_new, y_new) in feedback.items():
    append_feedback(fid, x_new, np.array([y_new]))
```

### Using cumulative downloads from the portal

If the portal lets you **download cumulative NumPy arrays** (all inputs/outputs from submission 1 through the current one), you can use those files directly as your “current data” and do not need to manually append each time—just load the latest cumulative file for the next run. If you prefer to keep a single copy that you update yourself, use the append workflow above and point your loader at your local `data/problems/` (or similar) instead of `initial_data/`.

### Best practices

- **Do not modify** the original `initial_data/` if it is provided as read-only; keep updated data in a separate directory (e.g. `data/problems/function_N/`).
- **Use all data** when fitting the surrogate: initial points + every appended point from previous submissions.
- **Check shapes**: for 2D functions, each new input should be shape `(2,)` or `(1, 2)`; for 8D, `(8,)` or `(1, 8)`. Outputs are typically scalars, so `np.append(Y, y_new)` is correct for a single new value.

For **what you receive** after each submission and **submission format**, see **Submission format and feedback** below.

---

## Problem setup and goals

### Q: What is the primary goal of this project?

This capstone project puts you in the role of an ML practitioner solving real-world optimisation challenges. You will:

- Develop practical skills in **Bayesian optimisation** – used when testing is expensive, time-consuming or resource-intensive (e.g. drug formulations, manufacturing parameters, hyperparameter tuning).
- Learn to make informed decisions with limited data and build a **portfolio piece** that demonstrates advanced optimisation techniques.

You need to optimise **eight synthetic black-box functions**. Each function:

- Has a different input dimensionality (2D to 8D).
- Returns a single output value.
- Is framed as a **maximisation problem** (even if the real-world analogy is minimisation; it is transformed so that higher is better).

Your task is to find the input combination that **maximises** the output, using limited queries and the initial data provided in `.npy` files.

---

### Q: What does "black-box function" mean in this capstone project? Why use a "black-box function"?

A black-box function mirrors real-world scenarios where you cannot access the internal workings of a system (e.g. proprietary algorithms, complex simulations, biological processes). It teaches you to optimise when you can only observe results, not mechanisms.

In this project it means:

- You **cannot** see the internal formula or logic of the function.
- You can only **query** it occasionally (limited evaluations).
- You must rely on **observed inputs and outputs** to guide your optimisation strategy.

---

### Q: What is provided to me?

You will be provided with:

- **Initial data in `.npy` files:**
  - `initial_inputs.npy` – input combinations tried so far.
  - `initial_outputs.npy` – corresponding outputs.
- A **description of each function** and its real-world analogy.

---

### Q: Does my work need a positive result?

No. During optimisation, the outputs of the objective functions **may be negative by design** (e.g. when penalties are applied). This is expected; intermediate output values may be negative while you explore the input space.

- **Input values** must always lie in the range **0.000000 to 0.999999** (≥ 0 and &lt; 1).
- Given these valid inputs, the objective functions may return **positive or negative** outputs.
- Each task is a **maximisation problem**. Your strategy should aim to **maximise** the output value of each objective function.

---

### Q: How should I handle minimisation problems?

Some real-world analogies involve minimisation (or side effects). These are transformed into maximisation by **negating the output** so that higher values are better. Your optimisation algorithm will always aim to maximise this transformed score.

---

## Method and process

### Q: What optimisation method should I use?

The **recommended approach is Bayesian optimisation** because:

- Evaluations are expensive (limited queries).
- Functions may be non-linear, noisy and have multiple local maxima.
- Bayesian optimisation balances **exploration** (trying new areas) and **exploitation** (refining near promising points).

---

### Q: What are the steps to complete the project?

#### First submission

1. **Understand the problem and function dimensionality**
   - Fully understand the problem statement and input–output relationships for each function.
   - Check the dimensionality of each function and the number of variables and their constraints.

2. **Load the initial data**
   - Load the initial input and output data for each function using `np.load()`. These are the starting point for Bayesian optimisation.

3. **Define the search space**
   - Define the range of values each input can take (e.g. [0, 1] per variable, or more complex boundaries as needed).

4. **Implement Bayesian optimisation**
   - **Fit a surrogate model:** e.g. a Gaussian process (GP) for predictions and uncertainty estimates.
   - **Define an acquisition function:** e.g. Expected Improvement (EI) or Upper Confidence Bound (UCB) to decide where to query next.
   - **Iteratively select new points:** use the acquisition function to suggest new points; after each query, update the surrogate with the new input–output pair.

5. **Run the optimisation** for a fixed number of iterations (e.g. 20 queries).

6. **Record the best input and output** found (highest output).

7. **Visualise progress and summarise results**
   - Plot how the best output improves over time.
   - Show how the acquisition function and surrogate model evolve.
   - Summarise in a report or graphs.

#### Following submissions

1. **Append data from the previous submission**
   - Append the new feedback (input–output pairs) to the existing data set for each function.
   - Keep accumulating data across submissions.
   - **See the [Appending data](#appending-data) section** for when to append, workflow, and example code (single function and all 8 functions).

2. **Repeat the same steps**
   - Load the existing data (including all previous submissions).
   - Update the search space if needed.
   - Run Bayesian optimisation with the updated data set.
   - Record new best inputs/outputs and visualise progress.

Goal: refine the model by adding more data and further optimising to achieve better output.

**Notes for following submissions:**

- **Data storage:** Keep track of all previously collected data to build a larger data set over time.
- **Optimisation continuity:** Use all available data so the model is refined over time.
- After each submission, follow the submission guidelines (see link in the original course materials).

---

## Deliverables and repository

### Q: What is the primary deliverable for this activity?

The primary deliverable is a **public GitHub repository** that contains all materials related to the Bayesian black-box optimisation (BBO) capstone project. The repository should clearly present the project from start to finish and be suitable as a **portfolio artefact**.

As you work through the query submission process and guided reflection activities, maintain a clear and organised record in this repository. You will refine and submit it as the final deliverable in Module 25.

---

### Q: What code should be included in the repository?

The repository should include **well-documented Python code**, preferably in **Jupyter Notebooks**, that implements Bayesian optimisation for all eight functions. The code should be clear, reproducible and easy to follow.

---

### Q: Are results and visualisations required?

Yes. Deliverables should include **plots** that show optimisation progress (e.g. convergence or performance over iterations) to demonstrate how the optimisation performs for each function.

---

### Q: What documentation is required?

The repository must include:

- A **README** with an approximately 100-word non-technical summary of the purpose, process and results of the project.
- A completed **datasheet** describing the data used, including limitations and context.
- A completed **model card** outlining model behaviour, assumptions, limitations and interpretability considerations.

---

### Q: Is a written summary of findings required?

Yes. Deliverables should summarise:

- The best input and output values found for each function.
- Why Bayesian optimisation was chosen for this problem.
- Challenges encountered and key insights gained.

This summary can be in the notebook(s) or the README.

---

### Q: How should data be handled in the repository?

Do not store large data sets directly on GitHub. Describe them in the README and include a link to the external data source if applicable.

---

### Q: How should the repository be submitted?

The repository must be set to **public**, and the **GitHub link** should be submitted via the discussion board as the final submission.

---

## Submission format and feedback

### Q: What type of processed data points will I receive after every submission?

After submitting to the capstone portal, you will receive:

- The **input values** you submitted for each function.
- The **corresponding output values** (format example below).
- The ability to **download files** containing cumulative NumPy array data points from the first submission through the current one.

Example format (values are illustrative):

- **This submission's input values:**  
  Function 1: [0.472352, 1.055087], Function 2: [0.977791, 0.261960], …

- **This submission's output values:**  
  Function 1: -1.802…e-144, Function 2: 0.0213…, …

(Exact format may vary; follow the portal’s download options.)

---

### Q: How long does it take to receive the processed data point with every submission?

The system will process each submission **at the end of each module**.

---

### Q: What if I make a late submission?

Late submissions will be processed within **24–48 hours**. If your submission is not processed within 48 hours, raise a support ticket.

---

### Q: What should I do if I receive the error "Function 1 input does not match the required pattern"?

This error means the submission format does not meet the required specifications. Each query must follow:

**Format:** `x1-x2-x3-...-xn`

Requirements:

- Each value must **start with 0** (e.g. `0.123456`).
- Each value must have **exactly six decimal places**.
- Values must be **separated by hyphens, with no spaces**.

Examples:

- **Incorrect:** `0.498317 - 0.625531` (spaces around hyphen).
- **Correct:** `0.498317-0.625531`.

Action: Review all function inputs; remove spaces around hyphens; ensure every number has six decimal places and starts with 0. If the issue persists, raise a support ticket.

---

## Tips for success

- Start with **Function 1 (2D)** to understand the workflow.
- Gradually move to **higher dimensions**.
- **Visualise results** to check whether optimisation is improving.
- **Document your reasoning and choices** clearly.

---

*Source: Capstone Project FAQs (PDF). For official deadlines, links and submission portals, refer to the course platform and Mini-lessons 12.7 and 12.8.*
