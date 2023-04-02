# Use the official Python image as the base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app.py and score.py into the container
COPY app.py score.py ./

# Copy the model and data files into the container
COPY model/model.pkl /app/model/
COPY data/train.csv /app/data/

# Expose the port the app runs on
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]
