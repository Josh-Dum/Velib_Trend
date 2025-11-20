# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY docker/frontend/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary source code
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/

# Expose the port the app runs on
EXPOSE 8501

# Define environment variable to point to the backend
ENV VELIB_API_BASE="http://backend:8000"

# Run the application
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
