# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/app

# Copy the requirements file into the container at /usr/src/app
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

COPY entrypoint.sh ./

RUN chmod +x entrypoint.sh

# Use the entrypoint script to run the container
ENTRYPOINT ["./entrypoint.sh"]