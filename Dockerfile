# Stage 1: The Builder -- Install all dependencies
# We use a Miniconda base image as recommended for rdkit and pytorch
FROM continuumio/miniconda3:latest AS builder

# Set the working directory inside the container
WORKDIR /app

# Copy the environment file and requirements file
COPY environment.yml .
COPY requirements.txt .

# Create the Conda environment from the YAML file
# This installs Python, RDKit, PyTorch, and all pip packages
RUN conda env create -f environment.yml

# --- Pre-download the Hugging Face model ---
# This step is crucial for reproducibility and fast startup.
# It "bakes" the ChemFM model into the image so it doesn't need to be downloaded at runtime.
# We activate the conda environment to use its pip-installed libraries.
RUN conda run -n chemsynth huggingface-cli download --repo-type model ChemFM/ChemFM-3B


# Stage 2: The Final Image -- Create a lean, runnable image
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Copy the activated Conda environment from the builder stage
COPY --from=builder /opt/conda/envs/chemsynth /opt/conda/envs/chemsynth

# Copy the pre-downloaded Hugging Face model cache
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Set environment variables to make the new Conda environment the default
ENV PATH /opt/conda/envs/chemsynth/bin:$PATH
ENV HOME /root

# Copy the rest of the application code into the image
# We copy everything needed for the app to run
COPY app.py .
COPY config.py .
COPY utils/ ./utils/
COPY templates/ ./templates/
COPY static/ ./static/

# IMPORTANT: The core_logic and models directories will be mounted as volumes,
# so we do NOT copy them here. This keeps the image small.

# Expose the port the Flask app runs on
EXPOSE 5000

# The command to run the application when the container starts
# We use gunicorn for a more robust production server than Flask's default
# First, install gunicorn
RUN pip install gunicorn
# Now, define the command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
