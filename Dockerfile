FROM debian:11-slim AS build
RUN apt-get update && apt-get upgrade -yy && \
    apt-get install --no-install-suggests --no-install-recommends --yes python3-venv gcc  libpython3-dev && \
    python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel

FROM build AS build-venv
COPY requirements.txt /requirements.txt
RUN /venv/bin/pip install --disable-pip-version-check -r /requirements.txt

# Copy the virtualenv into a distroless image
FROM gcr.io/distroless/python3-debian11
COPY --from=build-venv /venv /venv
COPY captioner.py /app/captioner.py
WORKDIR /app
ARG MODEL="Salesforce/blip-image-captioning-base"
ENV MODEL=${MODEL}
ENTRYPOINT ["/venv/bin/python3", "/venv/bin/uvicorn", "--host", "0.0.0.0", "captioner:app"]
EXPOSE 8000
USER nonroot:nonroot