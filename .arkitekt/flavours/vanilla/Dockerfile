FROM julia:latest

RUN apt update && apt install -y python3 python3-pip

RUN pip install --break-system-packages arkitekt[all] juliacall matplotlib Pillow

RUN julia -e 'using Pkg;Pkg.add(url="https://github.com/Marc-3d/multi_quickPIV.git"); Pkg.instantiate()'

RUN python3 -c "from juliacall import Main as jl"

RUN mkdir /app
WORKDIR /app
COPY .arkitekt /app/.arkitekt
COPY app.py /app/app.py
