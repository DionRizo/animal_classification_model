import paramiko
from scp import SCPClient
import threading

# Lista de IPs de los esclavos
slave_ips = [
    '18.119.122.168',
    '18.191.62.86',
    '18.226.181.67',
    '18.227.49.57',
    '18.118.30.31',
    '18.220.151.108'
]

# Ruta de la llave privada
key_path = 'id_rsa'
key = paramiko.RSAKey.from_private_key_file(key_path)

# Contenido del Dockerfile
dockerfile_content = """
FROM pytorch/pytorch

WORKDIR /app
COPY main.py /app/main.py
COPY model_scripted_efficientnet_lr0.001_aughigh.pt /app/model_scripted_efficientnet_lr0.001_aughigh.pt

RUN pip install fastapi uvicorn pillow python-multipart

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

def setup_slave(slave_ip):
    print(f"[+] Conectando a {slave_ip}")
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(slave_ip, username='ec2-user', pkey=key)

    # Subir archivos con SCP
    scp_client = SCPClient(ssh_client.get_transport())
    scp_client.put('../main.py', '/home/ec2-user/main.py')
    scp_client.put('../model_scripted_efficientnet_lr0.001_aughigh.pt', '/home/ec2-user/model_scripted_efficientnet_lr0.001_aughigh.pt')
    scp_client.close()
    print(f"[+] Archivos subidos a {slave_ip}")

    # Crear el Dockerfile
    sftp_client = ssh_client.open_sftp()
    with sftp_client.open('/home/ec2-user/Dockerfile', 'w') as dockerfile:
        dockerfile.write(dockerfile_content)
    sftp_client.close()
    print(f"[+] Dockerfile creado en {slave_ip}")

    # Construir imagen y correr el contenedor
    commands = [
        'docker build -t fastapi_app .',
        'docker run -d -p 8000:8000 fastapi_app'
    ]
    for cmd in commands:
        stdin, stdout, stderr = ssh_client.exec_command(cmd)
        print(f"[{slave_ip}] {stdout.read().decode()}")
        print(f"[{slave_ip}] {stderr.read().decode()}")

    ssh_client.close()
    print(f"[+] Proceso completado para {slave_ip}\n")


# Usar hilos para hacerlo en paralelo
threads = []

for ip in slave_ips:
    thread = threading.Thread(target=setup_slave, args=(ip,))
    thread.start()
    threads.append(thread)

for t in threads:
    t.join()

print("✓ Todos los esclavos fueron configurados.")
