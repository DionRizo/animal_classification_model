import boto3
import paramiko

# Crear una sesión de boto3 (no se usa directamente en este script)
ec2_client = boto3.client('ec2', region_name='us-east-2')

# IPs de los nodos
micro_instances = [
    '18.119.122.168',
    '18.191.62.86',
    '18.226.181.67',
    '18.227.49.57'
]

large_instances = [
    '18.118.30.31',
    '18.220.151.108'
]

master_ip = '18.118.164.84'  # IP del master

# Configurar la conexión SSH
key_path = 'id_rsa'  # archivo generado por Terraform
key = paramiko.RSAKey.from_private_key_file(key_path)
ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Conectar al master
ssh_client.connect(master_ip, username='ec2-user', pkey=key)

# Generar el bloque de upstream con pesos
upstream_servers = ""

# Agregar micro instances con peso 1
for ip in micro_instances:
    upstream_servers += f"        server {ip}:8000 weight=1;\n"

# Agregar large instances con peso 3
for ip in large_instances:
    upstream_servers += f"        server {ip}:8000 weight=3;\n"

# Archivo de configuración NGINX
nginx_config = f"""
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {{
    worker_connections 1024;
}}

http {{
    upstream fastapi_app {{
{upstream_servers.strip()}
    }}

    server {{
        listen 80;

        location / {{
            proxy_pass http://fastapi_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}
    }}
}}
"""

# Subir el archivo nginx.conf al master
sftp_client = ssh_client.open_sftp()
with sftp_client.open('/home/ec2-user/nginx.conf', 'w') as conf_file:
    conf_file.write(nginx_config)
sftp_client.close()

# Mover archivo y reiniciar nginx
commands = [
    "sudo mv /home/ec2-user/nginx.conf /etc/nginx/nginx.conf",
    "sudo systemctl restart nginx"
]

for cmd in commands:
    stdin, stdout, stderr = ssh_client.exec_command(cmd)
    print(stdout.read().decode(), stderr.read().decode())

ssh_client.close()
print("✅ Configuración de NGINX con balanceo de carga ponderado aplicada con éxito.")
