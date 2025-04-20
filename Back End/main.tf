provider "aws" {
  region = "us-east-2"
}

# Generar clave SSH
resource "tls_private_key" "deployer" {
  algorithm = "RSA"
  rsa_bits  = 2048
}

resource "local_file" "private_key" {
  filename        = "id_rsa"
  content         = tls_private_key.deployer.private_key_pem
  file_permission = "0600"
}

resource "aws_key_pair" "deployer" {
  key_name   = "deployer-key"
  public_key = tls_private_key.deployer.public_key_openssh
}

# Grupo de seguridad
resource "aws_security_group" "pods_security_group" {
  name_prefix = "pods-security-group"

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# 4 Instancias SLAVE con FastAPI y PyTorch
resource "aws_instance" "slave" {
  count           = 4
  ami             = "ami-0100e595e1cc1ff7f"
  instance_type   = "t2.micro"
  key_name        = aws_key_pair.deployer.key_name
  security_groups = [aws_security_group.pods_security_group.name]

  root_block_device {
    volume_size = 32
  }

  tags = {
    Name = "slave_micro_${count.index + 1}"
  }
}

# 2  Instancias SLAVE Large con FastAPI y PyTorch
resource "aws_instance" "slave2" {
  count           = 2
  ami             = "ami-060a84cbcb5c14844"
  instance_type   = "t2.large"
  key_name        = aws_key_pair.deployer.key_name
  security_groups = [aws_security_group.pods_security_group.name]

  root_block_device {
    volume_size = 32
  }

  tags = {
    Name = "slave_large_${count.index + 1}"
  }
}

# Instancia MASTER con NGINX reverse proxy
resource "aws_instance" "master" {
  ami             = "ami-0100e595e1cc1ff7f"
  instance_type   = "t2.micro"
  key_name        = aws_key_pair.deployer.key_name
  security_groups = [aws_security_group.pods_security_group.name]

  tags = {
    Name = "master"
  }
}

output "master_ip" {
  value = aws_instance.master.public_ip
}

output "slave_ips" {
  value = [for instance in aws_instance.slave : instance.public_ip]
}

output "slave_ips2" {
  value = [for instance in aws_instance.slave2 : instance.public_ip]
}
