# setup_dvc.py
import dvc_utils as dvc

# Inicializa DVC en tu proyecto
dvc.init_dvc_repo()

# Añade tu dataset a DVC
dvc.add_data("animals10")

# Configura el pipeline para que incorpore tu modelo
dvc.setup_animal_classification_pipeline()

print("Configuración de DVC completada con éxito")