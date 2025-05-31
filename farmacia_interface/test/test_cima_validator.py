# test_cima_validator.py
from farmacia_interface.core.cima_validator import CIMAValidator

validator = CIMAValidator(debug=True)

# Probar con códigos de ejemplo
test_codes = ["8470007200116", "8470006514733", "1234567890123"]

for code in test_codes:
    result = validator.validar_medicamento(code)
    print(f"Código {code}: {'✅ Válido' if result['valido'] else '❌ Inválido'}")
    if result['valido']:
        print(f"  Nombre: {result.get('nombre', 'N/A')}")
