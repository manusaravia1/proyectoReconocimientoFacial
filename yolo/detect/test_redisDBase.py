__author__ = 'Manuel Saravia Enrech'
 
from redisDbase import rdb

# For testing RedisDbase module
# Importamos el objeto rdb de la clase RedisDbase
# Chequeamos todos los métodos de la clase RedisDbase
def test_RedisDbase():
	print(">>>>> Inicio test module redisDbase.py <<<<<")
	# Vaciamos la db
	print(">>> Vaciamos la db redis")
	rdb.empty()
	# Comprobamos que la db está vacía
	print("isEmpty = {}".format(rdb.isEmpty()))
	print("howMany = {}".format(rdb.howMany()))
	# Añadimos un par de "encodings" simulados
	print(">>> Añadimos un par de 'encodings' simulados")
	l1 = [0.854, 1.12, 2.21, 3.05, 88.0, 33.22, 99.123]
	l2 = [0.24, 0.122, 3.214, 0.05, 78.0, 53.02, 9.12]
	rdb.addEncoding('MSE1', l1)
	rdb.addEncoding('MSE2', l2)
	# Comprobamos que la db ahora no está vacía
	print("isEmpty = {}".format(rdb.isEmpty()))
	print("howMany = {}".format(rdb.howMany()))
	# Recuperamos el par de "encodings"
	print(">>> Recuperamos el par de 'encodings'")
	l1r = rdb.getEncoding('MSE1')
	l2r = rdb.getEncoding('MSE2')
	# Comprobamos que los encodings añadidos y recuperados son iguales
	print(">>> Comprobamos que los encodings añadidos y recuperados son iguales")
	print("l1 to redis   = {}".format(l1))
	print("l1 from redis = {}".format(l1r))
	print("l2 to redis   = {}".format(l2))
	print("l2 from redis = {}".format(l2r))
	# Recuperamos todas las claves (nombres)
	print(">>> Recuperamos todas las claves")
	print("AllKeys = {}".format(rdb.getAllKeys()))
	# Recuperamos todos los encodings
	print(">>> Recuperamos todos los encodings")
	print("AllEncodings = {}".format(rdb.getAllEncodings()))
	print(">>>>> Fin test module redisDbase.py <<<<<")


if __name__ == "__main__":
	test_RedisDbase()

