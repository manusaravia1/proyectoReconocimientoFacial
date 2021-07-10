__author__ = 'Manuel Saravia Enrech'

import random
import time
from redisDbase import rdb


# For testing RedisDbase module
# Importamos el objeto rdb de la clase RedisDbase
# Chequeamos todos los métodos de la clase RedisDbase
# Primero chequeamos todos los métodos
# Luego medimos el tiempo de inserción y recuperación de encodings en Redis

def test_RedisDbase():
	print(">>>>> Inicio test_RedisDbase <<<<<")
	# Vaciamos la db
	print(">> Vaciamos la db redis")
	rdb.empty()
	# Comprobamos que la db está vacía
	print("isEmpty = {}".format(rdb.isEmpty()))
	print("howMany = {}".format(rdb.howMany()))
	# Añadimos un par de "encodings" simulados
	print(">> Añadimos un par de 'encodings' simulados")
	l1 = [0.854, 1.12, 2.21, 3.05, 88.0, 33.22, 99.123]
	l2 = [0.24, 0.122, 3.214, 0.05, 78.0, 53.02, 9.12]
	rdb.addEncoding('MSE1', l1)
	rdb.addEncoding('MSE2', l2)
	# Comprobamos que la db ahora no está vacía
	print("isEmpty = {}".format(rdb.isEmpty()))
	print("howMany = {}".format(rdb.howMany()))
	# Recuperamos el par de "encodings"
	print(">> Recuperamos el par de 'encodings'")
	l1r = rdb.getEncoding('MSE1')
	l2r = rdb.getEncoding('MSE2')
	# Comprobamos que los encodings añadidos y recuperados son iguales
	print(">> Comprobamos que los encodings añadidos y recuperados son iguales")
	print("l1 to redis   = {}".format(l1))
	print("l1 from redis = {}".format(l1r))
	print("l2 to redis   = {}".format(l2))
	print("l2 from redis = {}".format(l2r))
	# Recuperamos todas las claves (nombres)
	print(">> Recuperamos todas las claves")
	print("AllKeys = {}".format(rdb.getAllKeys()))
	# Recuperamos todos los encodings
	print(">> Recuperamos todos los encodings")
	print("AllEncodings = {}".format(rdb.getAllEncodings()))
	print(">>>>> Fin test_RedisDbase <<<<<\n")


def testtime_RedisDbase(n):
	print(">>>>> Inicio testtime_RedisDbase <<<<<")
	rdb.empty()
	# Simulo n encodings para el test
	random.seed(101)
	names = []
	encodings = []
	for i in range(n):
		names.append("Name" + str(i))
		encodings.append([random.uniform(-1, 1) for j in range(128)])

	# Control de tiempo para insertar n encondings
	inicio = time.time()
	for i in range(n):
		rdb.addEncoding(names[i], encodings[i])
	fin = time.time()
	print("  Time consumed to add {} encodings: {:.7f} seconds".format(n, fin - inicio))

	# Control de tiempo para recuperar un encoding aleatorio
	k = random.choice(range(n))
	inicio = time.time()
	encoding = rdb.getEncoding(names[k])
	fin = time.time()
	print("  Time consumed to get a random encoding: {:.7f} seconds".format(fin - inicio))
	print(">>>>> Fin testtime_RedisDbase <<<<<")


if __name__ == "__main__":
	# Pruebas de todos los métodos del módulo
	test_RedisDbase()

	# Pruebas de tiempo de inserción y recuperación de encodings
	# Se puede comprobar como redis opera muy rápido y en tiempo constante O(1)
	testtime_RedisDbase(10)
	testtime_RedisDbase(100)
	testtime_RedisDbase(1000)
	testtime_RedisDbase(10000)
