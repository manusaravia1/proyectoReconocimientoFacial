__author__ = 'Manuel Saravia Enrech'

import redis


''' Class RedisDbase: to handle encodings with redis '''
class RedisDbase:
    r = None
    id_db = None

    def __init__(self, id_db):
        self.id_db = id_db
        self.r = redis.Redis(host='localhost', port=6379, db=id_db)

    def addEncoding(self, key, encoding):
        # add (key = nombre, encoding = lista de floats)
        # retorna el numero de elementos del encoding añadido
        return self.r.rpush(key, *encoding)

    def getEncoding(self, key):
        # get (key = nombre) y retorna el encoding (lista de floats)
        return [float(v) for v in self.r.lrange(key, 0, -1)]

    def getAllKeys(self):
        # get () retorna una lista con todas las claves (nombres)
        return [key.decode("utf-8") for key in self.r.keys('*')]

    def getAllEncodings(self):
        return [self.getEncoding(key) for key in self.r.keys('*')]

    def empty(self):
        self.r.flushdb()

    def howMany(self):
        return len(self.r.keys('*'))

    def isEmpty(self):
        return (self.howMany() == 0)


rdb = RedisDbase(1)  # id_db = 1. rdb variable to be imported by other modules