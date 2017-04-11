#/bin/bash

echo -n "Python ujson module, DECODE: "
python -m timeit -s "import ujson; data = open('`dirname $0`/example.json', 'r').read()" 'ujson.loads(data)'

echo -n "Python ujson module, ENCODE: "
python -m timeit -s "import ujson; data = open('`dirname $0`/example.json', 'r').read(); obj = ujson.loads(data)" 'ujson.dumps(obj)'
