#!/bin/sh

gnuplot >size-rate.jpg << _EOF_

set terminal png nocrop enhanced font verdana 12 size 640,480
set logscale x
set logscale y
set grid
set xlabel 'Serialized object size, octets'
set ylabel 'decode(encode(obj)) rate, 1/sec'
plot '00' using 1:2 title 'json' smooth bezier, '00' using 1:3 title 'msgpack' smooth bezier, '00' using 1:4 title 'bison' smooth bezier

_EOF_
