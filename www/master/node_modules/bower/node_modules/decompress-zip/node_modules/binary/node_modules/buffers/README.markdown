buffers
=======

Treat a collection of Buffers as a single contiguous partially mutable Buffer.

Where possible, operations execute without creating a new Buffer and copying
everything over.

This is a cleaner more Buffery rehash of
[bufferlist](http://github.com/substack/node-bufferlist).

example
=======

slice
-----

    var Buffers = require('buffers');
    var bufs = Buffers();
    bufs.push(new Buffer([1,2,3]));
    bufs.push(new Buffer([4,5,6,7]));
    bufs.push(new Buffer([8,9,10]));
    
    console.dir(bufs.slice(2,8))

output:

    $ node examples/slice.js 
    <Buffer 03 04 05 06 07 08>

splice
------

    var Buffers = require('buffers');
    var bufs = Buffers([
        new Buffer([1,2,3]),
        new Buffer([4,5,6,7]),
        new Buffer([8,9,10]),
    ]);
    
    var removed = bufs.splice(2, 4);
    console.dir({
        removed : removed.slice(),
        bufs : bufs.slice(),
    });
    
output:

    $ node examples/splice.js
    { removed: <Buffer 03 04 05 06>,
      bufs: <Buffer 01 02 07 08 09 0a> }

methods
=======

Buffers(buffers)
----------------

Create a Buffers with an array of `Buffer`s if specified, else `[]`.

.push(buf1, buf2...)
--------------------

Push buffers onto the end. Just like `Array.prototype.push`.

.unshift(buf1, buf2...)
-----------------------

Unshift buffers onto the head. Just like `Array.prototype.unshift`.

.slice(i, j)
------------

Slice a range out of the buffer collection as if it were contiguous.
Works just like the `Array.prototype.slice` version.

.splice(i, howMany, replacements)
---------------------------------

Splice the buffer collection as if it were contiguous.
Works just like `Array.prototype.splice`, even the replacement part!

.copy(dst, dstStart, start, end)
--------------------------------

Copy the buffer collection as if it were contiguous to the `dst` Buffer with the
specified bounds.
Works just like `Buffer.prototype.copy`.

.get(i)
-------

Get a single element at index `i`.

.set(i, x)
----------

Set a single element's value at index `i`.

.indexOf(needle, offset)
----------

Find a string or buffer `needle` inside the buffer collection. Returns
the position of the search string or -1 if the search string was not
found.

Provide an `offset` to skip that number of characters at the beginning
of the search. This can be used to find additional matches.

This function will return the correct result even if the search string
is spread out over multiple internal buffers.

.toBuffer()
-----------

Convert the buffer collection to a single buffer, equivalent with `.slice(0, buffers.length)`;

.toString(encoding, start, end)
-----------

Decodes and returns a string from the buffer collection.
Works just like `Buffer.prototype.toString`
