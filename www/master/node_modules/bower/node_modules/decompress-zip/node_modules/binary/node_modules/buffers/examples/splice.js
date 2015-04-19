var Buffers = require('buffers');
var bufs = Buffers([
    new Buffer([1,2,3]),
    new Buffer([4,5,6,7]),
    new Buffer([8,9,10]),
]);

var removed = bufs.splice(2, 4, new Buffer('ab'), new Buffer('cd'));
console.dir({
    removed : removed.slice(),
    bufs : bufs.slice(),
});

/* Output:
{ removed: <Buffer 03 04 05 06>,
  bufs: <Buffer 01 02 07 08 09 0a> }
*/
