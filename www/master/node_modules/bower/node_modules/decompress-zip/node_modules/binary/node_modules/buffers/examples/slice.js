var Buffers = require('buffers');
var bufs = Buffers();
bufs.push(new Buffer([1,2,3]));
bufs.push(new Buffer([4,5,6,7]));
bufs.push(new Buffer([8,9,10]));

console.dir(bufs.slice(2,8))

// Output: <Buffer 03 04 05 06 07 08>
