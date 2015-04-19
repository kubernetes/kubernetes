var Traverse = require('traverse');
var obj = [ 5, 6, -3, [ 7, 8, -2, 1 ], { f : 10, g : -13 } ];

Traverse(obj).forEach(function (x) {
    if (x < 0) this.update(x + 128);
});

console.dir(obj);
