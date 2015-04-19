/* Tests borrowed from substack's node-mkdirp
 * https://github.com/substack/node-mkdirp */

var mkpath = require('../');
var path = require('path');
var fs = require('fs');
var test = require('tap').test;

test('root', function (t) {
    // '/' on unix, 'c:/' on windows.
    var file = path.resolve('/');

    mkpath(file, 0755, function (err) {
        if (err) throw err
        fs.stat(file, function (er, stat) {
            if (er) throw er
            t.ok(stat.isDirectory(), 'target is a directory');
            t.end();
        })
    });
});

