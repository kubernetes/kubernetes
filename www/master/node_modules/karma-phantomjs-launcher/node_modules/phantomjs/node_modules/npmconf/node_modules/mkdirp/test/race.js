var mkdirp = require('../').mkdirp;
var path = require('path');
var fs = require('fs');
var exists = fs.exists || path.exists;
var test = require('tap').test;

test('race', function (t) {
    t.plan(6);
    var ps = [ '', 'tmp' ];
    
    for (var i = 0; i < 25; i++) {
        var dir = Math.floor(Math.random() * Math.pow(16,4)).toString(16);
        ps.push(dir);
    }
    var file = ps.join('/');
    
    var res = 2;
    mk(file, function () {
        if (--res === 0) t.end();
    });
    
    mk(file, function () {
        if (--res === 0) t.end();
    });
    
    function mk (file, cb) {
        mkdirp(file, 0755, function (err) {
            t.ifError(err);
            exists(file, function (ex) {
                t.ok(ex, 'file created');
                fs.stat(file, function (err, stat) {
                    t.ifError(err);
                    t.equal(stat.mode & 0777, 0755);
                    t.ok(stat.isDirectory(), 'target not a directory');
                    if (cb) cb();
                });
            })
        });
    }
});
