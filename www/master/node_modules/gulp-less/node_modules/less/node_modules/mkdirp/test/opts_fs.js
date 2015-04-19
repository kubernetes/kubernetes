var mkdirp = require('../');
var path = require('path');
var test = require('tap').test;
var mockfs = require('mock-fs');

test('opts.fs', function (t) {
    t.plan(5);
    
    var x = Math.floor(Math.random() * Math.pow(16,4)).toString(16);
    var y = Math.floor(Math.random() * Math.pow(16,4)).toString(16);
    var z = Math.floor(Math.random() * Math.pow(16,4)).toString(16);
    
    var file = '/beep/boop/' + [x,y,z].join('/');
    var xfs = mockfs.fs();
    
    mkdirp(file, { fs: xfs, mode: 0755 }, function (err) {
        t.ifError(err);
        xfs.exists(file, function (ex) {
            t.ok(ex, 'created file');
            xfs.stat(file, function (err, stat) {
                t.ifError(err);
                t.equal(stat.mode & 0777, 0755);
                t.ok(stat.isDirectory(), 'target not a directory');
            });
        });
    });
});
