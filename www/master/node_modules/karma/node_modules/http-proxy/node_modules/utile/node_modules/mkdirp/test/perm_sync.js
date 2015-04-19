var mkdirp = require('../');
var path = require('path');
var fs = require('fs');
var exists = fs.exists || path.exists;
var test = require('tap').test;

test('sync perm', function (t) {
    t.plan(4);
    var file = '/tmp/' + (Math.random() * (1<<30)).toString(16) + '.json';
    
    mkdirp.sync(file, 0755);
    exists(file, function (ex) {
        t.ok(ex, 'file created');
        fs.stat(file, function (err, stat) {
            t.ifError(err);
            t.equal(stat.mode & 0777, 0755);
            t.ok(stat.isDirectory(), 'target not a directory');
        });
    });
});

test('sync root perm', function (t) {
    t.plan(3);
    
    var file = '/tmp';
    mkdirp.sync(file, 0755);
    exists(file, function (ex) {
        t.ok(ex, 'file created');
        fs.stat(file, function (err, stat) {
            t.ifError(err);
            t.ok(stat.isDirectory(), 'target not a directory');
        })
    });
});
