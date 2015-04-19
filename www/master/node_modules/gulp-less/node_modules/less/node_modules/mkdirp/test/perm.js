var mkdirp = require('../');
var path = require('path');
var fs = require('fs');
var exists = fs.exists || path.exists;
var test = require('tap').test;

test('async perm', function (t) {
    t.plan(5);
    var file = '/tmp/' + (Math.random() * (1<<30)).toString(16);
    
    mkdirp(file, 0755, function (err) {
        t.ifError(err);
        exists(file, function (ex) {
            t.ok(ex, 'file created');
            fs.stat(file, function (err, stat) {
                t.ifError(err);
                t.equal(stat.mode & 0777, 0755);
                t.ok(stat.isDirectory(), 'target not a directory');
            })
        })
    });
});

test('async root perm', function (t) {
    mkdirp('/tmp', 0755, function (err) {
        if (err) t.fail(err);
        t.end();
    });
    t.end();
});
