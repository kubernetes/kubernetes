/* Tests borrowed from substack's node-mkdirp
 * https://github.com/substack/node-mkdirp */

var mkpath = require('../');
var path = require('path');
var fs = require('fs');
var test = require('tap').test;

test('implicit mode from umask', function (t) {
    t.plan(2);
    var x = Math.floor(Math.random() * Math.pow(16,4)).toString(16);
    var y = Math.floor(Math.random() * Math.pow(16,4)).toString(16);
    var z = Math.floor(Math.random() * Math.pow(16,4)).toString(16);
    
    var file = '/tmp/' + [x,y,z].join('/');
    
    mkpath(file, function (err) {
        if (err) t.fail(err);
        else path.exists(file, function (ex) {
            if (!ex) t.fail('file not created')
            else fs.stat(file, function (err, stat) {
                if (err) t.fail(err)
                else {
                    t.equal(stat.mode & 0777, 0777 & (~process.umask()));
                    t.ok(stat.isDirectory(), 'target not a directory');
                    t.end();
                }
            })
        })
    });
});

