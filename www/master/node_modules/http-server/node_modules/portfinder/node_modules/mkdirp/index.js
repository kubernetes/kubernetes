var path = require('path');
var fs = require('fs');

var exports = module.exports = function mkdirP (p, mode, f) {
    var cb = f || function () {};
    p = path.resolve(p);
    
    var ps = path.normalize(p).split('/');
    path.exists(p, function (exists) {
        if (exists) cb(null);
        else mkdirP(ps.slice(0,-1).join('/'), mode, function (err) {
            if (err && err.code !== 'EEXIST') cb(err)
            else fs.mkdir(p, mode, function (err) {
                if (err && err.code !== 'EEXIST') cb(err)
                else cb()
            });
        });
    });
};
exports.mkdirp = exports.mkdirP = module.exports;
