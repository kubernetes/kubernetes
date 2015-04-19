var fs = require('fs');
var path = require('path');

var mkpath = function mkpath(dirpath, mode, callback) {
    dirpath = path.resolve(dirpath);
    
    if (typeof mode === 'function' || typeof mode === 'undefined') {
        callback = mode;
        mode = 0777 & (~process.umask());
    }
    
    if (!callback) {
        callback = function () {};
    }
    
    fs.stat(dirpath, function (err, stats) {
        if (err) {
            if (err.code === 'ENOENT') {
                mkpath(path.dirname(dirpath), mode, function (err) {
                    if (err) {
                        callback(err);
                    } else {
                        fs.mkdir(dirpath, mode, callback);
                    }
                });
            } else {
                callback(err);
            }
        } else if (stats.isDirectory()) {
            callback(null);
        } else {
            callback(new Error(dirpath + ' exists and is not a directory'));
        }
    });
};

mkpath.sync = function mkpathsync(dirpath, mode) {
    dirpath = path.resolve(dirpath);
    
    if (typeof mode === 'undefined') {
        mode = 0777 & (~process.umask());
    }
    
    try {
        if (!fs.statSync(dirpath).isDirectory()) {
            throw new Error(dirpath + ' exists and is not a directory');
        }
    } catch (err) {
        if (err.code === 'ENOENT') {
            mkpathsync(path.dirname(dirpath), mode);
            fs.mkdirSync(dirpath, mode);
        } else {
            throw err;
        }
    }
};

module.exports = mkpath;

