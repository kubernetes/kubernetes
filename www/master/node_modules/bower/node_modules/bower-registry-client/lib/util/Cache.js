var fs = require('graceful-fs');
var path = require('path');
var async = require('async');
var mkdirp = require('mkdirp');
var LRU = require('lru-cache');
var md5 = require('./md5');

function Cache(dir, options) {
    options = options || {};

    this._dir = dir;
    this._options = options;
    this._cache = this.constructor._cache.get(this._dir);

    if (!this._cache) {
        this._cache = new LRU(options);
        this.constructor._cache.set(this._dir, this._cache);
    }

    if (dir) {
        mkdirp.sync(dir);
    }
}

Cache.prototype.get = function (key, callback) {
    var file;
    var json = this._cache.get(key);

    // Check in memory
    if (json) {
        if (this._hasExpired(json)) {
            this.del(key, callback);
        } else {
            callback(null, json.value);
        }

        return;
    }

    // Check in disk
    if (!this._dir) {
        return callback(null);
    }

    file = this._getFile(key);
    fs.readFile(file, function (err, contents) {
        var json;

        // Check if there was an error reading
        // Note that if the file does not exist then
        // we don't have its value
        if (err) {
            return callback(err.code === 'ENOENT' ? null : err);
        }

        // If there was an error reading the file as json
        // simply assume it doesn't exist
        try {
            json = JSON.parse(contents.toString());
        } catch (e) {
            return this.del(key, callback);  // If so, delete it
        }

        // Check if it has expired
        if (this._hasExpired(json)) {
            return this.del(key, callback);
        }

        this._cache.set(key, json);
        callback(null, json.value);
    }.bind(this));
};

Cache.prototype.set = function (key, value, maxAge, callback) {
    var file;
    var entry;
    var str;

    maxAge = maxAge != null ? maxAge : this._options.maxAge;
    entry = {
        expires: maxAge ? Date.now() + maxAge : null,
        value: value
    };

    // Store in memory
    this._cache.set(key, entry);

    // Store in disk
    if (!this._dir) {
        return callback(null);
    }

    // If there was an error generating the json
    // then there's some cyclic reference or some other issue
    try {
        str = JSON.stringify(entry);
    } catch (e) {
        return callback(e);
    }

    file = this._getFile(key);
    fs.writeFile(file, str, callback);
};

Cache.prototype.del = function (key, callback) {
    // Delete from memory
    this._cache.del(key);

    // Delete from disk
    if (!this._dir) {
        return callback(null);
    }

    fs.unlink(this._getFile(key), function (err) {
        if (err && err.code !== 'ENOENT') {
            return callback(err);
        }

        callback();
    });
};

Cache.prototype.clear = function (callback) {
    var dir = this._dir;

    // Clear in memory cache
    this._cache.reset();

    // Clear everything from the disk
    if (!dir) {
        return callback(null);
    }

    fs.readdir(dir, function (err, files) {
        if (err) {
            return callback(err);
        }

        // Delete every file in parallel
        async.forEach(files, function (file, next) {
            fs.unlink(path.join(dir, file), function (err) {
                if (err && err.code !== 'ENOENT') {
                    return next(err);
                }

                next();
            });
        }, callback);
    });
};

Cache.prototype.reset = function () {
    this._cache.reset();
};

Cache.clearRuntimeCache = function () {
    // Note that _cache refers to the static _cache variable
    // that holds other caches per dir!
    // Do not confuse it with the instance cache

    // Clear cache of each directory
    this._cache.forEach(function (cache) {
        cache.reset();
    });

    // Clear root cache
    this._cache.reset();
};

//-------------------------------

Cache.prototype._hasExpired = function (json) {
    var expires = json.expires;

    if (!expires || this._options.useStale) {
        return false;
    }

    // Check if the key has expired
    return Date.now() > expires;
};

Cache.prototype._getFile = function (key) {
    // Append a truncated md5 to the end of the file to solve case issues
    // on case insensitive file systems
    // See: https://github.com/bower/bower/issues/859
    return path.join(this._dir, encodeURIComponent(key) + '_' + md5(key).substr(0, 5));
};

Cache._cache = new LRU({
    max: 5,
    maxAge: 60 * 30 * 1000  // 30 minutes
});

module.exports = Cache;
