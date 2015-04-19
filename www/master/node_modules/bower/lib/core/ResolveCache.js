var fs = require('graceful-fs');
var path = require('path');
var mout = require('mout');
var Q = require('q');
var mkdirp = require('mkdirp');
var rimraf = require('rimraf');
var LRU = require('lru-cache');
var lockFile = require('lockfile');
var semver = require('../util/semver');
var readJson = require('../util/readJson');
var copy = require('../util/copy');
var md5 = require('../util/md5');

function ResolveCache(config) {
    // TODO: Make some config entries, such as:
    //       - Max MB
    //       - Max versions per source
    //       - Max MB per source
    //       - etc..
    this._config = config;
    this._dir = this._config.storage.packages;
    this._lockDir = this._config.storage.packages;

    mkdirp.sync(this._lockDir);

    // Cache is stored/retrieved statically to ensure singularity
    // among instances
    this._cache = this.constructor._cache.get(this._dir);
    if (!this._cache) {
        this._cache = new LRU({
            max: 100,
            maxAge: 60 * 5 * 1000  // 5 minutes
        });
        this.constructor._cache.set(this._dir, this._cache);
    }

    // Ensure dir is created
    mkdirp.sync(this._dir);
}

// -----------------

ResolveCache.prototype.retrieve = function (source, target) {
    var sourceId = md5(source);
    var dir = path.join(this._dir, sourceId);
    var that = this;

    target = target || '*';

    return this._getVersions(sourceId)
    .spread(function (versions) {
        var suitable;

        // If target is a semver, find a suitable version
        if (semver.validRange(target)) {
            suitable = semver.maxSatisfying(versions, target, true);

            if (suitable) {
                return suitable;
            }
        }

        // If target is '*' check if there's a cached '_wildcard'
        if (target === '*') {
            return mout.array.find(versions, function (version) {
                return version === '_wildcard';
            });
        }

        // Otherwise check if there's an exact match
        return mout.array.find(versions, function (version) {
            return version === target;
        });
    })
    .then(function (version) {
        var canonicalDir;

        if (!version) {
            return [];
        }

        // Resolve with canonical dir and package meta
        canonicalDir = path.join(dir, encodeURIComponent(version));
        return that._readPkgMeta(canonicalDir)
        .then(function (pkgMeta) {
            return [canonicalDir, pkgMeta];
        }, function () {
            // If there was an error, invalidate the in-memory cache,
            // delete the cached package and try again
            that._cache.del(sourceId);

            return Q.nfcall(rimraf, canonicalDir)
            .then(function () {
                return that.retrieve(source, target);
            });
        });
    });
};

ResolveCache.prototype.store = function (canonicalDir, pkgMeta) {
    var sourceId;
    var release;
    var dir;
    var pkgLock;
    var promise;
    var that = this;

    promise = pkgMeta ? Q.resolve(pkgMeta) : this._readPkgMeta(canonicalDir);

    return promise
    .then(function (pkgMeta) {
        sourceId = md5(pkgMeta._source);
        release = that._getPkgRelease(pkgMeta);
        dir = path.join(that._dir, sourceId, release);
        pkgLock = path.join(that._lockDir, sourceId + '-' + release + '.lock');

        // Check if destination directory exists to prevent issuing lock at all times
        return Q.nfcall(fs.stat, dir)
        .fail(function (err) {
            var lockParams = { wait: 250, retries: 25, stale: 60000 };
            return Q.nfcall(lockFile.lock, pkgLock, lockParams).then(function () {
                // Ensure other process didn't start copying files before lock was created
                return Q.nfcall(fs.stat, dir)
                .fail(function (err) {
                    // If stat fails, it is expected to return ENOENT
                    if (err.code !== 'ENOENT') {
                        throw err;
                    }

                    // Create missing directory and copy files there
                    return Q.nfcall(mkdirp, path.dirname(dir)).then(function () {
                        return Q.nfcall(fs.rename, canonicalDir, dir)
                        .fail(function (err) {
                            // If error is EXDEV it means that we are trying to rename
                            // across different drives, so we copy and remove it instead
                            if (err.code !== 'EXDEV') {
                                throw err;
                            }

                            return copy.copyDir(canonicalDir, dir);
                        });
                    });
                });
            }).finally(function () {
                lockFile.unlockSync(pkgLock);
            });
        }).finally(function () {
            // Ensure no tmp dir is left on disk.
            return Q.nfcall(rimraf, canonicalDir);
        });
    })
    .then(function () {
        var versions = that._cache.get(sourceId);

        // Add it to the in memory cache
        // and sort the versions afterwards
        if (versions && versions.indexOf(release) === -1) {
            versions.push(release);
            that._sortVersions(versions);
        }

        // Resolve with the final location
        return dir;
    });
};

ResolveCache.prototype.eliminate = function (pkgMeta) {
    var sourceId = md5(pkgMeta._source);
    var release = this._getPkgRelease(pkgMeta);
    var dir = path.join(this._dir, sourceId, release);
    var that = this;

    return Q.nfcall(rimraf, dir)
    .then(function () {
        var versions = that._cache.get(sourceId) || [];
        mout.array.remove(versions, release);

        // If this was the last package in the cache,
        // delete the parent folder (source)
        // For extra security, check against the file system
        // if this was really the last package
        if (!versions.length) {
            that._cache.del(sourceId);

            return that._getVersions(sourceId)
            .spread(function (versions) {
                if (!versions.length) {
                    // Do not keep in-memory cache if it's completely
                    // empty
                    that._cache.del(sourceId);

                    return Q.nfcall(rimraf, path.dirname(dir));
                }
            });
        }
    });
};

ResolveCache.prototype.clear = function () {
    return Q.nfcall(rimraf, this._dir)
    .then(function () {
        return Q.nfcall(fs.mkdir, this._dir);
    }.bind(this))
    .then(function () {
        this._cache.reset();
    }.bind(this));
};

ResolveCache.prototype.reset = function () {
    this._cache.reset();
    return this;
};

ResolveCache.prototype.versions = function (source) {
    var sourceId = md5(source);

    return this._getVersions(sourceId)
    .spread(function (versions) {
        return versions.filter(function (version) {
            return semver.valid(version);
        });
    });
};

ResolveCache.prototype.list = function () {
    var promises;
    var dirs = [];
    var that = this;

    // Get the list of directories
    return Q.nfcall(fs.readdir, this._dir)
    .then(function (sourceIds) {
        promises = sourceIds.map(function (sourceId) {
            return Q.nfcall(fs.readdir, path.join(that._dir, sourceId))
            .then(function (versions) {
                versions.forEach(function (version) {
                    var dir = path.join(that._dir, sourceId, version);
                    dirs.push(dir);
                });
            }, function (err) {
                // Ignore lurking files, e.g.: .DS_Store if the user
                // has navigated throughout the cache
                if (err.code === 'ENOTDIR' && err.path) {
                    return Q.nfcall(rimraf, err.path);
                }

                throw err;
            });
        });

        return Q.all(promises);
    })
    // Read every package meta
    .then(function () {
        promises = dirs.map(function (dir) {
            return that._readPkgMeta(dir)
            .then(function (pkgMeta) {
                return {
                    canonicalDir: dir,
                    pkgMeta: pkgMeta
                };
            }, function () {
                // If it fails to read, invalidate the in memory
                // cache for the source and delete the entry directory
                var sourceId = path.basename(path.dirname(dir));
                that._cache.del(sourceId);

                return Q.nfcall(rimraf, dir);
            });
        });

        return Q.all(promises);
    })
    // Sort by name ASC & release ASC
    .then(function (entries) {
        // Ignore falsy entries due to errors reading
        // package metas
        entries = entries.filter(function (entry) {
            return !!entry;
        });

        return entries.sort(function (entry1, entry2) {
            var pkgMeta1 = entry1.pkgMeta;
            var pkgMeta2 = entry2.pkgMeta;
            var comp = pkgMeta1.name.localeCompare(pkgMeta2.name);

            // Sort by name
            if (comp) {
                return comp;
            }

            // Sort by version
            if (pkgMeta1.version && pkgMeta2.version) {
                return semver.compare(pkgMeta1.version, pkgMeta2.version);
            }
            if (pkgMeta1.version) {
                return -1;
            }
            if (pkgMeta2.version) {
                return 1;
            }

            // Sort by target
            return pkgMeta1._target.localeCompare(pkgMeta2._target);
        });
    });
};

// ------------------------

ResolveCache.clearRuntimeCache = function () {
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

// ------------------------

ResolveCache.prototype._getPkgRelease = function (pkgMeta) {
    var release = pkgMeta.version || (pkgMeta._target === '*' ? '_wildcard' : pkgMeta._target);

    // Encode some dangerous chars such as / and \
    release = encodeURIComponent(release);

    return release;
};

ResolveCache.prototype._readPkgMeta = function (dir) {
    var filename = path.join(dir, '.bower.json');

    return readJson(filename)
    .spread(function (json) {
        return json;
    });
};

ResolveCache.prototype._getVersions = function (sourceId) {
    var dir;
    var versions = this._cache.get(sourceId);
    var that = this;

    if (versions) {
        return Q.resolve([versions, true]);
    }

    dir = path.join(this._dir, sourceId);
    return Q.nfcall(fs.readdir, dir)
    .then(function (versions) {
        // Sort and cache in memory
        that._sortVersions(versions);
        versions = versions.map(decodeURIComponent);
        that._cache.set(sourceId, versions);
        return [versions, false];
    }, function (err) {
        // If the directory does not exists, resolve
        // as an empty array
        if (err.code === 'ENOENT') {
            versions = [];
            that._cache.set(sourceId, versions);
            return [versions, false];
        }

        throw err;
    });
};

ResolveCache.prototype._sortVersions = function (versions) {
    // Sort DESC
    versions.sort(function (version1, version2) {
        var validSemver1 = semver.valid(version1);
        var validSemver2 = semver.valid(version2);

        // If both are semvers, compare them
        if (validSemver1 && validSemver2) {
            return semver.rcompare(version1, version2);
        }

        // If one of them are semvers, give higher priority
        if (validSemver1) {
            return -1;
        }
        if (validSemver2) {
            return 1;
        }

        // Otherwise they are considered equal
        return 0;
    });
};

// ------------------------

ResolveCache._cache = new LRU({
    max: 5,
    maxAge: 60 * 30 * 1000  // 30 minutes
});

module.exports = ResolveCache;
