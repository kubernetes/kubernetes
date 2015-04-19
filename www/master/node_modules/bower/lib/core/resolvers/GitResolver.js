var util = require('util');
var path = require('path');
var Q = require('q');
var chmodr = require('chmodr');
var rimraf = require('rimraf');
var mkdirp = require('mkdirp');
var which = require('which');
var LRU = require('lru-cache');
var mout = require('mout');
var Resolver = require('./Resolver');
var semver = require('../../util/semver');
var createError = require('../../util/createError');

var hasGit;

// Check if git is installed
try {
    which.sync('git');
    hasGit = true;
} catch (ex) {
    hasGit = false;
}

function GitResolver(decEndpoint, config, logger) {
    // Set template dir to the empty directory so that user templates are not run
    // This environment variable is not multiple config aware but it's not documented
    // anyway
    mkdirp.sync(config.storage.empty);
    process.env.GIT_TEMPLATE_DIR = config.storage.empty;

    Resolver.call(this, decEndpoint, config, logger);

    if (!hasGit) {
        throw createError('git is not installed or not in the PATH', 'ENOGIT');
    }
}

util.inherits(GitResolver, Resolver);
mout.object.mixIn(GitResolver, Resolver);

// -----------------

GitResolver.prototype._hasNew = function (canonicalDir, pkgMeta) {
    var oldResolution = pkgMeta._resolution || {};

    return this._findResolution()
    .then(function (resolution) {
        // Check if resolution types are different
        if (oldResolution.type !== resolution.type) {
            return true;
        }

        // If resolved to a version, there is new content if the tags are not equal
        if (resolution.type === 'version' && semver.neq(resolution.tag, oldResolution.tag)) {
            return true;
        }

        // As last check, we compare both commit hashes
        return resolution.commit !== oldResolution.commit;
    });
};

GitResolver.prototype._resolve = function () {
    var that = this;

    return this._findResolution()
    .then(function () {
        return that._checkout()
        // Always run cleanup after checkout to ensure that .git is removed!
        // If it's not removed, problems might arise when the "tmp" module attempts
        // to delete the temporary folder
        .fin(function () {
            return that._cleanup();
        });
    });
};

// -----------------

// Abstract functions that should be implemented by concrete git resolvers
GitResolver.prototype._checkout = function () {
    throw new Error('_checkout not implemented');
};

GitResolver.refs = function (source) {
    throw new Error('refs not implemented');
};

// -----------------

GitResolver.prototype._findResolution = function (target) {
    var err;
    var self = this.constructor;
    var that = this;

    target = target || this._target || '*';

    // Target is a commit, so it's a stale target (not a moving target)
    // There's nothing to do in this case
    if ((/^[a-f0-9]{40}$/).test(target)) {
        this._resolution = { type: 'commit', commit: target };
        return Q.resolve(this._resolution);
    }

    // Target is a range/version
    if (semver.validRange(target)) {
        return self.versions(this._source, true)
        .then(function (versions) {
            var versionsArr,
                version,
                index;

            versionsArr = versions.map(function (obj) { return obj.version; });

            // If there are no tags and target is *,
            // fallback to the latest commit on master
            if (!versions.length && target === '*') {
                return that._findResolution('master');
            }

            versionsArr = versions.map(function (obj) { return obj.version; });
            // Find a satisfying version, enabling strict match so that pre-releases
            // have lower priority over normal ones when target is *
            index = semver.maxSatisfyingIndex(versionsArr, target, true);
            if (index !== -1) {
                version = versions[index];
                return that._resolution = { type: 'version', tag: version.tag, commit: version.commit };
            }

            // Check if there's an exact branch/tag with this name as last resort
            return Q.all([
                self.branches(that._source),
                self.tags(that._source)
            ])
            .spread(function (branches, tags) {
                // Use hasOwn because a branch/tag could have a name like "hasOwnProperty"
                if (mout.object.hasOwn(tags, target)) {
                    return that._resolution = { type: 'tag', tag: target, commit: tags[target] };
                }
                if (mout.object.hasOwn(branches, target)) {
                    return that._resolution = { type: 'branch', branch: target, commit: branches[target] };
                }

                throw createError('No tag found that was able to satisfy ' + target, 'ENORESTARGET', {
                    details: !versions.length ?
                        'No versions found in ' + that._source :
                        'Available versions: ' + versions.map(function (version) { return version.version; }).join(', ')
                });
            });
        });
    }

    // Otherwise, target is either a tag or a branch
    return Q.all([
        self.branches(that._source),
        self.tags(that._source)
    ])
    .spread(function (branches, tags) {
        // Use hasOwn because a branch/tag could have a name like "hasOwnProperty"
        if (mout.object.hasOwn(tags, target)) {
            return that._resolution = { type: 'tag', tag: target, commit: tags[target] };
        }
        if (mout.object.hasOwn(branches, target)) {
            return that._resolution = { type: 'branch', branch: target, commit: branches[target] };
        }

        if ((/^[a-f0-9]{4,40}$/).test(target)) {
            if (target.length < 12) {
                that._logger.warn(
                    'short-sha',
                    'Consider using longer commit SHA to avoid conflicts'
                );
            }

            that._resolution = { type: 'commit', commit: target };
            return that._resolution;
        }

        branches = Object.keys(branches);
        tags = Object.keys(tags);

        err = createError('Tag/branch ' + target + ' does not exist', 'ENORESTARGET');
        err.details = !tags.length ?
                'No tags found in ' + that._source :
                'Available tags: ' + tags.join(', ');
        err.details += '\n';
        err.details += !branches.length ?
                'No branches found in ' + that._source :
                'Available branches: ' + branches.join(', ');

        throw err;
    });
};

GitResolver.prototype._cleanup = function () {
    var gitFolder = path.join(this._tempDir, '.git');

    // Remove the .git folder
    // Note that on windows, we need to chmod to 0777 before due to a bug in git
    // See: https://github.com/isaacs/rimraf/issues/19
    if (process.platform === 'win32') {
        return Q.nfcall(chmodr, gitFolder, 0777)
        .then(function () {
            return Q.nfcall(rimraf, gitFolder);
        }, function (err) {
            // If .git does not exist, chmodr returns ENOENT
            // so, we ignore that error code
            if (err.code !== 'ENOENT') {
                throw err;
            }
        });
    } else {
        return Q.nfcall(rimraf, gitFolder);
    }
};

GitResolver.prototype._savePkgMeta = function (meta) {
    var version;

    if (this._resolution.type === 'version') {
        version = semver.clean(this._resolution.tag);

        // Warn if the package meta version is different than the resolved one
        if (typeof meta.version === 'string' && semver.neq(meta.version, version)) {
            this._logger.warn('mismatch', 'Version declared in the json (' + meta.version + ') is different than the resolved one (' + version + ')', {
                resolution: this._resolution,
                pkgMeta: meta
            });
        }

        // Ensure package meta version is the same as the resolution
        meta.version = version;
    } else {
        // If resolved to a target that is not a version,
        // remove the version from the meta
        delete meta.version;
    }

    // Save version/tag/commit in the release
    // Note that we can't store branches because _release is supposed to be
    // an unique id of this ref.
    meta._release = version ||
                    this._resolution.tag ||
                    this._resolution.commit.substr(0, 10);

    // Save resolution to be used in hasNew later
    meta._resolution = this._resolution;

    return Resolver.prototype._savePkgMeta.call(this, meta);
};

// ------------------------------

GitResolver.versions = function (source, extra) {
    var value = this._cache.versions.get(source);

    if (value) {
        return Q.resolve(value)
        .then(function () {
            var versions = this._cache.versions.get(source);

            // If no extra information was requested,
            // resolve simply with the versions
            if (!extra) {
                versions = versions.map(function (version) {
                    return version.version;
                });
            }

            return versions;
        }.bind(this));
    }

    value = this.tags(source)
    .then(function (tags) {
        var tag;
        var version;
        var versions = [];

        // For each tag
        for (tag in tags) {
            version = semver.clean(tag);
            if (version) {
                versions.push({ version: version, tag: tag, commit: tags[tag] });
            }
        }

        // Sort them by DESC order
        versions.sort(function (a, b) {
            return semver.rcompare(a.version, b.version);
        });

        this._cache.versions.set(source, versions);

        // Call the function again to keep it DRY
        return this.versions(source, extra);
    }.bind(this));


    // Store the promise to be reused until it resolves
    // to a specific value
    this._cache.versions.set(source, value);

    return value;
};

GitResolver.tags = function (source) {
    var value = this._cache.tags.get(source);

    if (value) {
        return Q.resolve(value);
    }

    value = this.refs(source)
    .then(function (refs) {
        var tags = {};

        // For each line in the refs, match only the tags
        refs.forEach(function (line) {
            var match = line.match(/^([a-f0-9]{40})\s+refs\/tags\/(\S+)/);

            if (match && !mout.string.endsWith(match[2], '^{}')) {
                tags[match[2]] = match[1];
            }
        });

        this._cache.tags.set(source, tags);

        return tags;
    }.bind(this));

    // Store the promise to be reused until it resolves
    // to a specific value
    this._cache.tags.set(source, value);

    return value;
};

GitResolver.branches = function (source) {
    var value = this._cache.branches.get(source);

    if (value) {
        return Q.resolve(value);
    }

    value = this.refs(source)
    .then(function (refs) {
        var branches = {};

        // For each line in the refs, extract only the heads
        // Organize them in an object where keys are branches and values
        // the commit hashes
        refs.forEach(function (line) {
            var match = line.match(/^([a-f0-9]{40})\s+refs\/heads\/(\S+)/);

            if (match) {
                branches[match[2]] = match[1];
            }
        });

        this._cache.branches.set(source, branches);

        return branches;
    }.bind(this));

    // Store the promise to be reused until it resolves
    // to a specific value
    this._cache.branches.set(source, value);

    return value;
};

GitResolver.clearRuntimeCache = function () {
    // Reset cache for branches, tags, etc
    mout.object.forOwn(GitResolver._cache, function (lru) {
        lru.reset();
    });
};

GitResolver._cache = {
    branches: new LRU({ max: 50, maxAge: 5 * 60 * 1000 }),
    tags: new LRU({ max: 50, maxAge: 5 * 60 * 1000 }),
    versions: new LRU({ max: 50, maxAge: 5 * 60 * 1000 }),
    refs: new LRU({ max: 50, maxAge: 5 * 60 * 1000 })
};

module.exports = GitResolver;
