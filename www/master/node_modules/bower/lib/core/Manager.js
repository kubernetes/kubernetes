var Q = require('q');
var mout = require('mout');
var path = require('path');
var mkdirp = require('mkdirp');
var rimraf = require('rimraf');
var fs = require('graceful-fs');
var endpointParser = require('bower-endpoint-parser');
var PackageRepository = require('./PackageRepository');
var semver = require('../util/semver');
var copy = require('../util/copy');
var createError = require('../util/createError');
var scripts = require('./scripts');

function Manager(config, logger) {
    this._config = config;
    this._logger = logger;
    this._repository = new PackageRepository(this._config, this._logger);

    this.configure({});
}

// -----------------

Manager.prototype.configure = function (setup) {
    var targetsHash = {};

    this._conflicted = {};

    // Targets
    this._targets = setup.targets || [];
    this._targets.forEach(function (decEndpoint) {
        decEndpoint.initialName = decEndpoint.name;
        decEndpoint.dependants = mout.object.values(decEndpoint.dependants);
        targetsHash[decEndpoint.name] = true;

        // If the endpoint is marked as newly, make it unresolvable
        decEndpoint.unresolvable = !!decEndpoint.newly;
    });

    // Resolved & installed
    this._resolved = {};
    this._installed = {};
    mout.object.forOwn(setup.resolved, function (decEndpoint, name) {
        decEndpoint.dependants = mout.object.values(decEndpoint.dependants);
        this._resolved[name] = [decEndpoint];
        this._installed[name] = decEndpoint.pkgMeta;
    }, this);

    // Installed
    mout.object.mixIn(this._installed, setup.installed);

    // Incompatibles
    this._incompatibles = {};
    setup.incompatibles = this._uniquify(setup.incompatibles || []);
    setup.incompatibles.forEach(function (decEndpoint) {
        var name = decEndpoint.name;

        this._incompatibles[name] = this._incompatibles[name] || [];
        this._incompatibles[name].push(decEndpoint);
        decEndpoint.dependants = mout.object.values(decEndpoint.dependants);

        // Mark as conflicted so that the resolution is not removed
        this._conflicted[name] = true;

        // If not a target/resolved, add as target
        if (!targetsHash[name] && !this._resolved[name]) {
            this._targets.push(decEndpoint);
        }
    }, this);

    // Resolutions
    this._resolutions = setup.resolutions || {};

    // Uniquify targets
    this._targets = this._uniquify(this._targets);

    // Force-latest
    this._forceLatest = !!setup.forceLatest;

    return this;
};

Manager.prototype.resolve = function () {
    // If already resolving, error out
    if (this._working) {
        return Q.reject(createError('Already working', 'EWORKING'));
    }

    // Reset stuff
    this._fetching = {};
    this._nrFetching = 0;
    this._failed = {};
    this._hasFailed = false;
    this._deferred = Q.defer();

    // If there's nothing to resolve, simply dissect
    if (!this._targets.length) {
        process.nextTick(this._dissect.bind(this));
    // Otherwise, fetch each target from the repository
    // and let the process roll out
    } else {
        this._targets.forEach(this._fetch.bind(this));
    }

    // Unset working flag when done
    return this._deferred.promise
    .fin(function () {
        this._working = false;
    }.bind(this));
};

Manager.prototype.preinstall = function (json) {
    var that = this;
    var componentsDir = path.join(this._config.cwd, this._config.directory);

    // If nothing to install, skip the code bellow
    if (mout.lang.isEmpty(that._dissected)) {
        return Q.resolve({});
    }

    return Q.nfcall(mkdirp, componentsDir)
    .then(function () {
        return scripts.preinstall(
            that._config, that._logger, that._dissected, that._installed, json
        );
    });
};

Manager.prototype.postinstall = function (json) {
    var that = this;
    var componentsDir = path.join(this._config.cwd, this._config.directory);

    // If nothing to install, skip the code bellow
    if (mout.lang.isEmpty(that._dissected)) {
        return Q.resolve({});
    }

    return Q.nfcall(mkdirp, componentsDir)
    .then(function () {
        return scripts.postinstall(
            that._config, that._logger, that._dissected, that._installed, json
        );
    });
};

Manager.prototype.install = function (json) {
    var componentsDir;
    var that = this;

    // If already resolving, error out
    if (this._working) {
        return Q.reject(createError('Already working', 'EWORKING'));
    }

    // If nothing to install, skip the code bellow
    if (mout.lang.isEmpty(that._dissected)) {
        return Q.resolve({});
    }

    componentsDir = path.join(this._config.cwd, this._config.directory);
    return Q.nfcall(mkdirp, componentsDir)
    .then(function () {
        var promises = [];

        mout.object.forOwn(that._dissected, function (decEndpoint, name) {
            var promise;
            var dst;
            var release = decEndpoint.pkgMeta._release;

            that._logger.action('install', name + (release ? '#' + release : ''), that.toData(decEndpoint));

            dst = path.join(componentsDir, name);

            // Remove existent and copy canonical dir
            promise = Q.nfcall(rimraf, dst)
            .then(copy.copyDir.bind(copy, decEndpoint.canonicalDir, dst))
            .then(function () {
                var metaFile = path.join(dst, '.bower.json');

                decEndpoint.canonicalDir = dst;

                // Store additional metadata in bower.json
                return Q.nfcall(fs.readFile, metaFile)
                .then(function (contents) {
                    var json = JSON.parse(contents.toString());

                    json._target = decEndpoint.target;
                    json._originalSource = decEndpoint.source;
                    if (decEndpoint.newly) {
                        json._direct = true;
                    }

                    json = JSON.stringify(json, null, '  ');
                    return Q.nfcall(fs.writeFile, metaFile, json);
                });
            });

            promises.push(promise);
        });

        return Q.all(promises);
    })
    .then(function () {
        // Sync up dissected dependencies and dependants
        // See: https://github.com/bower/bower/issues/879
        mout.object.forOwn(that._dissected, function (pkg) {
            // Sync dependencies
            mout.object.forOwn(pkg.dependencies, function (dependency, name) {
                var dissected = this._dissected[name] || (this._resolved[name] ? this._resolved[name][0] : dependency);
                pkg.dependencies[name] = dissected;
            }, this);

            // Sync dependants
            pkg.dependants = pkg.dependants.map(function (dependant) {
                var name = dependant.name;
                var dissected = this._dissected[name] || (this._resolved[name] ? this._resolved[name][0] : dependant);

                return dissected;
            }, this);
        }, that);

        // Resolve with meaningful data
        return mout.object.map(that._dissected, function (decEndpoint) {
            return this.toData(decEndpoint);
        }, that);
    })
    .fin(function () {
        this._working = false;
    }.bind(this));
};

Manager.prototype.toData = function (decEndpoint, extraKeys, upperDeps) {
    var names;
    var extra;

    var data = {};

    upperDeps = upperDeps || [];
    data.endpoint = mout.object.pick(decEndpoint, ['name', 'source', 'target']);

    if (decEndpoint.canonicalDir) {
        data.canonicalDir = decEndpoint.canonicalDir;
        data.pkgMeta = decEndpoint.pkgMeta;
    }

    if (extraKeys) {
        extra = mout.object.pick(decEndpoint, extraKeys);
        extra = mout.object.filter(extra, function (value) {
            return !!value;
        });
        mout.object.mixIn(data, extra);
    }

    if (decEndpoint.dependencies) {
        data.dependencies = {};

        // Call recursively for each dependency but ordered
        // by dependency names
        names = Object.keys(decEndpoint.dependencies).sort();
        names.forEach(function (name) {

            // Prevent from infinite recursion when installing cyclic
            // dependencies
            if (!mout.array.contains(upperDeps, name)) {
                data.dependencies[name] = this.toData(decEndpoint.dependencies[name],
                    extraKeys,
                    upperDeps.concat(decEndpoint.name));
            }
        }, this);
    }

    data.nrDependants = mout.object.size(decEndpoint.dependants);

    return data;
};

Manager.prototype.getPackageRepository = function () {
    return this._repository;
};

// -----------------

Manager.prototype._fetch = function (decEndpoint) {
    var name = decEndpoint.name;

    // Check if the whole process started to fail fast
    if (this._hasFailed) {
        return;
    }

    // Mark as being fetched
    this._fetching[name] = this._fetching[name] || [];
    this._fetching[name].push(decEndpoint);
    this._nrFetching++;

    // Fetch it from the repository
    // Note that the promise is stored in the decomposed endpoint
    // because it might be reused if a similar endpoint needs to be resolved
    return decEndpoint.promise = this._repository.fetch(decEndpoint)
    // When done, call onFetchSuccess
    .spread(this._onFetchSuccess.bind(this, decEndpoint))
    // If it fails, call onFetchFailure
    .fail(this._onFetchError.bind(this, decEndpoint));
};

Manager.prototype._onFetchSuccess = function (decEndpoint, canonicalDir, pkgMeta, isTargetable) {
    var name;
    var resolved;
    var index;
    var incompatibles;
    var initialName = decEndpoint.initialName != null ? decEndpoint.initialName : decEndpoint.name;
    var fetching = this._fetching[initialName];

    // Remove from being fetched list
    mout.array.remove(fetching, decEndpoint);
    this._nrFetching--;

    // Store some needed stuff
    decEndpoint.name = name = decEndpoint.name || pkgMeta.name;
    decEndpoint.canonicalDir = canonicalDir;
    decEndpoint.pkgMeta = pkgMeta;
    delete decEndpoint.promise;

    // Add to the resolved list
    // If there's an exact equal endpoint, replace instead of adding
    // This can happen because the name might not be known from the start
    resolved = this._resolved[name] = this._resolved[name] || [];
    index = mout.array.findIndex(resolved, function (resolved) {
        return resolved.target === decEndpoint.target;
    });
    if (index !== -1) {
        // Merge dependants
        decEndpoint.dependants.push.apply(decEndpoint.dependants, resolved[index.dependants]);
        decEndpoint.dependants = this._uniquify(decEndpoint.dependants);
        resolved.splice(index, 1);
    }
    resolved.push(decEndpoint);

    // Parse dependencies
    this._parseDependencies(decEndpoint, pkgMeta, 'dependencies');

    // Check if there are incompatibilities for this package name
    // If there are, we need to fetch them
    incompatibles = this._incompatibles[name];
    if (incompatibles) {
        // Filter already resolved
        incompatibles = incompatibles.filter(function (incompatible) {
            return !resolved.some(function (decEndpoint) {
                return incompatible.target === decEndpoint.target;
            });
        }, this);
        // Filter being resolved
        incompatibles = incompatibles.filter(function (incompatible) {
            return !fetching.some(function (decEndpoint) {
                return incompatible.target === decEndpoint.target;
            });
        }, this);

        incompatibles.forEach(this._fetch.bind(this));
        delete this._incompatibles[name];
    }

    // If the package is not targetable, flag it
    // It will be needed later so that untargetable endpoints
    // will not get * converted to ~version
    if (!isTargetable) {
        decEndpoint.untargetable = true;
    }

    // If there are no more packages being fetched,
    // finish the resolve process by dissecting all resolved packages
    if (this._nrFetching <= 0) {
        process.nextTick(this._dissect.bind(this));
    }
};

Manager.prototype._onFetchError = function (decEndpoint, err) {
    var name = decEndpoint.name;

    err.data = err.data || {};
    err.data.endpoint = mout.object.pick(decEndpoint, ['name', 'source', 'target']);

    // Remove from being fetched list
    mout.array.remove(this._fetching[name], decEndpoint);
    this._nrFetching--;

    // Add to the failed list
    this._failed[name] = this._failed[name] || [];
    this._failed[name].push(err);
    delete decEndpoint.promise;

    // Make the whole process to fail fast
    this._failFast();

    // If there are no more packages being fetched,
    // finish the resolve process (with an error)
    if (this._nrFetching <= 0) {
        process.nextTick(this._dissect.bind(this));
    }
};

Manager.prototype._failFast = function () {
    if (this._hasFailed) {
        return;
    }

    this._hasFailed = true;

    // If after some amount of time all pending tasks haven't finished,
    // we force the process to end
    this._failFastTimeout = setTimeout(function () {
        this._nrFetching = Infinity;
        this._dissect();
    }.bind(this), 20000);
};

Manager.prototype._parseDependencies = function (decEndpoint, pkgMeta, jsonKey) {
    var pending = [];

    decEndpoint.dependencies = decEndpoint.dependencies || {};

    // Parse package dependencies
    mout.object.forOwn(pkgMeta[jsonKey], function (value, key) {
        var resolved;
        var fetching;
        var compatible;
        var childDecEndpoint = endpointParser.json2decomposed(key, value);

        // Check if a compatible one is already resolved
        // If there's one, we don't need to resolve it twice
        resolved = this._resolved[key];
        if (resolved) {
            // Find if there's one with the exact same target
            compatible = mout.array.find(resolved, function (resolved) {
                return childDecEndpoint.target === resolved.target;
            }, this);

            // If we found one, merge stuff instead of adding as resolved
            if (compatible) {
                decEndpoint.dependencies[key] = compatible;
                compatible.dependants.push(decEndpoint);
                compatible.dependants = this._uniquify(compatible.dependants);

                return;
            }

            // Find one that is compatible
            compatible = mout.array.find(resolved, function (resolved) {
                return this._areCompatible(childDecEndpoint, resolved);
            }, this);

            // If we found one, add as resolved
            // and copy resolved properties from the compatible one
            if (compatible) {
                decEndpoint.dependencies[key] = compatible;
                childDecEndpoint.canonicalDir = compatible.canonicalDir;
                childDecEndpoint.pkgMeta = compatible.pkgMeta;
                childDecEndpoint.dependencies = compatible.dependencies;
                childDecEndpoint.dependants = [decEndpoint];
                this._resolved[key].push(childDecEndpoint);

                return;
            }
        }

        // Check if a compatible one is being fetched
        // If there's one, we wait and reuse it to avoid resolving it twice
        fetching = this._fetching[key];
        if (fetching) {
            compatible = mout.array.find(fetching, function (fetching) {
                return this._areCompatible(childDecEndpoint, fetching);
            }, this);

            if (compatible) {
                pending.push(compatible.promise);
                return;
            }
        }

        // Mark endpoint as unresolvable if the parent is also unresolvable
        childDecEndpoint.unresolvable = !!decEndpoint.unresolvable;

        // Otherwise, just fetch it from the repository
        decEndpoint.dependencies[key] = childDecEndpoint;
        childDecEndpoint.dependants = [decEndpoint];
        this._fetch(childDecEndpoint);
    }, this);

    if (pending.length > 0) {
        Q.all(pending)
        .then(function () {
            this._parseDependencies(decEndpoint, pkgMeta, jsonKey);
        }.bind(this));
    }
};

Manager.prototype._dissect = function () {
    var err;
    var componentsDir;
    var promise = Q.resolve();
    var suitables = {};
    var that = this;

    // If something failed, reject the whole resolve promise
    // with the first error
    if (this._hasFailed) {
        clearTimeout(this._failFastTimeout); // Cancel fail fast timeout

        err = mout.object.values(this._failed)[0][0];
        this._deferred.reject(err);
        return;
    }

    // Find a suitable version for each package name
    mout.object.forOwn(this._resolved, function (decEndpoints, name) {
        var semvers;
        var nonSemvers;

        // Filter out non-semver ones
        semvers = decEndpoints.filter(function (decEndpoint) {
            return !!decEndpoint.pkgMeta.version;
        });

        // Sort semver ones DESC
        semvers.sort(function (first, second) {
            var result = semver.rcompare(first.pkgMeta.version, second.pkgMeta.version);

            // If they are equal and one of them is a wildcard target,
            // give lower priority
            if (!result) {
                if (first.target === '*') {
                    return 1;
                }
                if (second.target === '*') {
                    return -1;
                }
            }

            return result;
        });

        // Convert wildcard targets to semver range targets if they are newly
        // Note that this can only be made if they can be targetable
        // If they are not, the resolver is incapable of handling targets
        semvers.forEach(function (decEndpoint) {
            if (decEndpoint.newly && decEndpoint.target === '*' && !decEndpoint.untargetable) {
                decEndpoint.target = '~' + decEndpoint.pkgMeta.version;
                decEndpoint.originalTarget = '*';
            }
        });

        // Filter non-semver ones
        nonSemvers = decEndpoints.filter(function (decEndpoint) {
            return !decEndpoint.pkgMeta.version;
        });

        promise = promise.then(function () {
            return that._electSuitable(name, semvers, nonSemvers)
            .then(function (suitable) {
                suitables[name] = suitable;
            });
        });
    }, this);

    // After a suitable version has been elected for every package
    promise
    .then(function () {
        // Look for extraneous resolutions
        mout.object.forOwn(this._resolutions, function (resolution, name) {
            if (this._conflicted[name]) {
                return;
            }

            this._logger.warn('extra-resolution', 'Unnecessary resolution: ' + name + '#' + resolution, {
                name: name,
                resolution: resolution,
                action: 'delete'
            });
        }, this);

        // Filter only packages that need to be installed
        componentsDir = path.resolve(that._config.cwd, that._config.directory);
        this._dissected = mout.object.filter(suitables, function (decEndpoint, name) {
            var installedMeta = this._installed[name];
            var dst;

            // Skip linked dependencies
            if (decEndpoint.linked) {
                return false;
            }

            // Skip if source is the same as dest
            dst = path.join(componentsDir, name);
            if (dst === decEndpoint.canonicalDir) {
                return false;
            }

            // Analyse a few props
            if (installedMeta &&
                installedMeta._target === decEndpoint.target &&
                installedMeta._originalSource === decEndpoint.source &&
                installedMeta._release === decEndpoint.pkgMeta._release
            ) {
                return this._config.force;
            }

            return true;
        }, this);
    }.bind(this))
    .then(this._deferred.resolve, this._deferred.reject);
};

Manager.prototype._electSuitable = function (name, semvers, nonSemvers) {
    var suitable;
    var resolution;
    var unresolvable;
    var dataPicks;
    var save;
    var choices;
    var picks = [];

    // If there are both semver and non-semver, there's no way
    // to figure out the suitable one
    if (semvers.length && nonSemvers.length) {
        picks.push.apply(picks, semvers);
        picks.push.apply(picks, nonSemvers);
    // If there are only non-semver ones, the suitable is elected
    // only if there's one
    } else if (nonSemvers.length) {
        if (nonSemvers.length === 1) {
            return Q.resolve(nonSemvers[0]);
        }

        picks.push.apply(picks, nonSemvers);
    // If there are only semver ones, figure out which one is
    // compatible with every requirement
    } else {
        suitable = mout.array.find(semvers, function (subject) {
            return semvers.every(function (decEndpoint) {
                return subject === decEndpoint ||
                       semver.satisfies(subject.pkgMeta.version, decEndpoint.target);
            });
        });

        if (suitable) {
            return Q.resolve(suitable);
        }

        picks.push.apply(picks, semvers);
    }

    // At this point, there's a conflict
    this._conflicted[name] = true;

    // Prepare data to be sent bellow
    // 1 - Sort picks by version/release
    picks.sort(function (pick1, pick2) {
        var version1 = pick1.pkgMeta.version;
        var version2 = pick2.pkgMeta.version;
        var comp;

        // If both have versions, compare their versions using semver
        if (version1 && version2) {
            comp = semver.compare(version1, version2);
            if (comp) {
                return comp;
            }
        } else {
            // If one of them has a version, it's considered higher
            if (version1) {
                return 1;
            }
            if (version2) {
                return -1;
            }
        }

        // Give priority to the one with most dependants
        if (pick1.dependants.length > pick2.dependants.length) {
            return -1;
        }
        if (pick1.dependants.length < pick2.dependants.length) {
            return 1;
        }

        return 0;
    });

    // 2 - Transform data
    dataPicks = picks.map(function (pick) {
        var dataPick = this.toData(pick);
        dataPick.dependants = pick.dependants.map(this.toData, this);
        dataPick.dependants.sort(function (dependant1, dependant2) {
            return dependant1.endpoint.name.localeCompare(dependant2.endpoint.name);
        });
        return dataPick;
    }, this);

    // Check if there's a resolution that resolves the conflict
    // Note that if one of them is marked as unresolvable,
    // the resolution has no effect
    resolution = this._resolutions[name];
    unresolvable = mout.object.find(picks, function (pick) {
        return pick.unresolvable;
    });

    if (resolution && !unresolvable) {
        suitable = -1;

        // Range resolution
        if (semver.validRange(resolution)) {
            suitable = mout.array.findIndex(picks, function (pick) {
                return pick.pkgMeta.version &&
                       semver.satisfies(pick.pkgMeta.version, resolution);
            });
        }

        // Exact match resolution (e.g. branches/tags)
        if (suitable === -1) {
            suitable = mout.array.findIndex(picks, function (pick) {
                return pick.target === resolution ||
                       pick.pkgMeta._release === resolution;
            });
        }

        if (suitable === -1) {
            this._logger.warn('resolution', 'Unsuitable resolution declared for ' + name + ': ' + resolution, {
                name: name,
                picks: dataPicks,
                resolution: resolution
            });
        } else {
            this._logger.conflict('solved', 'Unable to find suitable version for ' + name, {
                name: name,
                picks: dataPicks,
                resolution: resolution,
                suitable: dataPicks[suitable]
            });
            return Q.resolve(picks[suitable]);
        }
    }

    // If force latest is enabled, resolve to the highest semver version
    // or whatever non-semver if none available
    if (this._forceLatest) {
        suitable = picks.length - 1;

        this._logger.conflict('solved', 'Unable to find suitable version for ' + name, {
            name: name,
            picks: dataPicks,
            suitable: dataPicks[suitable],
            forced: true
        });

        // Save resolution
        this._storeResolution(picks[suitable]);

        return Q.resolve(picks[suitable]);
    }

    // If interactive is disabled, error out
    if (!this._config.interactive) {
        throw createError('Unable to find suitable version for ' + name, 'ECONFLICT', {
            name: name,
            picks: dataPicks
        });
    }

    // At this point the user needs to make a decision
    this._logger.conflict('incompatible', 'Unable to find suitable version for ' + name, {
        name: name,
        picks: dataPicks
    });

    choices = picks.map(function (pick, index) { return index + 1; });
    return Q.nfcall(this._logger.prompt.bind(this._logger), {
        type: 'input',
        message: 'Answer',
        validate: function (choice) {
            choice = Number(mout.string.trim(choice.trim(), '!'));

            if (!choice || choice < 1 || choice > picks.length) {
                return 'Invalid choice';
            }

            return true;
        }
    })
    .then(function (choice) {
        var pick;

        // Sanitize choice
        choice = choice.trim();
        save = /^!/.test(choice) || /!$/.test(choice);  // Save if prefixed or suffixed with !
        choice = Number(mout.string.trim(choice, '!'));
        pick = picks[choice - 1];

        // Save resolution
        if (save) {
            this._storeResolution(pick);
        }

        return pick;
    }.bind(this));
};

Manager.prototype._storeResolution = function (pick) {
    var resolution;
    var name = pick.name;

    if (pick.target === '*') {
        resolution = pick.pkgMeta._release || '*';
    } else {
        resolution = pick.target;
    }

    this._logger.info('resolution', 'Saved ' + name + '#' + resolution + ' as resolution', {
        name: name,
        resolution: resolution,
        action: this._resolutions[name] ? 'edit' : 'add'
    });
    this._resolutions[name] = resolution;
};

/**
 * Checks if some endpoint is compatible with already resolved target.
 *
 * It is used in two situations:
 *   * checks if resolved component matches dependency constraint
 *   * checks if not resolved component matches already fetched component
 *
 * If candidate matches already resolved component, it won't be downloaded.
 *
 * @param {Endpoint} candidate endpoint
 * @param {Endpoint} resolved endpoint
 *
 * @return {Boolean}
 */
Manager.prototype._areCompatible = function (candidate, resolved) {
    var resolvedVersion;
    var highestCandidate;
    var highestResolved;
    var candidateIsRange = semver.validRange(candidate.target);
    var resolvedIsRange = semver.validRange(resolved.target);
    var candidateIsVersion = semver.valid(candidate.target);
    var resolvedIsVersion = semver.valid(resolved.target);

    // Check if targets are equal
    if (candidate.target === resolved.target) {
        return true;
    }

    resolvedVersion = resolved.pkgMeta && resolved.pkgMeta.version;
    // If there is no pkgMeta, resolvedVersion is downloading now
    // Check based on target requirements
    if (!resolvedVersion) {
        // If one of the targets is range and other is version,
        // check version against the range
        if (candidateIsVersion && resolvedIsRange) {
            return semver.satisfies(candidate.target, resolved.target);
        }

        if (resolvedIsVersion && candidateIsRange) {
            return semver.satisfies(resolved.target, candidate.target);
        }

        if (resolvedIsVersion && candidateIsVersion) {
            return semver.eq(resolved.target, candidate.target);
        }

        // If both targets are range, check that both have same
        // higher cap
        if (resolvedIsRange && candidateIsRange) {
            highestCandidate =
                this._getCap(semver.toComparators(candidate.target), 'highest');
            highestResolved =
                this._getCap(semver.toComparators(resolved.target), 'highest');

            // This never happens, but you can't be sure without tests
            if (!highestResolved.version || !highestCandidate.version) {
                return false;
            }

            return semver.eq(highestCandidate.version, highestResolved.version) &&
                highestCandidate.comparator === highestResolved.comparator;
        }
        return false;
    }

    // If target is a version, compare against the resolved version
    if (candidateIsVersion) {
        return semver.eq(candidate.target, resolvedVersion);
    }

    // If target is a range, check if resolved version satisfies it
    if (candidateIsRange) {
        return semver.satisfies(resolvedVersion, candidate.target);
    }

    return false;
};

/**
 * Gets highest/lowest version from set of comparators.
 *
 * The only thing that matters for this function is version number.
 * Returned comparator is splitted to comparator and version parts.
 *
 * It is used to receive lowest / highest bound of toComparators result:
 * semver.toComparators('~0.1.1') // => [ [ '>=0.1.1-0', '<0.2.0-0' ] ]
 *
 * Examples:
 *
 * _getCap([['>=2.1.1-0', '<2.2.0-0'], '<3.2.0'], 'highest')
 * // => { comparator: '<', version: '3.2.0' }
 *
 * _getCap([['>=2.1.1-0', '<2.2.0-0'], '<3.2.0'], 'lowest')
 * // => { comparator: '>=', version: '2.1.1-0' }
 *
 * @param {Array.<Array|string>} comparators
 * @param {string} side, 'highest' (default) or 'lowest'
 *
 * @return {{ comparator: string, version: string }}
 */
Manager.prototype._getCap = function (comparators, side) {
    var matches;
    var candidate;
    var cap = {};
    var compare = side === 'lowest' ? semver.lt : semver.gt;

    comparators.forEach(function (comparator) {
        // Get version of this comparator
        // If it's an array, call recursively
        if (Array.isArray(comparator)) {
            candidate = this._getCap(comparator, side);

            // Compare with the current highest version
            if (!cap.version || compare(candidate.version, cap.version)) {
                cap = candidate;
            }
        // Otherwise extract the version from the comparator
        // using a simple regexp
        } else {
            matches = comparator.match(/(.*?)(\d+\.\d+\.\d+.*)$/);
            if (!matches) {
                return;
            }

            // Compare with the current highest version
            if (!cap.version || compare(matches[2], cap.version)) {
                cap.version = matches[2];
                cap.comparator = matches[1];
            }
        }
    }, this);

    return cap;
};

/**
 * Filters out unique endpoints, comparing by name and then source.
 *
 * It leaves last matching endpoint.
 *
 * Examples:
 *
 *  manager._uniquify([
 *      { name: 'foo', source: 'google.com' },
 *      { name: 'foo', source: 'facebook.com' }
 *  ]);
 *  // => { name: 'foo', source: 'facebook.com' }
 *
 * @param {Array.<Endpoint>} decEndpoints
 * @return {Array.<Endpoint>} Filtered elements of decEndpoints
 *
 */
Manager.prototype._uniquify = function (decEndpoints) {
    var length = decEndpoints.length;

    return decEndpoints.filter(function (decEndpoint, index) {
        var x;
        var current;

        for (x = index + 1; x < length; ++x) {
            current = decEndpoints[x];

            if (current === decEndpoint) {
                return false;
            }

            // Compare name if both set
            // Fallback to compare sources
            if (!current.name && !decEndpoint.name) {
                if (current.source !== decEndpoint.source) {
                    continue;
                }
            } else if (current.name !== decEndpoint.name) {
                continue;
            }

            // Compare targets if name/sources are equal
            if (current.target === decEndpoint.target) {
                return false;
            }
        }

        return true;
    });
};

module.exports = Manager;
