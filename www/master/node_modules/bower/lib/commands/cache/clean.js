var fs = require('graceful-fs');
var path = require('path');
var mout = require('mout');
var Q = require('q');
var rimraf = require('rimraf');
var endpointParser = require('bower-endpoint-parser');
var PackageRepository = require('../../core/PackageRepository');
var semver = require('../../util/semver');
var defaultConfig = require('../../config');

function clean(logger, endpoints, options, config) {
    var decEndpoints;
    var names;

    options = options || {};
    config = defaultConfig(config);

    // If endpoints is an empty array, null them
    if (endpoints && !endpoints.length) {
        endpoints = null;
    }

    // Generate decomposed endpoints and names based on the endpoints
    if (endpoints) {
        decEndpoints = endpoints.map(function (endpoint) {
            return endpointParser.decompose(endpoint);
        });
        names = decEndpoints.map(function (decEndpoint) {
            return decEndpoint.name || decEndpoint.source;
        });
    }

    return Q.all([
        clearPackages(decEndpoints, config, logger),
        clearLinks(names, config, logger)
    ])
    .spread(function (entries) {
        return entries;
    });
}

function clearPackages(decEndpoints, config, logger) {
    var repository =  new PackageRepository(config, logger);

    return repository.list()
    .then(function (entries) {
        var promises;

        // Filter entries according to the specified packages
        if (decEndpoints) {
            entries = entries.filter(function (entry) {
                return !!mout.array.find(decEndpoints, function (decEndpoint) {
                    var entryPkgMeta = entry.pkgMeta;

                    // Check if name or source match the entry
                    if  (decEndpoint.name !== entryPkgMeta.name &&
                        decEndpoint.source !== entryPkgMeta.name &&
                        decEndpoint.source !== entryPkgMeta._source
                    ) {
                        return false;
                    }

                    // If target is a wildcard, simply return true
                    if (decEndpoint.target === '*') {
                        return true;
                    }

                    // If it's a semver target, compare using semver spec
                    if (semver.validRange(decEndpoint.target)) {
                        return semver.satisfies(entryPkgMeta.version, decEndpoint.target);
                    }

                    // Otherwise, compare against target/release
                    return decEndpoint.target === entryPkgMeta._target ||
                           decEndpoint.target === entryPkgMeta._release;
                });
            });
        }

        promises = entries.map(function (entry) {
            return repository.eliminate(entry.pkgMeta)
            .then(function () {
                logger.info('deleted', 'Cached package ' + entry.pkgMeta.name + ': ' + entry.canonicalDir, {
                    file: entry.canonicalDir
                });
            });
        });

        return Q.all(promises)
        .then(function () {
            if (!decEndpoints) {
                // Ensure that everything is cleaned,
                // even invalid packages in the cache
                return repository.clear();
            }
        })
        .then(function () {
            return entries;
        });
    });
}

function clearLinks(names, config, logger) {
    var promise;
    var dir = config.storage.links;

    // If no names are passed, grab all links
    if (!names) {
        promise = Q.nfcall(fs.readdir, dir)
        .fail(function (err) {
            if (err.code === 'ENOENT') {
                return [];
            }

            throw err;
        });
    // Otherwise use passed ones
    } else {
        promise = Q.resolve(names);
    }

    return promise
    .then(function (names) {
        var promises;
        var linksToRemove = [];

        // Decide which links to delete
        promises = names.map(function (name) {
            var link = path.join(config.storage.links, name);

            return Q.nfcall(fs.readlink, link)
            .then(function (linkTarget) {
                // Link exists, check if it points to a folder
                // that still exists
                return Q.nfcall(fs.stat, linkTarget)
                .then(function (stat) {
                    // Target is not a folder..
                    if (!stat.isDirectory()) {
                        linksToRemove.push(link);
                    }
                })
                // Error occurred reading the link
                .fail(function () {
                    linksToRemove.push(link);
                });
            // Ignore if link does not exist
            }, function (err) {
                if (err.code !== 'ENOENT') {
                    linksToRemove.push(link);
                }
            });
        });

        return Q.all(promises)
        .then(function () {
            var promises;

            // Remove each link that was declared as invalid
            promises = linksToRemove.map(function (link) {
                return Q.nfcall(rimraf, link)
                .then(function () {
                    logger.info('deleted', 'Invalid link: ' + link, {
                        file: link
                    });
                });
            });

            return Q.all(promises);
        });
    });
}

// -------------------

clean.readOptions = function (argv) {
    var cli = require('../../util/cli');
    var options = cli.readOptions(argv);
    var endpoints = options.argv.remain.slice(2);

    delete options.argv;

    return [endpoints, options];
};

module.exports = clean;
