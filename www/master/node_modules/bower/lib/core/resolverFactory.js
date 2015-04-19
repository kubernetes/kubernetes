var Q = require('q');
var fs = require('graceful-fs');
var path = require('path');
var mout = require('mout');
var resolvers = require('./resolvers');
var createError = require('../util/createError');

function createInstance(decEndpoint, config, logger, registryClient) {
    return getConstructor(decEndpoint.source, config, registryClient)
    .spread(function (ConcreteResolver, source, fromRegistry) {
        var decEndpointCopy = mout.object.pick(decEndpoint, ['name', 'target']);

        decEndpointCopy.source = source;

        // Signal if it was fetched from the registry
        if (fromRegistry) {
            decEndpoint.registry = true;
            // If no name was specified, assume the name from the registry
            if (!decEndpointCopy.name) {
                decEndpointCopy.name = decEndpoint.name = decEndpoint.source;
            }
        }

        return new ConcreteResolver(decEndpointCopy, config, logger);
    });
}

function getConstructor(source, config, registryClient) {
    var absolutePath,
        promise;

    // Git case: git git+ssh, git+http, git+https
    //           .git at the end (probably ssh shorthand)
    //           git@ at the start
    if (/^git(\+(ssh|https?))?:\/\//i.test(source) || /\.git\/?$/i.test(source) || /^git@/i.test(source)) {
        source = source.replace(/^git\+/, '');
        return Q.fcall(function () {

            // If it's a GitHub repository, return the specialized resolver
            if (resolvers.GitHub.getOrgRepoPair(source)) {
                return [resolvers.GitHub, source];
            }

            return [resolvers.GitRemote, source];
        });
    }

    // SVN case: svn, svn+ssh, svn+http, svn+https, svn+file
    if (/^svn(\+(ssh|https?|file))?:\/\//i.test(source)) {
        return Q.fcall(function () {
            return [resolvers.Svn, source];
        });
    }

    // URL case
    if (/^https?:\/\//i.exec(source)) {
        return Q.fcall(function () {
            return [resolvers.Url, source];
        });
    }

    // Below we try a series of async tests to guess the type of resolver to use
    // If a step was unable to guess the resolver, it throws an error
    // If a step was able to guess the resolver, it resolves with a function
    // That function returns a promise that will resolve with the concrete type

    // If source is ./ or ../ or an absolute path
    absolutePath = path.resolve(config.cwd, source);

    if (/^\.\.?[\/\\]/.test(source) || /^~\//.test(source) || path.normalize(source).replace(/[\/\\]+$/, '') === absolutePath) {
        promise = Q.nfcall(fs.stat, path.join(absolutePath, '.git'))
        .then(function (stats) {
            if (stats.isDirectory()) {
                return function () {
                    return Q.resolve([resolvers.GitFs, absolutePath]);
                };
            }

            throw new Error('Not a Git repository');
        })
        // If not, check if source is a valid Subversion repository
        .fail(function () {
            return Q.nfcall(fs.stat, path.join(absolutePath, '.svn'))
            .then(function (stats) {
                if (stats.isDirectory()) {
                    return function () {
                        return Q.resolve([resolvers.Svn, absolutePath]);
                    };
                }

                throw new Error('Not a Subversion repository');
            });
        })
        // If not, check if source is a valid file/folder
        .fail(function () {
            return Q.nfcall(fs.stat, absolutePath)
            .then(function () {
                return function () {
                    return Q.resolve([resolvers.Fs, absolutePath]);
                };
            });
        });
    } else {
        promise = Q.reject(new Error('Not an absolute or relative file'));
    }

    return promise
    // Check if is a shorthand and expand it
    .fail(function (err) {
        var parts;

        // Skip ssh and/or URL with auth
        if (/[:@]/.test(source)) {
            throw err;
        }

        // Ensure exactly only one "/"
        parts = source.split('/');
        if (parts.length === 2) {
            source = mout.string.interpolate(config.shorthandResolver, {
                shorthand: source,
                owner: parts[0],
                package: parts[1]
            });

            return function () {
                return getConstructor(source, config, registryClient);
            };
        }

        throw err;
    })
    // As last resort, we try the registry
    .fail(function (err) {
        if (!registryClient) {
            throw err;
        }

        return function () {
            return Q.nfcall(registryClient.lookup.bind(registryClient), source)
            .then(function (entry) {
                if (!entry) {
                    throw createError('Package ' + source + ' not found', 'ENOTFOUND');
                }

                // TODO: Handle entry.type.. for now it's only 'alias'
                //       When we got published packages, this needs to be adjusted
                source = entry.url;

                return getConstructor(source, config, registryClient)
                .spread(function (ConcreteResolver, source) {
                    return [ConcreteResolver, source, true];
                });
            });
        };
    })
    // If we got the function, simply call and return
    .then(function (func) {
        return func();
    // Finally throw a meaningful error
    }, function () {
        throw createError('Could not find appropriate resolver for ' + source, 'ENORESOLVER');
    });
}

function clearRuntimeCache() {
    mout.object.values(resolvers).forEach(function (ConcreteResolver) {
        ConcreteResolver.clearRuntimeCache();
    });
}

module.exports = createInstance;
module.exports.getConstructor = getConstructor;
module.exports.clearRuntimeCache = clearRuntimeCache;
