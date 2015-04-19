var mout = require('mout');
var Q = require('q');
var RegistryClient = require('bower-registry-client');
var ResolveCache = require('./ResolveCache');
var resolverFactory = require('./resolverFactory');
var createError = require('../util/createError');

function PackageRepository(config, logger) {
    var registryOptions;

    this._config = config;
    this._logger = logger;

    // Instantiate the registry
    registryOptions = mout.object.deepMixIn({}, this._config);
    registryOptions.cache = this._config.storage.registry;
    this._registryClient = new RegistryClient(registryOptions, logger);

    // Instantiate the resolve cache
    this._resolveCache = new ResolveCache(this._config);
}

// -----------------

PackageRepository.prototype.fetch = function (decEndpoint) {
    var logger;
    var that = this;
    var isTargetable;
    var info = {
        decEndpoint: decEndpoint
    };

    // Create a new logger that pipes everything to ours that will be
    // used to fetch
    logger = this._logger.geminate();
    // Intercept all logs, adding additional information
    logger.intercept(function (log) {
        that._extendLog(log, info);
    });

    // Get the appropriate resolver
    return resolverFactory(decEndpoint, this._config, logger, this._registryClient)
    // Decide if we retrieve from the cache or not
    // Also decide if we validate the cached entry or not
    .then(function (resolver) {
        info.resolver = resolver;
        isTargetable = resolver.constructor.isTargetable;

        if (!resolver.isCacheable()) {
            return that._resolve(resolver, logger);
        }

        // If force flag is used, bypass cache, but write to cache anyway
        if (that._config.force) {
            logger.action('resolve', resolver.getSource() + '#' + resolver.getTarget());
            return that._resolve(resolver, logger);
        }

        // Note that we use the resolver methods to query the
        // cache because transformations/normalisations can occur
        return that._resolveCache.retrieve(resolver.getSource(), resolver.getTarget())
        // Decide if we can use the one from the resolve cache
        .spread(function (canonicalDir, pkgMeta) {
            // If there's no package in the cache
            if (!canonicalDir) {
                // And the offline flag is passed, error out
                if (that._config.offline) {
                    throw createError('No cached version for ' + resolver.getSource() + '#' + resolver.getTarget(), 'ENOCACHE', {
                        resolver: resolver
                    });
                }

                // Otherwise, we have to resolve it
                logger.info('not-cached', resolver.getSource() + (resolver.getTarget() ? '#' + resolver.getTarget() : ''));
                logger.action('resolve', resolver.getSource() + '#' + resolver.getTarget());

                return that._resolve(resolver, logger);
            }

            info.canonicalDir = canonicalDir;
            info.pkgMeta = pkgMeta;

            logger.info('cached', resolver.getSource() + (pkgMeta._release ? '#' + pkgMeta._release : ''));

            // If offline flag is used, use directly the cached one
            if (that._config.offline) {
                return [canonicalDir, pkgMeta, isTargetable];
            }

            // Otherwise check for new contents
            logger.action('validate', (pkgMeta._release ? pkgMeta._release + ' against ': '') +
                                      resolver.getSource() + (resolver.getTarget() ? '#' + resolver.getTarget() : ''));

            return resolver.hasNew(canonicalDir, pkgMeta)
            .then(function (hasNew) {
                // If there are no new contents, resolve to
                // the cached one
                if (!hasNew) {
                    return [canonicalDir, pkgMeta, isTargetable];
                }

                // Otherwise resolve to the newest one
                logger.info('new', 'version for ' + resolver.getSource() + '#' + resolver.getTarget());
                logger.action('resolve', resolver.getSource() + '#' + resolver.getTarget());

                return that._resolve(resolver, logger);
            });
        });
    })
    // If something went wrong, also extend the error
    .fail(function (err) {
        that._extendLog(err, info);
        throw err;
    });
};

PackageRepository.prototype.versions = function (source) {
    // Resolve the source using the factory because the
    // source can actually be a registry name
    return resolverFactory.getConstructor(source, this._config, this._registryClient)
    .spread(function (ConcreteResolver, source) {
        // If offline, resolve using the cached versions
        if (this._config.offline) {
            return this._resolveCache.versions(source);
        }

        // Otherwise, fetch remotely
        return ConcreteResolver.versions(source);
    }.bind(this));
};

PackageRepository.prototype.eliminate = function (pkgMeta) {
    return Q.all([
        this._resolveCache.eliminate(pkgMeta),
        Q.nfcall(this._registryClient.clearCache.bind(this._registryClient), pkgMeta.name)
    ]);
};

PackageRepository.prototype.clear = function () {
    return Q.all([
        this._resolveCache.clear(),
        Q.nfcall(this._registryClient.clearCache.bind(this._registryClient))
    ]);
};

PackageRepository.prototype.reset = function () {
    this._resolveCache.reset();
    this._registryClient.resetCache();
};

PackageRepository.prototype.list = function () {
    return this._resolveCache.list();
};

PackageRepository.prototype.getRegistryClient = function () {
    return this._registryClient;
};

PackageRepository.prototype.getResolveCache = function () {
    return this._resolveCache;
};

PackageRepository.clearRuntimeCache = function () {
    ResolveCache.clearRuntimeCache();
    RegistryClient.clearRuntimeCache();
    resolverFactory.clearRuntimeCache();
};

// ---------------------

PackageRepository.prototype._resolve = function (resolver, logger) {
    var that = this;

    // Resolve the resolver
    return resolver.resolve()
    // Store in the cache
    .then(function (canonicalDir) {
        if (!resolver.isCacheable()) {
            return canonicalDir;
        }

        return that._resolveCache.store(canonicalDir, resolver.getPkgMeta());
    })
    // Resolve promise with canonical dir and package meta
    .then(function (dir) {
        var pkgMeta = resolver.getPkgMeta();

        logger.info('resolved', resolver.getSource() + (pkgMeta._release ? '#' + pkgMeta._release : ''));
        return [dir, pkgMeta, resolver.constructor.isTargetable()];
    });
};

PackageRepository.prototype._extendLog = function (log, info) {
    log.data = log.data || {};

    // Store endpoint info in each log
    if (info.decEndpoint) {
        log.data.endpoint = mout.object.pick(info.decEndpoint, ['name', 'source', 'target']);
    }

    // Store the resolver info in each log
    if (info.resolver) {
        log.data.resolver = {
            name: info.resolver.getName(),
            source: info.resolver.getSource(),
            target: info.resolver.getTarget()
        };
    }

    // Store the canonical dir and its meta in each log
    if (info.canonicalDir) {
        log.data.canonicalDir = info.canonicalDir;
        log.data.pkgMeta = info.pkgMeta;
    }

    return log;
};

module.exports = PackageRepository;
