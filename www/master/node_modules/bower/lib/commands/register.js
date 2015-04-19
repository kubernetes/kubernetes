var mout = require('mout');
var Q = require('q');
var chalk = require('chalk');
var PackageRepository = require('../core/PackageRepository');
var Config = require('bower-config');
var Tracker = require('../util/analytics').Tracker;
var createError = require('../util/createError');
var defaultConfig = require('../config');
var GitHubResolver = require('../core/resolvers/GitHubResolver');

function register(logger, name, url, config) {
    var repository;
    var registryClient;
    var tracker;
    var force;

    config = defaultConfig(config);
    force = config.force;
    tracker = new Tracker(config);

    name = (name || '').trim();
    url = (url || '').trim();

    // Bypass any cache
    config.offline = false;
    config.force = true;

    return Q.try(function () {
        // Verify name and url
        if (!name || !url) {
            throw createError('Usage: bower register <name> <url>', 'EINVFORMAT');
        }

        // The public registry only allows git:// endpoints
        // As such, we attempt to convert URLs as necessary
        if (config.registry.register === Config.DEFAULT_REGISTRY) {
            url = convertUrl(url, logger);

            if (!mout.string.startsWith(url, 'git://')) {
                throw createError('The registry only accepts URLs starting with git://', 'EINVFORMAT');
            }
        }

        tracker.track('register');

        // Attempt to resolve the package referenced by the URL to ensure
        // everything is ok before registering
        repository = new PackageRepository(config, logger);
        return repository.fetch({ name: name, source: url, target: '*' });
    })
    .spread(function (canonicalDir, pkgMeta) {
        if (pkgMeta.private) {
            throw createError('The package you are trying to register is marked as private', 'EPRIV');
        }

        // If non interactive or user forced, bypass confirmation
        if (!config.interactive || force) {
            return true;
        }

        // Confirm if the user really wants to register
        return Q.nfcall(logger.prompt.bind(logger), {
            type: 'confirm',
            message: 'Registering a package will make it installable via the registry (' +
                chalk.cyan.underline(config.registry.register) + '), continue?',
            default: true
        });
    })
    .then(function (result) {
        // If user response was negative, abort
        if (!result) {
            return;
        }

        // Register
        registryClient = repository.getRegistryClient();

        logger.action('register', url, {
            name: name,
            url: url
        });

        return Q.nfcall(registryClient.register.bind(registryClient), name, url);
    });
}

function convertUrl(url, logger) {
    var pair;
    var newUrl;

    if (!mout.string.startsWith(url, 'git://')) {
        // Convert GitHub ssh & https to git://
        pair = GitHubResolver.getOrgRepoPair(url);
        if (pair) {
            newUrl = 'git://github.com/' + pair.org + '/' + pair.repo + '.git';
            logger.warn('convert', 'Converted ' + url + ' to ' + newUrl);
        }
    }

    return newUrl || url;
}

// -------------------

register.readOptions = function (argv) {
    var cli = require('../util/cli');

    var options = cli.readOptions(argv);
    var name = options.argv.remain[1];
    var url = options.argv.remain[2];

    return [name, url];
};

module.exports = register;
