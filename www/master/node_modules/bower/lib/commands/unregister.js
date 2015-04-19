var chalk = require('chalk');
var Q = require('q');

var defaultConfig = require('../config');
var PackageRepository = require('../core/PackageRepository');
var Tracker = require('../util/analytics').Tracker;
var createError = require('../util/createError');

function unregister(logger, name, config) {

    if (!name) {
        return;
    }

    var repository;
    var registryClient;
    var tracker;
    var force;

    config = defaultConfig(config);
    force = config.force;
    tracker = new Tracker(config);

    // Bypass any cache
    config.offline = false;
    config.force = true;

    // Trim name
    name = name.trim();

    repository = new PackageRepository(config, logger);

    tracker.track('unregister');

    if (!config.accessToken) {
        return logger.emit('error',
             createError('Use "bower login" with collaborator credentials', 'EFORBIDDEN')
        );
    }

    return Q.resolve()
    .then(function () {
        // If non interactive or user forced, bypass confirmation
        if (!config.interactive || force) {
            return true;
        }

        return Q.nfcall(logger.prompt.bind(logger), {
            type: 'confirm',
            message: 'You are about to remove component "' + chalk.cyan.underline(name) + '" from the bower registry (' + chalk.cyan.underline(config.registry.register) + '). It is generally considered bad behavior to remove versions of a library that others are depending on. Are you really sure?',
            default: false
        });
    })
    .then(function (result) {
        // If user response was negative, abort
        if (!result) {
            return;
        }

        registryClient = repository.getRegistryClient();

        logger.action('unregister', name, { name: name });

        return Q.nfcall(registryClient.unregister.bind(registryClient), name);
    })
    .then(function (result) {
        tracker.track('unregistered');
        logger.info('Package unregistered', name);

        return result;
    });
}

// -------------------

unregister.readOptions = function (argv) {
    var cli = require('../util/cli');

    var options = cli.readOptions(argv);
    var name = options.argv.remain[1];

    return [name];
};

module.exports = unregister;
