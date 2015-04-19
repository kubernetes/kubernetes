var Q = require('q');
var RegistryClient = require('bower-registry-client');
var defaultConfig = require('../config');

function lookup(logger, name, config) {
    if (!name) {
        return new Q(null);
    }

    var registryClient;

    config = defaultConfig(config);
    config.cache = config.storage.registry;

    registryClient = new RegistryClient(config, logger);

    return Q.nfcall(registryClient.lookup.bind(registryClient), name)
    .then(function (entry) {
        // TODO: Handle entry.type.. for now it's only 'alias'
        //       When we got published packages, this needs to be adjusted
        return !entry ? null : {
            name: name,
            url: entry && entry.url
        };
    });
}

// -------------------

lookup.readOptions = function (argv) {
    var cli = require('../util/cli');
    var options = cli.readOptions(argv);
    var name = options.argv.remain[1];

    return [name];
};

module.exports = lookup;
