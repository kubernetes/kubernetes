var object = require('mout/object');
var lang = require('mout/lang');
var string = require('mout/string');

function camelCase(config) {
    var camelCased = {};

    // Camel case
    object.forOwn(config, function (value, key) {
        // Ignore null values
        if (value == null) {
            return;
        }

        key = string.camelCase(key.replace(/_/g, '-'));
        camelCased[key] = lang.isPlainObject(value) ? camelCase(value) : value;
    });

    return camelCased;
}

function expand(config) {
    config = camelCase(config);

    // Expand some properties
    // Registry
    if (typeof config.registry === 'string') {
        config.registry = {
            search: [config.registry],
            register: config.registry,
            publish: config.registry
        };
    } else if (config.registry) {
        if (config.registry.search && !Array.isArray(config.registry.search)) {
            config.registry.search = [config.registry.search];
        }
    }

    // CA
    if (typeof config.ca === 'string') {
        config.ca = {
            search: [config.ca],
            register: config.ca,
            publish: config.ca
        };
    } else if (config.ca) {
        if (config.ca.search && !Array.isArray(config.ca.search)) {
            config.ca.search = [config.ca.search];
        }
    }

    return config;
}

module.exports = expand;
