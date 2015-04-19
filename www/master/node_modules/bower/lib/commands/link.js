var path = require('path');
var rimraf = require('rimraf');
var Q = require('q');
var Project = require('../core/Project');
var createLink = require('../util/createLink');
var defaultConfig = require('../config');

function link(logger, name, localName, config) {
    if (name) {
        return linkTo(logger, name, localName, config);
    } else {
        return linkSelf(logger, config);
    }
}

function linkSelf(logger, config) {
    var project;

    config = defaultConfig(config);
    project = new Project(config, logger);

    return project.getJson()
    .then(function (json) {
        var src = config.cwd;
        var dst = path.join(config.storage.links, json.name);

        // Delete previous link if any
        return Q.nfcall(rimraf, dst)
        // Link globally
        .then(function () {
            return createLink(src, dst);
        })
        .then(function () {
            return {
                src: src,
                dst: dst
            };
        });
    });
}

function linkTo(logger, name, localName, config) {
    var src;
    var dst;
    var project;

    config = defaultConfig(config);
    project = new Project(config, logger);

    localName = localName || name;
    src = path.join(config.storage.links, name);
    dst = path.join(config.cwd, config.directory, localName);

    // Delete destination folder if any
    return Q.nfcall(rimraf, dst)
    // Link locally
    .then(function () {
        return createLink(src, dst);
    })
    // Install linked package deps
    .then(function () {
        return project.update([localName]);
    })
    .then(function (installed) {
        return {
            src: src,
            dst: dst,
            installed: installed
        };
    });
}

// -------------------

link.readOptions = function (argv) {
    var cli = require('../util/cli');
    var options = cli.readOptions(argv);
    var name = options.argv.remain[1];
    var localName = options.argv.remain[2];

    return [name, localName];
};

module.exports = link;
