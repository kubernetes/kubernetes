var mout = require('mout');
var fs = require('graceful-fs');
var path = require('path');
var Q = require('q');
var endpointParser = require('bower-endpoint-parser');
var Project = require('../core/Project');
var defaultConfig = require('../config');
var GitHubResolver = require('../core/resolvers/GitHubResolver');
var GitFsResolver = require('../core/resolvers/GitFsResolver');
var cmd = require('../util/cmd');
var createError = require('../util/createError');

function init(logger, config) {
    var project;

    config = defaultConfig(config);

    // This command requires interactive to be enabled
    if (!config.interactive) {
        throw createError('Register requires an interactive shell', 'ENOINT', {
            details: 'Note that you can manually force an interactive shell with --config.interactive'
        });
    }

    project = new Project(config, logger);

    // Start with existing JSON details
    return readJson(project, logger)
    // Fill in defaults
    .then(setDefaults.bind(null, config))
    // Now prompt user to make changes
    .then(promptUser.bind(null, logger))
    // Set ignore based on the response
    .spread(setIgnore.bind(null, config))
    // Set dependencies based on the response
    .spread(setDependencies.bind(null, project))
    // All done!
    .spread(saveJson.bind(null, project, logger));
}

function readJson(project, logger) {
    return project.hasJson()
    .then(function (json) {
        if (json) {
            logger.warn('existing', 'The existing ' + path.basename(json) + ' file will be used and filled in');
        }

        return project.getJson();
    });
}

function saveJson(project, logger, json) {
    // Cleanup empty props (null values, empty strings, objects and arrays)
    mout.object.forOwn(json, function (value, key) {
        if (value == null || mout.lang.isEmpty(value)) {
            delete json[key];
        }
    });

    logger.info('json', 'Generated json', { json: json });

    // Confirm the json with the user
    return Q.nfcall(logger.prompt.bind(logger), {
        type: 'confirm',
        message: 'Looks good?',
        default: true
    })
    .then(function (good) {
        if (!good) {
            return null;
        }

        // Save json (true forces file creation)
        return project.saveJson(true);
    });
}

function setDefaults(config, json) {
    var name;
    var promise = Q.resolve();

    // Name
    if (!json.name) {
        json.name = path.basename(config.cwd);
    }

    // Version
    if (!json.version) {
        // Assume latest semver tag if it's a git repo
        promise = promise.then(function () {
            return GitFsResolver.versions(config.cwd)
            .then(function (versions) {
                json.version = versions[0] || '0.0.0';
            }, function () {
                json.version = '0.0.0';
            });
        });
    }

    // Main
    if (!json.main) {
        // Remove '.js' from the end of the package name if it is there
        name = path.basename(json.name, '.js');

        if (fs.existsSync(path.join(config.cwd, 'index.js'))) {
            json.main = 'index.js';
        } else if (fs.existsSync(path.join(config.cwd, name + '.js'))) {
            json.main = name + '.js';
        }
    }

    // Homepage
    if (!json.homepage) {
        // Set as GitHub homepage if it's a GitHub repository
        promise = promise.then(function () {
            return cmd('git', ['config', '--get', 'remote.origin.url'])
            .spread(function (stdout) {
                var pair;

                stdout = stdout.trim();
                if (!stdout) {
                    return;
                }

                pair = GitHubResolver.getOrgRepoPair(stdout);
                if (pair) {
                    json.homepage = 'https://github.com/' + pair.org + '/' + pair.repo;
                }
            })
            .fail(function () { });
        });
    }

    if (!json.authors) {
        promise = promise.then(function () {
            // Get the user name configured in git
            return cmd('git', ['config', '--get', '--global', 'user.name'])
            .spread(function (stdout) {
                var gitEmail;
                var gitName = stdout.trim();

                // Abort if no name specified
                if (!gitName) {
                    return;
                }

                // Get the user email configured in git
                return cmd('git', ['config', '--get', '--global', 'user.email'])
                .spread(function (stdout) {
                    gitEmail = stdout.trim();
                }, function () {})
                .then(function () {
                    json.authors = gitName;
                    json.authors += gitEmail ? ' <' + gitEmail + '>' : '';
                });
            }, function () {});
        });
    }

    return promise.then(function () {
        return json;
    });
}

function promptUser(logger, json) {
    var questions = [
        {
            'name': 'name',
            'message': 'name',
            'default': json.name,
            'type': 'input'
        },
        {
            'name': 'version',
            'message': 'version',
            'default': json.version,
            'type': 'input'
        },
        {
            'name': 'description',
            'message': 'description',
            'default': json.description,
            'type': 'input'
        },
        {
            'name': 'main',
            'message': 'main file',
            'default': json.main,
            'type': 'input'
        },
        {
            'name': 'moduleType',
            'message': 'what types of modules does this package expose?',
            'type': 'checkbox',
            'choices': ['amd', 'es6', 'globals', 'node', 'yui']
        },
        {
            'name': 'keywords',
            'message': 'keywords',
            'default': json.keywords ? json.keywords.toString() : null,
            'type': 'input'
        },
        {
            'name': 'authors',
            'message': 'authors',
            'default': json.authors ? json.authors.toString() : null,
            'type': 'input'
        },
        {
            'name': 'license',
            'message': 'license',
            'default': json.license || 'MIT',
            'type': 'input'
        },
        {
            'name': 'homepage',
            'message': 'homepage',
            'default': json.homepage,
            'type': 'input'
        },
        {
            'name': 'dependencies',
            'message': 'set currently installed components as dependencies?',
            'default': !mout.object.size(json.dependencies) && !mout.object.size(json.devDependencies),
            'type': 'confirm'
        },
        {
            'name': 'ignore',
            'message': 'add commonly ignored files to ignore list?',
            'default': true,
            'type': 'confirm'
        },
        {
            'name': 'private',
            'message': 'would you like to mark this package as private which prevents it from being accidentally published to the registry?',
            'default': !!json.private,
            'type': 'confirm'
        }
    ];

    return Q.nfcall(logger.prompt.bind(logger), questions)
    .then(function (answers) {
        json.name = answers.name;
        json.version = answers.version;
        json.description = answers.description;
        json.main = answers.main;
        json.moduleType = answers.moduleType;
        json.keywords = toArray(answers.keywords);
        json.authors = toArray(answers.authors, ',');
        json.license = answers.license;
        json.homepage = answers.homepage;
        json.private = answers.private || null;

        return [json, answers];
    });
}

function toArray(value, splitter) {
    var arr = value.split(splitter || /[\s,]/);

    // Trim values
    arr = arr.map(function (item) {
        return item.trim();
    });

    // Filter empty values
    arr = arr.filter(function (item) {
        return !!item;
    });

    return arr.length ? arr : null;
}

function setIgnore(config, json, answers) {
    if (answers.ignore) {
        json.ignore = mout.array.combine(json.ignore || [], [
            '**/.*',
            'node_modules',
            'bower_components',
            config.directory,
            'test',
            'tests'
        ]);
    }

    return [json, answers];
}

function setDependencies(project, json, answers) {
    if (answers.dependencies) {
        return project.getTree()
        .spread(function (tree, flattened, extraneous) {
            if (extraneous.length) {
                json.dependencies = {};

                // Add extraneous as dependencies
                extraneous.forEach(function (extra) {
                    var jsonEndpoint;

                    // Skip linked packages
                    if (extra.linked) {
                        return;
                    }

                    jsonEndpoint = endpointParser.decomposed2json(extra.endpoint);
                    mout.object.mixIn(json.dependencies, jsonEndpoint);
                });
            }

            return [json, answers];
        });
    }

    return [json, answers];
}

// -------------------

init.readOptions = function (argv) {
    return [];
};

module.exports = init;
