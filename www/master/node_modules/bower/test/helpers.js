require('chalk').enabled = false;

var Q = require('q');
var path = require('path');
var mkdirp = require('mkdirp');
var rimraf = require('rimraf');
var uuid = require('node-uuid');
var object = require('mout/object');
var fs = require('fs');
var glob = require('glob');
var os = require('os');
var which = require('which');
var path = require('path');
var proxyquire = require('proxyquire').noCallThru().noPreserveCache();
var spawnSync = require('spawn-sync');
var config = require('../lib/config');

// For better promise errors
Q.longStackSupport = true;

// Those are needed for Travis or not configured git environment
var env = {
    'GIT_AUTHOR_DATE': 'Sun Apr 7 22:13:13 2013 +0000',
    'GIT_AUTHOR_NAME': 'André Cruz',
    'GIT_AUTHOR_EMAIL': 'amdfcruz@gmail.com',
    'GIT_COMMITTER_DATE': 'Sun Apr 7 22:13:13 2013 +0000',
    'GIT_COMMITTER_NAME': 'André Cruz',
    'GIT_COMMITTER_EMAIL': 'amdfcruz@gmail.com',
};

object.mixIn(process.env, env);

var tmpLocation = path.join(
    os.tmpdir ? os.tmpdir() : os.tmpDir(),
    'bower-tests',
    uuid.v4().slice(0, 8)
);

exports.require = function (name, stubs) {
    if (stubs) {
        return proxyquire(path.join(__dirname, '../', name), stubs);
    } else {
        return require(path.join(__dirname, '../', name));
    }
};

// We need to reset cache because tests are reusing temp directories
beforeEach(function () {
    config.reset();
});

after(function () {
    rimraf.sync(tmpLocation);
});

exports.TempDir = (function() {
    function TempDir (defaults) {
        this.path = path.join(tmpLocation, uuid.v4());
        this.defaults = defaults;
    }

    TempDir.prototype.create = function (files, defaults) {
        var that = this;

        defaults = defaults || this.defaults || {};
        files = object.merge(files || {}, defaults);

        this.meta = function(tag) {
            if (tag) {
                return files[tag]['bower.json'];
            } else {
                return files['bower.json'];
            }
        };

        if (files) {
            object.forOwn(files, function (contents, filepath) {
                if (typeof contents === 'object') {
                    contents = JSON.stringify(contents, null, ' ') + '\n';
                }

                var fullPath = path.join(that.path, filepath);
                mkdirp.sync(path.dirname(fullPath));
                fs.writeFileSync(fullPath, contents);
            });
        }

        return this;
    };

    TempDir.prototype.prepare = function (files) {
        rimraf.sync(this.path);
        mkdirp.sync(this.path);
        this.create(files);

        return this;
    };

    // TODO: Rewrite to synchronous form
    TempDir.prototype.prepareGit = function (revisions) {
        var that = this;

        revisions = object.merge(revisions || {}, this.defaults);

        rimraf.sync(that.path);

        mkdirp.sync(that.path);

        this.git('init');

        this.glob('./!(.git)').map(function (removePath) {
            var fullPath = path.join(that.path, removePath);

            rimraf.sync(fullPath);
        });

        object.forOwn(revisions, function (files, tag) {
            this.create(files, {});
            this.git('add', '-A');
            this.git('commit', '-m"commit"');
            this.git('tag', tag);
        }.bind(this));

        return this;
    };

    TempDir.prototype.glob = function (pattern) {
        return glob.sync(pattern, {
            cwd: this.path,
            dot: true
        });
    };

    TempDir.prototype.getPath = function (name) {
        return path.join(this.path, name);
    };

    TempDir.prototype.read = function (name) {
        return fs.readFileSync(this.getPath(name), 'utf8');
    };

    TempDir.prototype.readJson = function (name) {
        return JSON.parse(this.read(name));
    };

    TempDir.prototype.git = function () {
        var args = Array.prototype.slice.call(arguments);
        var result = spawnSync('git', args, { cwd: this.path });

        if (result.status !== 0) {
            throw new Error(result.stderr);
        } else {
            return result.stdout.toString();
        }
    };

    TempDir.prototype.exists = function (name) {
        return fs.existsSync(path.join(this.path, name));
    };

    return TempDir;
})();

exports.expectEvent = function expectEvent(emitter, eventName) {
    var deferred = Q.defer();

    emitter.once(eventName, function () {
        deferred.resolve(arguments);
    });

    emitter.once('error', function (reason) {
        deferred.reject(reason);
    });

    return deferred.promise;
};

exports.command = function (command, stubs) {
    var rawCommand;
    var commandStubs = {};

    stubs = stubs || {};
    var cwd = stubs.cwd;
    delete stubs.cwd;

    rawCommand = exports.require(
        'lib/commands/' + command, stubs
    );

    commandStubs['./' + command] = function () {
        var args = [].slice.call(arguments);
        args[rawCommand.length - 1] = object.merge({ cwd: cwd }, args[rawCommand.length - 1] || {});
        return rawCommand.apply(null, args);
    };

    var instance = exports.require(
        'lib/commands/index', commandStubs
    );

    var commandParts = command.split('/');

    while (commandParts.length > 0) {
        instance = instance[commandParts.shift()];
    }

    if (!instance) {
        throw new Error('Unknown command: ' + command);
    }

    // TODO: refactor tests, so they can use readOptions directly
    instance.readOptions = function (argv) {
        argv = ['node', 'bower'].concat(argv);
        argv = command.split('/').concat(argv);

        return rawCommand.readOptions(argv);
    };

    return instance;
};

exports.run = function (command, args) {
    var logger = command.apply(command, args || []);

    // Hack so we can intercept prompring for data
    logger.prompt = function(data) {
        logger.emit('confirm', data);
    };

    var promise = exports.expectEvent(logger, 'end');

    promise.logger = logger;

    return promise;
};

// Captures all stdout and stderr
exports.capture = function(callback) {
    var oldStdout = process.stdout.write;
    var oldStderr = process.stderr.write;

    var stdout = '';
    var stderr = '';

    process.stdout.write = function(text) {
        stdout += text;
    };

    process.stderr.write = function(text) {
        stderr += text;
    };

    return Q.fcall(callback).then(function() {
        process.stdout.write = oldStdout;
        process.stderr.write = oldStderr;

        return [stdout, stderr];
    }).fail(function(e) {
        process.stdout.write = oldStdout;
        process.stderr.write = oldStderr;

        throw e;
    });
};

exports.hasSvn = function() {
    try {
        which.sync('svn');
        return true;
    } catch (ex) {
        return false;
    }
};

exports.isWin = function() {
    return process.platform === 'win32';
};

exports.localSource = function (localPath) {
    localPath = path.normalize(localPath);

    if (!exports.isWin()) {
        localPath = 'file://' + localPath;
    }

    return localPath;
};

// Used for example by "svn checkout" and "svn export"
exports.localUrl = function (localPath) {
    localPath = path.normalize(localPath);

    if (!exports.isWin()) {
        localPath = 'file://' + localPath;
    } else {
        localPath = 'file:///' + localPath;
    }

    return localPath;
};
