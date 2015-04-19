var Q = require('q');
var mkdirp = require('mkdirp');
var rimraf = require('rimraf');
var uuid = require('node-uuid');
var object = require('mout/object');
var fs = require('fs');
var glob = require('glob');
var os = require('os');
var path = require('path');

// For better promise errors
Q.longStackSupport = true;

var tmpLocation = path.join(
    os.tmpdir ? os.tmpdir() : os.tmpDir(),
    'bower-config-tests',
    uuid.v4().slice(0, 8)
);

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

        var promise = new Q();

        object.forOwn(revisions, function (files, tag) {
            promise = promise.then(function () {
                return that.git('init');
            }).then(function () {
                that.glob('./!(.git)').map(function (removePath) {
                    var fullPath = path.join(that.path, removePath);

                    rimraf.sync(fullPath);
                });

                that.create(files, {});
            }).then(function () {
                return that.git('add', '-A');
            }).then(function () {
                return that.git('commit', '-m"commit"');
            }).then(function () {
                return that.git('tag', tag);
            });
        });

        return promise;
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

    TempDir.prototype.exists = function (name) {
        return fs.existsSync(path.join(this.path, name));
    };

    return TempDir;
})();