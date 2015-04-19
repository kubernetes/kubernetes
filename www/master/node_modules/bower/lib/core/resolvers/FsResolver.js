var util = require('util');
var fs = require('graceful-fs');
var path = require('path');
var mout = require('mout');
var Q = require('q');
var junk = require('junk');
var Resolver = require('./Resolver');
var copy = require('../../util/copy');
var extract = require('../../util/extract');
var createError = require('../../util/createError');

function FsResolver(decEndpoint, config, logger) {
    Resolver.call(this, decEndpoint, config, logger);

    // Ensure absolute path
    this._source = path.resolve(this._config.cwd, this._source);

    // If target was specified, simply reject the promise
    if (this._target !== '*') {
        throw createError('File system sources can\'t resolve targets', 'ENORESTARGET');
    }

    // If the name was guessed
    if (this._guessedName) {
        // Remove extension
        this._name = this._name.substr(0, this._name.length - path.extname(this._name).length);
    }
}

util.inherits(FsResolver, Resolver);
mout.object.mixIn(FsResolver, Resolver);

// -----------------

FsResolver.isTargetable = function () {
    return false;
};

// TODO: Should we store latest mtimes in the resolution and compare?
//       This would be beneficial when copying big files/folders

// TODO: There's room for improvement by using streams if the source
//       is an archive file, by piping read stream to the zip extractor
//       This will likely increase the complexity of code but might worth it
FsResolver.prototype._resolve = function () {
    return this._copy()
    .then(this._extract.bind(this))
    .then(this._rename.bind(this));
};

// -----------------

FsResolver.prototype._copy = function () {
    var that = this;

    return Q.nfcall(fs.stat, this._source)
    .then(function (stat) {
        var dst;
        var copyOpts;
        var promise;

        that._sourceStat = stat;
        copyOpts = { mode: stat.mode };

        // If it's a folder
        if (stat.isDirectory()) {
            dst = that._tempDir;

            // Read the bower.json inside the folder, so that we
            // copy only the necessary files if it has ignore specified
            promise = that._readJson(that._source)
            .then(function (json) {
                copyOpts.ignore = json.ignore;
                return copy.copyDir(that._source, dst, copyOpts);
            })
            .then(function () {
                // Resolve to null because it's a dir
                return;
            });
        // Else it's a file
        } else {
            dst = path.join(that._tempDir, path.basename(that._source));
            promise = copy.copyFile(that._source, dst, copyOpts)
            .then(function () {
                return dst;
            });
        }

        that._logger.action('copy', that._source, {
            src: that._source,
            dst: dst
        });

        return promise;
    });
};

FsResolver.prototype._extract = function (file) {
    if (!file || !extract.canExtract(file)) {
        return Q.resolve();
    }

    this._logger.action('extract', path.basename(this._source), {
        archive: file,
        to: this._tempDir
    });

    return extract(file, this._tempDir);
};

FsResolver.prototype._rename = function () {
    return Q.nfcall(fs.readdir, this._tempDir)
    .then(function (files) {
        var file;
        var oldPath;
        var newPath;

        // Remove any OS specific files from the files array
        // before checking its length
        files = files.filter(junk.isnt);

        // Only rename if there's only one file and it's not the json
        if (files.length === 1 && !/^(bower|component)\.json$/.test(files[0])) {
            file = files[0];
            this._singleFile = 'index' + path.extname(file);
            oldPath = path.join(this._tempDir, file);
            newPath = path.join(this._tempDir, this._singleFile);

            return Q.nfcall(fs.rename, oldPath, newPath);
        }
    }.bind(this));
};

FsResolver.prototype._savePkgMeta = function (meta) {
    // Store main if is a single file
    if (this._singleFile) {
        meta.main = this._singleFile;
    }

    return Resolver.prototype._savePkgMeta.call(this, meta);
};

module.exports = FsResolver;
