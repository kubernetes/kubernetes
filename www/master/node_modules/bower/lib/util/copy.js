var fstream = require('fstream');
var fstreamIgnore = require('fstream-ignore');
var fs = require('graceful-fs');
var Q = require('q');

function copy(reader, writer) {
    var deferred;
    var ignore;

    // Filter symlinks because they are not 100% portable, specially
    // when linking between different drives
    // Following can't be enabled either because symlinks that reference
    // another symlinks will get filtered
    // See: https://github.com/bower/bower/issues/699
    reader.filter = filterSymlinks;
    reader.follow = false;

    if (reader.type === 'Directory' && reader.ignore) {
        ignore = reader.ignore;
        reader = fstreamIgnore(reader);
        reader.addIgnoreRules(ignore);
    } else {
        reader = fstream.Reader(reader);
    }

    deferred = Q.defer();

    reader
    .on('error', deferred.reject)
    // Pipe to writer
    .pipe(fstream.Writer(writer))
    .on('error', deferred.reject)
    .on('close', deferred.resolve);

    return deferred.promise;
}

function copyMode(src, dst) {
    return Q.nfcall(fs.stat, src)
    .then(function (stat) {
        return Q.nfcall(fs.chmod, dst, stat.mode);
    });
}

function filterSymlinks(entry) {
    return entry.type !== 'SymbolicLink';
}

function parseOptions(opts) {
    opts = opts || {};

    if (opts.mode != null) {
        opts.copyMode = false;
    } else if (opts.copyMode == null) {
        opts.copyMode = true;
    }

    return opts;
}

// ---------------------

// Available options:
// - mode: force final mode of dst (defaults to null)
// - copyMode: copy mode of src to dst, only if mode is not specified (defaults to true)
function copyFile(src, dst, opts) {
    var promise;

    opts = parseOptions(opts);

    promise = copy({
        path: src,
        type: 'File'
    }, {
        path: dst,
        mode: opts.mode,
        type: 'File'
    });

    if (opts.copyMode) {
        promise = promise.then(copyMode.bind(copyMode, src, dst));
    }

    return promise;
}

// Available options:
// - ignore: array of patterns to be ignored (defaults to null)
// - mode: force final mode of dst (defaults to null)
// - copyMode: copy mode of src to dst, only if mode is not specified (defaults to true)
function copyDir(src, dst, opts) {
    var promise;

    opts = parseOptions(opts);

    promise = copy({
        path: src,
        type: 'Directory',
        ignore: opts.ignore
    }, {
        path: dst,
        mode: opts.mode,
        type: 'Directory'
    });

    if (opts.copyMode) {
        promise = promise.then(copyMode.bind(copyMode, src, dst));
    }

    return promise;
}

module.exports.copyDir = copyDir;
module.exports.copyFile = copyFile;
