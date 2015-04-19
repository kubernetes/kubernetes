var util = require('util');
var Q = require('q');
var mout = require('mout');
var path = require('path');
var GitResolver = require('./GitResolver');
var copy = require('../../util/copy');
var cmd = require('../../util/cmd');

function GitFsResolver(decEndpoint, config, logger) {
    GitResolver.call(this, decEndpoint, config, logger);

    // Ensure absolute path
    this._source = path.resolve(this._config.cwd, this._source);
}

util.inherits(GitFsResolver, GitResolver);
mout.object.mixIn(GitFsResolver, GitResolver);

// -----------------

// Override the checkout function to work with the local copy
GitFsResolver.prototype._checkout = function () {
    var resolution = this._resolution;

    // The checkout process could be similar to the GitRemoteResolver by prepending file:// to the source
    // But from my performance measures, it's faster to copy the folder and just checkout in there
    this._logger.action('checkout', resolution.tag || resolution.branch || resolution.commit, {
        resolution: resolution,
        to: this._tempDir
    });

    // Copy files to the temporary directory first
    return this._copy()
    .then(cmd.bind(cmd, 'git', ['checkout', '-f', resolution.tag || resolution.branch || resolution.commit], { cwd: this._tempDir }))
    // Cleanup unstaged files
    .then(cmd.bind(cmd, 'git', ['clean', '-f', '-d'], { cwd: this._tempDir }));
};

GitFsResolver.prototype._copy = function () {
    return copy.copyDir(this._source, this._tempDir);
};

// -----------------

// Grab refs locally
GitFsResolver.refs = function (source) {
    var value;

    // TODO: Normalize source because of the various available protocols?
    value = this._cache.refs.get(source);
    if (value) {
        return Q.resolve(value);
    }

    value = cmd('git', ['show-ref', '--tags', '--heads'], { cwd : source })
    .spread(function (stdout) {
        var refs;

        refs = stdout.toString()
        .trim()                         // Trim trailing and leading spaces
        .replace(/[\t ]+/g, ' ')        // Standardize spaces (some git versions make tabs, other spaces)
        .split(/[\r\n]+/);              // Split lines into an array

        // Update the refs with the actual refs
        this._cache.refs.set(source, refs);

        return refs;
    }.bind(this));

    // Store the promise to be reused until it resolves
    // to a specific value
    this._cache.refs.set(source, value);

    return value;
};

module.exports = GitFsResolver;
