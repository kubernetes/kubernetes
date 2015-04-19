var util = require('util');
var path = require('path');
var mout = require('mout');
var Q = require('q');
var GitRemoteResolver = require('./GitRemoteResolver');
var download = require('../../util/download');
var extract = require('../../util/extract');
var createError = require('../../util/createError');

function GitHubResolver(decEndpoint, config, logger) {
    var pair;

    GitRemoteResolver.call(this, decEndpoint, config, logger);

    // Grab the org/repo
    // /xxxxx/yyyyy.git or :xxxxx/yyyyy.git (.git is optional)
    pair = GitHubResolver.getOrgRepoPair(this._source);
    if (!pair) {
        throw createError('Invalid GitHub URL', 'EINVEND', {
            details: this._source + ' does not seem to be a valid GitHub URL'
        });
    }

    this._org = pair.org;
    this._repo = pair.repo;

    // Ensure trailing for all protocols
    if (!mout.string.endsWith(this._source, '.git')) {
        this._source += '.git';
    }

    // Check if it's public
    this._public = mout.string.startsWith(this._source, 'git://');

    // Use https:// rather than git:// if on a proxy
    if (this._config.proxy || this._config.httpsProxy) {
        this._source = this._source.replace('git://', 'https://');
    }

    // Enable shallow clones for GitHub repos
    this._shallowClone = function() {
        return Q.resolve(true);
    };
}

util.inherits(GitHubResolver, GitRemoteResolver);
mout.object.mixIn(GitHubResolver, GitRemoteResolver);

// -----------------

GitHubResolver.prototype._checkout = function () {
    // Only fully works with public repositories and tags
    // Could work with https/ssh protocol but not with 100% certainty
    if (!this._public || !this._resolution.tag) {
        return GitRemoteResolver.prototype._checkout.call(this);
    }

    var msg;
    var tarballUrl = 'https://github.com/' + this._org + '/' + this._repo + '/archive/' + this._resolution.tag + '.tar.gz';
    var file = path.join(this._tempDir, 'archive.tar.gz');
    var reqHeaders = {};
    var that = this;

    if (this._config.userAgent) {
        reqHeaders['User-Agent'] = this._config.userAgent;
    }

    this._logger.action('download', tarballUrl, {
        url: that._source,
        to: file
    });

    // Download tarball
    return download(tarballUrl, file, {
        proxy: this._config.httpsProxy,
        strictSSL: this._config.strictSsl,
        timeout: this._config.timeout,
        headers: reqHeaders
    })
    .progress(function (state) {
        // Retry?
        if (state.retry) {
            msg = 'Download of ' + tarballUrl + ' failed with ' + state.error.code + ', ';
            msg += 'retrying in ' + (state.delay / 1000).toFixed(1) + 's';
            that._logger.debug('error', state.error.message, { error: state.error });
            return that._logger.warn('retry', msg);
        }

        // Progress
        msg = 'received ' + (state.received / 1024 / 1024).toFixed(1) + 'MB';
        if (state.total) {
            msg += ' of ' + (state.total / 1024 / 1024).toFixed(1) + 'MB downloaded, ';
            msg += state.percent + '%';
        }
        that._logger.info('progress', msg);
    })
    .then(function () {
        // Extract archive
        that._logger.action('extract', path.basename(file), {
            archive: file,
            to: that._tempDir
        });

        return extract(file, that._tempDir)
        // Fallback to standard git clone if extraction failed
        .fail(function (err) {
            msg =  'Decompression of ' + path.basename(file) + ' failed' + (err.code ? ' with ' + err.code : '') + ', ';
            msg += 'trying with git..';
            that._logger.debug('error', err.message, { error: err });
            that._logger.warn('retry', msg);

            return that._cleanTempDir()
            .then(GitRemoteResolver.prototype._checkout.bind(that));
        });
    // Fallback to standard git clone if download failed
    }, function (err) {
        msg = 'Download of ' + tarballUrl + ' failed' + (err.code ? ' with ' + err.code : '') + ', ';
        msg += 'trying with git..';
        that._logger.debug('error', err.message, { error: err });
        that._logger.warn('retry', msg);

        return that._cleanTempDir()
        .then(GitRemoteResolver.prototype._checkout.bind(that));

    });
};

GitHubResolver.prototype._savePkgMeta = function (meta) {
    // Set homepage if not defined
    if (!meta.homepage) {
        meta.homepage = 'https://github.com/' + this._org + '/' + this._repo;
    }

    return GitRemoteResolver.prototype._savePkgMeta.call(this, meta);
};

// ----------------

GitHubResolver.getOrgRepoPair = function (url) {
    var match;

    match = url.match(/(?:@|:\/\/)github.com[:\/]([^\/\s]+?)\/([^\/\s]+?)(?:\.git)?\/?$/i);
    if (!match) {
        return null;
    }

    return {
        org: match[1],
        repo: match[2]
    };
};

module.exports = GitHubResolver;
