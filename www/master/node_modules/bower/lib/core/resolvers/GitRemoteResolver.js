var util = require('util');
var url = require('url');
var Q = require('q');
var mout = require('mout');
var LRU = require('lru-cache');
var GitResolver = require('./GitResolver');
var cmd = require('../../util/cmd');

function GitRemoteResolver(decEndpoint, config, logger) {
    GitResolver.call(this, decEndpoint, config, logger);

    if (!mout.string.startsWith(this._source, 'file://')) {
        // Trim trailing slashes
        this._source = this._source.replace(/\/+$/, '');
    }

    // If the name was guessed, remove the trailing .git
    if (this._guessedName && mout.string.endsWith(this._name, '.git')) {
        this._name = this._name.slice(0, -4);
    }

    // Get the host of this source
    if (!/:\/\//.test(this._source)) {
        this._host = url.parse('ssh://' + this._source).host;
    } else {
        this._host = url.parse(this._source).host;
    }

    this._remote = url.parse(this._source);

    // Verify whether the server supports shallow cloning
    this._shallowClone = this._supportsShallowCloning;
}

util.inherits(GitRemoteResolver, GitResolver);
mout.object.mixIn(GitRemoteResolver, GitResolver);

// -----------------

GitRemoteResolver.prototype._checkout = function () {
    var promise;
    var timer;
    var reporter;
    var that = this;
    var resolution = this._resolution;

    this._logger.action('checkout', resolution.tag || resolution.branch || resolution.commit, {
        resolution: resolution,
        to: this._tempDir
    });

    // If resolution is a commit, we need to clone the entire repo and check it out
    // Because a commit is not a named ref, there's no better solution
    if (resolution.type === 'commit') {
        promise = this._slowClone(resolution);
    // Otherwise we are checking out a named ref so we can optimize it
    } else {
        promise = this._fastClone(resolution);
    }

    // Throttle the progress reporter to 1 time each sec
    reporter = mout.fn.throttle(function (data) {
        var lines;

        lines = data.split(/[\r\n]+/);
        lines.forEach(function (line) {
            if (/\d{1,3}\%/.test(line)) {
                // TODO: There are some strange chars that appear once in a while (\u001b[K)
                //       Trim also those?
                that._logger.info('progress', line.trim());
            }
        });
    }, 1000);

    // Start reporting progress after a few seconds
    timer = setTimeout(function () {
        promise.progress(reporter);
    }, 8000);

    return promise
    // Add additional proxy information to the error if necessary
    .fail(function (err) {
        that._suggestProxyWorkaround(err);
        throw err;
    })
    // Clear timer at the end
    .fin(function () {
        clearTimeout(timer);
        reporter.cancel();
    });
};

GitRemoteResolver.prototype._findResolution = function (target) {
    var that = this;

    // Override this function to include a meaningful message related to proxies
    // if necessary
    return GitResolver.prototype._findResolution.call(this, target)
    .fail(function (err) {
        that._suggestProxyWorkaround(err);
        throw err;
    });
};

// ------------------------------

GitRemoteResolver.prototype._slowClone = function (resolution) {
    return cmd('git', ['clone', this._source, this._tempDir, '--progress'])
    .then(cmd.bind(cmd, 'git', ['checkout', resolution.commit], { cwd: this._tempDir }));
};

GitRemoteResolver.prototype._fastClone = function (resolution) {
    var branch,
        args,
        that = this;

    branch = resolution.tag || resolution.branch;
    args = ['clone',  this._source, '-b', branch, '--progress', '.'];

    return this._shallowClone().then(function (shallowCloningSupported) {
        // If the host does not support shallow clones, we don't use --depth=1
        if (shallowCloningSupported && !GitRemoteResolver._noShallow.get(this._host)) {
            args.push('--depth', 1);
        }

        return cmd('git', args, { cwd: that._tempDir })
        .spread(function (stdout, stderr) {
            // Only after 1.7.10 --branch accepts tags
            // Detect those cases and inform the user to update git otherwise it's
            // a lot slower than newer versions
            if (!/branch .+? not found/i.test(stderr)) {
                return;
            }

            that._logger.warn('old-git', 'It seems you are using an old version of git, it will be slower and propitious to errors!');
            return cmd('git', ['checkout', resolution.commit], { cwd: that._tempDir });
        }, function (err) {
            // Some git servers do not support shallow clones
            // When that happens, we mark this host and try again
            if (!GitRemoteResolver._noShallow.has(that._source) &&
                err.details &&
                /(rpc failed|shallow|--depth)/i.test(err.details)
                ) {
                GitRemoteResolver._noShallow.set(that._host, true);
                return that._fastClone(resolution);
            }

            throw err;
        });
    });
};

GitRemoteResolver.prototype._suggestProxyWorkaround = function (err) {
    if ((this._config.proxy || this._config.httpsProxy) &&
        mout.string.startsWith(this._source, 'git://') &&
        err.code === 'ECMDERR' && err.details
    ) {
        err.details = err.details.trim();
        err.details += '\n\nWhen under a proxy, you must configure git to use https:// instead of git://.';
        err.details += '\nYou can configure it for every endpoint or for this specific host as follows:';
        err.details += '\ngit config --global url."https://".insteadOf git://';
        err.details += '\ngit config --global url."https://' + this._host + '".insteadOf git://' + this._host;
        err.details += 'Ignore this suggestion if you already have this configured.';
    }
};

// Verifies whether the server supports shallow cloning.
// This is done according to the rules found in the following links:
// * https://github.com/dimitri/el-get/pull/1921/files
// * http://stackoverflow.com/questions/9270488/is-it-possible-to-detect-whether-a-http-git-remote-is-smart-or-dumb
//
// Summary of the rules:
// * Protocols like ssh or git always support shallow cloning
// * HTTP-based protocols can be verified by sending a HEAD or GET request to the URI (appended to the URL of the Git repo):
//       /info/refs?service=git-upload-pack
// * If the server responds with a 'Content-Type' header of 'application/x-git-upload-pack-advertisement',
//      the server supports shallow cloning ("smart server")
// * If the server responds with a different content type, the server does not support shallow cloning ("dumb server")
// * Instead of doing the HEAD or GET request using an HTTP client, we're letting Git and Curl do the heavy lifting.
//      Calling Git with the GIT_CURL_VERBOSE=2 env variable will provide the Git and Curl output, which includes
//      the content type. This has the advantage that Git will take care of using stored credentials and any additional
//      negotiation that needs to take place.
//
// The above should cover most cases, including BitBucket.
GitRemoteResolver.prototype._supportsShallowCloning = function () {
    var value = true;

    // Verify that the remote could be parsed and that a protocol is set
    // This case is unlikely, but let's still cover it.
    if (this._remote == null || this._remote.protocol == null) {
        return Q.resolve(false);
    }

    // Check for protocol - the remote check for hosts supporting shallow cloning is only required for
    // HTTP or HTTPS, not for Git or SSH.
    // Also check for hosts that have been checked in a previous request and have been found to support
    // shallow cloning.
    if (mout.string.startsWith(this._remote.protocol, 'http')
            && !GitRemoteResolver._canShallow.get(this._host)) {
        // Provide GIT_CURL_VERBOSE=2 environment variable to capture curl output.
        // Calling ls-remote includes a call to the git-upload-pack service, which returns the content type in the response.
        var processEnv = mout.object.merge(process.env, { 'GIT_CURL_VERBOSE': 2 });

        value = cmd('git', ['ls-remote', '--heads', this._source], {
            env: processEnv
        })
        .spread(function (stdout, stderr) {
            // Check stderr for content-type, ignore stdout
            var isSmartServer;

            // If the content type is 'x-git', then the server supports shallow cloning
            isSmartServer = mout.string.contains(stderr,
                'Content-Type: application/x-git-upload-pack-advertisement');

            this._logger.debug('detect-smart-git', 'Smart Git host detected: ' + isSmartServer);

            if (isSmartServer) {
                // Cache this host
                GitRemoteResolver._canShallow.set(this._host, true);
            }

            return isSmartServer;
        }.bind(this));
    }
    else {
        // One of the following cases:
        // * A non-HTTP/HTTPS protocol
        // * A host that has been checked before and that supports shallow cloning
        return Q.resolve(true);
    }

    return value;
};

// ------------------------------

// Grab refs remotely
GitRemoteResolver.refs = function (source) {
    var value;

    // TODO: Normalize source because of the various available protocols?
    value = this._cache.refs.get(source);
    if (value) {
        return Q.resolve(value);
    }

    // Store the promise in the refs object
    value = cmd('git', ['ls-remote', '--tags', '--heads', source])
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

// Store hosts that do not support shallow clones here
GitRemoteResolver._noShallow = new LRU({ max: 50, maxAge: 5 * 60 * 1000 });

// Store hosts that support shallow clones here
GitRemoteResolver._canShallow = new LRU({ max: 50, maxAge: 5 * 60 * 1000 });

module.exports = GitRemoteResolver;
