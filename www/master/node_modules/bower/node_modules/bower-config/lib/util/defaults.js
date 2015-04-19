var path = require('path');
var paths = require('./paths');

// Guess proxy defined in the env
/*jshint camelcase: false*/
var proxy = process.env.HTTP_PROXY
    || process.env.http_proxy
    || null;

var httpsProxy = process.env.HTTPS_PROXY
    || process.env.https_proxy
    || proxy;
/*jshint camelcase: true*/

// Use a well known user agent (in this case, curl) when using a proxy,
// to avoid potential filtering on many corporate proxies with blank or unknown agents
var userAgent = !proxy && !httpsProxy
    ? 'node/' + process.version + ' ' + process.platform + ' ' + process.arch
    : 'curl/7.21.4 (universal-apple-darwin11.0) libcurl/7.21.4 OpenSSL/0.9.8r zlib/1.2.5';

var defaults = {
    'cwd': process.cwd(),
    'directory': 'bower_components',
    'registry': 'https://bower.herokuapp.com',
    'shorthand-resolver': 'git://github.com/{{owner}}/{{package}}.git',
    'tmp': paths.tmp,
    'proxy': proxy,
    'https-proxy': httpsProxy,
    'timeout': 30000,
    'ca': { search: [] },
    'strict-ssl': true,
    'user-agent': userAgent,
    'color': true,
    'interactive': null,
    'storage': {
        packages: path.join(paths.cache, 'packages'),
        links: path.join(paths.data, 'links'),
        completion: path.join(paths.data, 'completion'),
        registry: path.join(paths.cache, 'registry'),
        empty: path.join(paths.data, 'empty')  // Empty dir, used in GIT_TEMPLATE_DIR among others
    }
};

module.exports = defaults;
