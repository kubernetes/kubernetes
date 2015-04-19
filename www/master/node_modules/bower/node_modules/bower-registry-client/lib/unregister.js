var parseUrl = require('url').parse;
var request = require('request');
var createError = require('./util/createError');

function unregister(name, callback) {
    var config = this._config;
    var requestUrl = config.registry.register + '/packages/' + name;
    var remote = parseUrl(requestUrl);
    var headers = {};

    if (config.userAgent) {
        headers['User-Agent'] = config.userAgent;
    }

    if (config.accessToken) {
        requestUrl += '?access_token=' + config.accessToken;
    }

    request.del({
        url: requestUrl,
        proxy: remote.protocol === 'https:' ? config.httpsProxy : config.proxy,
        headers: headers,
        ca: config.ca.register,
        strictSSL: config.strictSsl,
        timeout: config.timeout
    }, function (err, response) {
        // If there was an internal error (e.g. timeout)
        if (err) {
            return callback(createError('Request to ' + requestUrl + ' failed: ' + err.message, err.code));
        }

        // Forbidden
        if (response.statusCode === 403) {
            return callback(createError(response.body, 'EFORBIDDEN'));
        }

        // Everything other than 204 is unknown
        if (response.statusCode !== 204) {
            return callback(createError(response.body, 'EUNKNOWN'));
        }

        callback(null, {
            name: name
        });
    });
}

module.exports = unregister;
