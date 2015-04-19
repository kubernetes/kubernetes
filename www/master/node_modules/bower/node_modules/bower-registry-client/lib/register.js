var parseUrl = require('url').parse;
var request = require('request');
var createError = require('./util/createError');

function register(name, url, callback) {
    var config = this._config;
    var requestUrl = config.registry.register + '/packages';
    var remote = parseUrl(requestUrl);
    var headers = {};

    if (config.userAgent) {
        headers['User-Agent'] = config.userAgent;
    }

    if (config.accessToken) {
        requestUrl += '?access_token=' + config.accessToken;
    }

    request.post({
        url: requestUrl,
        proxy: remote.protocol === 'https:' ? config.httpsProxy : config.proxy,
        headers: headers,
        ca: config.ca.register,
        strictSSL: config.strictSsl,
        timeout: config.timeout,
        json: true,
        form: {
            name: name,
            url: url
        }
    }, function (err, response) {
        // If there was an internal error (e.g. timeout)
        if (err) {
            return callback(createError('Request to ' + requestUrl + ' failed: ' + err.message, err.code));
        }

        // Duplicate
        if (response.statusCode === 403) {
            return callback(createError('Duplicate package', 'EDUPLICATE'));
        }

        // Invalid format
        if (response.statusCode === 400) {
            return callback(createError('Invalid URL format', 'EINVFORMAT'));
        }

        // Everything other than 201 is unknown
        if (response.statusCode !== 201) {
            return callback(createError('Unknown error: ' + response.statusCode + ' - ' + response.body, 'EUNKNOWN'));
        }

        callback(null, {
            name: name,
            url: url
        });
    });
}

module.exports = register;
