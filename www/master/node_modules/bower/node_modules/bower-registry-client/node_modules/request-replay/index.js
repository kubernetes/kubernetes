'use strict';

var retry = require('retry');

var errorCodes = [
    'EADDRINFO',
    'ETIMEDOUT',
    'ECONNRESET',
    'ESOCKETTIMEDOUT'
];

function mixIn(dst, src) {
    var key;

    for (key in src) {
        dst[key] = src[key];
    }

    return dst;
}

function requestReplay(request, options) {
    var originalEmit = request.emit;
    var operation;
    var timeout;
    var retrying = false;
    var attempts = 0;

    // Default options
    options = mixIn({
        errorCodes: errorCodes,
        retries: 5,
        factor: 2,
        minTimeout: 1000,
        maxTimeout: 35000,
        randomize: true
    }, options || {});

    // Init retry
    operation = retry.operation(options);
    operation.attempt(function () {
        retrying = false;

        if (attempts) {
            request.init();
            request.start();
        }

        attempts++;
    });

    // Increase maxListeners because start() adds a new listener each time
    request._maxListeners += options.retries + 1;

    // Monkey patch emit to catch errors and retry
    request.emit = function (name, error) {
        // If name is replay, pass-through
        if (name === 'replay') {
            return originalEmit.apply(this, arguments);
        }

        // Do not emit anything if we are retrying
        if (retrying) {
            return;
        }

        // If not a retry error code, pass-through
        if (name !== 'error' || options.errorCodes.indexOf(error.code) === -1) {
            return originalEmit.apply(this, arguments);
        }

        timeout = operation._timeouts[0];

        // Retry
        if (operation.retry(error)) {
            retrying = true;
            request.abort();
            request._aborted = false;
            this.emit('replay', {
                number: attempts - 1,
                error: error,
                delay: timeout
            });
            return 0;
        }

        // No more retries available, error out
        error.replays = attempts - 1;
        return originalEmit.apply(this, arguments);
    };

    return request;
}

module.exports = requestReplay;
