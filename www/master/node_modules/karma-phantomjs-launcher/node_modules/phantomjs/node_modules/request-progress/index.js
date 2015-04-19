'use strict';

var throttle = require('throttleit');

function requestProgress(request, options) {
    var reporter;
    var onResponse;
    var delayTimer;
    var delayCompleted;
    var totalSize;
    var previousReceivedSize;
    var receivedSize = 0;
    var state = {};

    options = options || {};
    options.throttle = options.throttle == null ? 1000 : options.throttle;
    options.delay = options.delay || 0;

    // Throttle the progress report function
    reporter = throttle(function () {
        // If the received size is the same, do not report
        if (previousReceivedSize === receivedSize) {
            return;
        }

        // Update received
        previousReceivedSize = receivedSize;
        state.received = receivedSize;

        // Update percentage
        // Note that the totalSize might available
        state.percent = totalSize ? Math.round(receivedSize / totalSize * 100) : null;

        request.emit('progress', state);
    }, options.throttle);

    // On response handler
    onResponse = function (response) {
        totalSize = Number(response.headers['content-length']);
        receivedSize = 0;

        // Note that the totalSize might available
        state.total = totalSize || null;

        // Delay the progress report
        delayCompleted = false;
        delayTimer = setTimeout(function () {
            delayCompleted = true;
            delayTimer = null;
        }, options.delay);
    };

    request
    .on('request', function () {
        receivedSize = 0;
    })
    .on('response', onResponse)
    .on('data', function (data) {
        receivedSize += data.length;

        if (delayCompleted) {
            reporter();
        }
    })
    .on('end', function () {
        if (delayTimer) {
            clearTimeout(delayTimer);
            delayTimer = null;
        }
    });

    // If we already got a response, call the on response handler
    if (request.response) {
        onResponse(request.response);
    }

    return request;
}

module.exports = requestProgress;
