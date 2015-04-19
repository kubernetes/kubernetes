// Load modules

var Http = require('http');
var NodeUtil = require('util');
var Hoek = require('hoek');


// Declare internals

var internals = {};


exports = module.exports = internals.Boom = function (/* (new Error) or (code, message) */) {

    var self = this;

    Hoek.assert(this.constructor === internals.Boom, 'Error must be instantiated using new');

    Error.call(this);
    this.isBoom = true;

    this.response = {
        code: 0,
        payload: {},
        headers: {}
        // type: 'content-type'
    };

    if (arguments[0] instanceof Error) {

        // Error

        var error = arguments[0];

        this.data = error;
        this.response.code = error.code || 500;
        if (error.message) {
            this.message = error.message;
        }
    }
    else {

        // code, message

        var code = arguments[0];
        var message = arguments[1];

        Hoek.assert(!isNaN(parseFloat(code)) && isFinite(code) && code >= 400, 'First argument must be a number (400+)');

        this.response.code = code;
        if (message) {
            this.message = message;
        }
    }

    // Response format

    this.reformat();

    return this;
};

NodeUtil.inherits(internals.Boom, Error);


internals.Boom.prototype.reformat = function () {

    this.response.payload.code = this.response.code;
    this.response.payload.error = Http.STATUS_CODES[this.response.code] || 'Unknown';
    if (this.message) {
        this.response.payload.message = Hoek.escapeHtml(this.message);         // Prevent XSS from error message
    }
};


// Utilities

internals.Boom.badRequest = function (message) {

    return new internals.Boom(400, message);
};


internals.Boom.unauthorized = function (message, scheme, attributes) {          // Or function (message, wwwAuthenticate[])

    var err = new internals.Boom(401, message);

    if (!scheme) {
        return err;
    }

    var wwwAuthenticate = '';

    if (typeof scheme === 'string') {

        // function (message, scheme, attributes)

        wwwAuthenticate = scheme;
        if (attributes) {
            var names = Object.keys(attributes);
            for (var i = 0, il = names.length; i < il; ++i) {
                if (i) {
                    wwwAuthenticate += ',';
                }

                var value = attributes[names[i]];
                if (value === null ||
                    value === undefined) {              // Value can be zero

                    value = '';
                }
                wwwAuthenticate += ' ' + names[i] + '="' + Hoek.escapeHeaderAttribute(value.toString()) + '"';
            }
        }

        if (message) {
            if (attributes) {
                wwwAuthenticate += ',';
            }
            wwwAuthenticate += ' error="' + Hoek.escapeHeaderAttribute(message) + '"';
        }
        else {
            err.isMissing = true;
        }
    }
    else {

        // function (message, wwwAuthenticate[])

        var wwwArray = scheme;
        for (var i = 0, il = wwwArray.length; i < il; ++i) {
            if (i) {
                wwwAuthenticate += ', ';
            }

            wwwAuthenticate += wwwArray[i];
        }
    }

    err.response.headers['WWW-Authenticate'] = wwwAuthenticate;

    return err;
};


internals.Boom.clientTimeout = function (message) {

    return new internals.Boom(408, message);
};


internals.Boom.serverTimeout = function (message) {

    return new internals.Boom(503, message);
};


internals.Boom.forbidden = function (message) {

    return new internals.Boom(403, message);
};


internals.Boom.notFound = function (message) {

    return new internals.Boom(404, message);
};


internals.Boom.internal = function (message, data) {

    var err = new internals.Boom(500, message);

    if (data && data.stack) {
        err.trace = data.stack.split('\n');
        err.outterTrace = Hoek.displayStack(1);
    }
    else {
        err.trace = Hoek.displayStack(1);
    }

    err.data = data;
    err.response.payload.message = 'An internal server error occurred';                     // Hide actual error from user

    return err;
};


internals.Boom.passThrough = function (code, payload, contentType, headers) {

    var err = new internals.Boom(500, 'Pass-through');                                      // 500 code is only used to initialize

    err.data = {
        code: code,
        payload: payload,
        type: contentType
    };

    err.response.code = code;
    err.response.type = contentType;
    err.response.headers = headers;
    err.response.payload = payload;

    return err;
};


