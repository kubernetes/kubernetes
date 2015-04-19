// Load modules


// Declare internals

var internals = {};


internals.stringify = function (obj, prefix) {

    if (Buffer.isBuffer(obj)) {
        obj = obj.toString();
    }
    else if (obj instanceof Date) {
        obj = obj.toISOString();
    }

    if (typeof obj === 'string' ||
        typeof obj === 'number' ||
        typeof obj === 'boolean') {

        return [prefix + '=' + encodeURIComponent(obj)];
    }

    if (obj === null) {
        return [prefix];
    }

    var values = [];

    for (var key in obj) {
        if (obj.hasOwnProperty(key)) {
            values = values.concat(internals.stringify(obj[key], prefix + '[' + encodeURIComponent(key) + ']'));
        }
    }

    return values;
};


module.exports = function (obj) {

    var keys = [];

    for (var key in obj) {
        if (obj.hasOwnProperty(key)) {
            keys = keys.concat(internals.stringify(obj[key], encodeURIComponent(key)));
        }
    }

    return keys.join('&');
};
