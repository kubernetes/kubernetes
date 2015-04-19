// Load modules

var Fs = require('fs');
var Escape = require('./escape');


// Declare internals

var internals = {};


// Clone object or array

exports.clone = function (obj, seen) {

    if (typeof obj !== 'object' ||
        obj === null) {

        return obj;
    }

    seen = seen || { orig: [], copy: [] };

    var lookup = seen.orig.indexOf(obj);
    if (lookup !== -1) {
        return seen.copy[lookup];
    }

    var newObj = (obj instanceof Array) ? [] : {};

    seen.orig.push(obj);
    seen.copy.push(newObj);

    for (var i in obj) {
        if (obj.hasOwnProperty(i)) {
            if (obj[i] instanceof Buffer) {
                newObj[i] = new Buffer(obj[i]);
            }
            else if (obj[i] instanceof Date) {
                newObj[i] = new Date(obj[i].getTime());
            }
            else if (obj[i] instanceof RegExp) {
                var flags = '' + (obj[i].global ? 'g' : '') + (obj[i].ignoreCase ? 'i' : '') + (obj[i].multiline ? 'm' : '');
                newObj[i] = new RegExp(obj[i].source, flags);
            }
            else {
                newObj[i] = exports.clone(obj[i], seen);
            }
        }
    }

    return newObj;
};


// Merge all the properties of source into target, source wins in conflic, and by default null and undefined from source are applied

exports.merge = function (target, source, isNullOverride /* = true */, isMergeArrays /* = true */) {

    exports.assert(target && typeof target == 'object', 'Invalid target value: must be an object');
    exports.assert(source === null || source === undefined || typeof source === 'object', 'Invalid source value: must be null, undefined, or an object');

    if (!source) {
        return target;
    }

    if (source instanceof Array) {
        exports.assert(target instanceof Array, 'Cannot merge array onto an object');
        if (isMergeArrays === false) {                                                  // isMergeArrays defaults to true
            target.length = 0;                                                          // Must not change target assignment
        }

        for (var i = 0, il = source.length; i < il; ++i) {
            target.push(source[i]);
        }

        return target;
    }

    var keys = Object.keys(source);
    for (var k = 0, kl = keys.length; k < kl; ++k) {
        var key = keys[k];
        var value = source[key];
        if (value &&
            typeof value === 'object') {

            if (!target[key] ||
                typeof target[key] !== 'object') {

                target[key] = exports.clone(value);
            }
            else {
                exports.merge(target[key], source[key], isNullOverride, isMergeArrays);
            }
        }
        else {
            if (value !== null && value !== undefined) {            // Explicit to preserve empty strings
                target[key] = value;
            }
            else if (isNullOverride !== false) {                    // Defaults to true
                target[key] = value;
            }
        }
    }

    return target;
};


// Apply options to a copy of the defaults

exports.applyToDefaults = function (defaults, options) {

    exports.assert(defaults && typeof defaults == 'object', 'Invalid defaults value: must be an object');
    exports.assert(!options || options === true || typeof options === 'object', 'Invalid options value: must be true, falsy or an object');

    if (!options) {                                                 // If no options, return null
        return null;
    }

    var copy = exports.clone(defaults);

    if (options === true) {                                         // If options is set to true, use defaults
        return copy;
    }

    return exports.merge(copy, options, false, false);
};


// Remove duplicate items from array

exports.unique = function (array, key) {

    var index = {};
    var result = [];

    for (var i = 0, il = array.length; i < il; ++i) {
        var id = (key ? array[i][key] : array[i]);
        if (index[id] !== true) {

            result.push(array[i]);
            index[id] = true;
        }
    }

    return result;
};


// Convert array into object

exports.mapToObject = function (array, key) {

    if (!array) {
        return null;
    }

    var obj = {};
    for (var i = 0, il = array.length; i < il; ++i) {
        if (key) {
            if (array[i][key]) {
                obj[array[i][key]] = true;
            }
        }
        else {
            obj[array[i]] = true;
        }
    }

    return obj;
};


// Find the common unique items in two arrays

exports.intersect = function (array1, array2, justFirst) {

    if (!array1 || !array2) {
        return [];
    }

    var common = [];
    var hash = (array1 instanceof Array ? exports.mapToObject(array1) : array1);
    var found = {};
    for (var i = 0, il = array2.length; i < il; ++i) {
        if (hash[array2[i]] && !found[array2[i]]) {
            if (justFirst) {
                return array2[i];
            }

            common.push(array2[i]);
            found[array2[i]] = true;
        }
    }

    return (justFirst ? null : common);
};


// Find which keys are present

exports.matchKeys = function (obj, keys) {

    var matched = [];
    for (var i = 0, il = keys.length; i < il; ++i) {
        if (obj.hasOwnProperty(keys[i])) {
            matched.push(keys[i]);
        }
    }
    return matched;
};


// Flatten array

exports.flatten = function (array, target) {

    var result = target || [];

    for (var i = 0, il = array.length; i < il; ++i) {
        if (Array.isArray(array[i])) {
            exports.flatten(array[i], result);
        }
        else {
            result.push(array[i]);
        }
    }

    return result;
};


// Remove keys

exports.removeKeys = function (object, keys) {

    for (var i = 0, il = keys.length; i < il; i++) {
        delete object[keys[i]];
    }
};


// Convert an object key chain string ('a.b.c') to reference (object[a][b][c])

exports.reach = function (obj, chain) {

    var path = chain.split('.');
    var ref = obj;
    for (var i = 0, il = path.length; i < il; ++i) {
        if (ref) {
            ref = ref[path[i]];
        }
    }

    return ref;
};


// Inherits a selected set of methods from an object, wrapping functions in asynchronous syntax and catching errors

exports.inheritAsync = function (self, obj, keys) {

    keys = keys || null;

    for (var i in obj) {
        if (obj.hasOwnProperty(i)) {
            if (keys instanceof Array &&
                keys.indexOf(i) < 0) {

                continue;
            }

            self.prototype[i] = (function (fn) {

                return function (next) {

                    var result = null;
                    try {
                        result = fn();
                    }
                    catch (err) {
                        return next(err);
                    }

                    return next(null, result);
                };
            })(obj[i]);
        }
    }
};


exports.formatStack = function (stack) {

    var trace = [];
    for (var i = 0, il = stack.length; i < il; ++i) {
        var item = stack[i];
        trace.push([item.getFileName(), item.getLineNumber(), item.getColumnNumber(), item.getFunctionName(), item.isConstructor()]);
    }

    return trace;
};


exports.formatTrace = function (trace) {

    var display = [];

    for (var i = 0, il = trace.length; i < il; ++i) {
        var row = trace[i];
        display.push((row[4] ? 'new ' : '') + row[3] + ' (' + row[0] + ':' + row[1] + ':' + row[2] + ')');
    }

    return display;
};


exports.callStack = function (slice) {

    // http://code.google.com/p/v8/wiki/JavaScriptStackTraceApi

    var v8 = Error.prepareStackTrace;
    Error.prepareStackTrace = function (err, stack) {

        return stack;
    };

    var capture = {};
    Error.captureStackTrace(capture, arguments.callee);
    var stack = capture.stack;

    Error.prepareStackTrace = v8;

    var trace = exports.formatStack(stack);

    if (slice) {
        return trace.slice(slice);
    }

    return trace;
};


exports.displayStack = function (slice) {

    var trace = exports.callStack(slice === undefined ? 1 : slice + 1);

    return exports.formatTrace(trace);
};


exports.abortThrow = false;


exports.abort = function (message, hideStack) {

    if (process.env.NODE_ENV === 'test' || exports.abortThrow === true) {
        throw new Error(message || 'Unknown error');
    }

    var stack = '';
    if (!hideStack) {
        stack = exports.displayStack(1).join('\n\t');
    }
    console.log('ABORT: ' + message + '\n\t' + stack);
    process.exit(1);
};


exports.assert = function (condition /*, msg1, msg2, msg3 */) {

    if (condition) {
        return;
    }

    var msgs = Array.prototype.slice.call(arguments, 1);
    msgs = msgs.map(function (msg) {

        return typeof msg === 'string' ? msg : msg instanceof Error ? msg.message : JSON.stringify(msg);
    });
    throw new Error(msgs.join(' ') || 'Unknown error');
};


exports.loadDirModules = function (path, excludeFiles, target) {      // target(filename, name, capName)

    var exclude = {};
    for (var i = 0, il = excludeFiles.length; i < il; ++i) {
        exclude[excludeFiles[i] + '.js'] = true;
    }

    var files = Fs.readdirSync(path);
    for (i = 0, il = files.length; i < il; ++i) {
        var filename = files[i];
        if (/\.js$/.test(filename) &&
            !exclude[filename]) {

            var name = filename.substr(0, filename.lastIndexOf('.'));
            var capName = name.charAt(0).toUpperCase() + name.substr(1).toLowerCase();

            if (typeof target !== 'function') {
                target[capName] = require(path + '/' + name);
            }
            else {
                target(path + '/' + name, name, capName);
            }
        }
    }
};


exports.rename = function (obj, from, to) {

    obj[to] = obj[from];
    delete obj[from];
};


exports.Timer = function () {

    this.reset();
};


exports.Timer.prototype.reset = function () {

    this.ts = Date.now();
};


exports.Timer.prototype.elapsed = function () {

    return Date.now() - this.ts;
};


// Load and parse package.json process root or given directory

exports.loadPackage = function (dir) {

    var result = {};
    var filepath = (dir || process.env.PWD) + '/package.json';
    if (Fs.existsSync(filepath)) {
        try {
            result = JSON.parse(Fs.readFileSync(filepath));
        }
        catch (e) { }
    }

    return result;
};


// Escape string for Regex construction

exports.escapeRegex = function (string) {

    // Escape ^$.*+-?=!:|\/()[]{},
    return string.replace(/[\^\$\.\*\+\-\?\=\!\:\|\\\/\(\)\[\]\{\}\,]/g, '\\$&');
};


// Return an error as first argument of a callback

exports.toss = function (condition /*, [message], next */) {

    var message = (arguments.length === 3 ? arguments[1] : '');
    var next = (arguments.length === 3 ? arguments[2] : arguments[1]);

    var err = (message instanceof Error ? message : (message ? new Error(message) : (condition instanceof Error ? condition : new Error())));

    if (condition instanceof Error ||
        !condition) {

        return next(err);
    }
};


// Base64url (RFC 4648) encode

exports.base64urlEncode = function (value) {

    return (new Buffer(value, 'binary')).toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/\=/g, '');
};


// Base64url (RFC 4648) decode

exports.base64urlDecode = function (encoded) {

    if (encoded &&
        !encoded.match(/^[\w\-]*$/)) {

        return new Error('Invalid character');
    }

    try {
        return (new Buffer(encoded.replace(/-/g, '+').replace(/:/g, '/'), 'base64')).toString('binary');
    }
    catch (err) {
        return err;
    }
};


// Escape attribute value for use in HTTP header

exports.escapeHeaderAttribute = function (attribute) {

    // Allowed value characters: !#$%&'()*+,-./:;<=>?@[]^_`{|}~ and space, a-z, A-Z, 0-9, \, "

    exports.assert(attribute.match(/^[ \w\!#\$%&'\(\)\*\+,\-\.\/\:;<\=>\?@\[\]\^`\{\|\}~\"\\]*$/), 'Bad attribute value (' + attribute + ')');

    return attribute.replace(/\\/g, '\\\\').replace(/\"/g, '\\"');                             // Escape quotes and slash
};


exports.escapeHtml = function (string) {

    return Escape.escapeHtml(string);
};


exports.escapeJavaScript = function (string) {

    return Escape.escapeJavaScript(string);
};


/*
var event = {
    timestamp: now.getTime(),
    tags: ['tag'],
    data: { some: 'data' }
};
*/

exports.consoleFunc = console.log;

exports.printEvent = function (event) {

    var pad = function (value) {

        return (value < 10 ? '0' : '') + value;
    };

    var now = new Date(event.timestamp);
    var timestring = (now.getYear() - 100).toString() +
        pad(now.getMonth() + 1) +
        pad(now.getDate()) +
        '/' +
        pad(now.getHours()) +
        pad(now.getMinutes()) +
        pad(now.getSeconds()) +
        '.' +
        now.getMilliseconds();

    var data = event.data;
    if (typeof event.data !== 'string') {
        try {
            data = JSON.stringify(event.data);
        }
        catch (e) {
            data = 'JSON Error: ' + e.message;
        }
    }

    var output = timestring + ', ' + event.tags[0] + ', ' + data;
    exports.consoleFunc(output);
};


exports.nextTick = function (callback) {

    return function () {

        var args = arguments;
        process.nextTick(function () {

            callback.apply(null, args);
        });
    };
};
