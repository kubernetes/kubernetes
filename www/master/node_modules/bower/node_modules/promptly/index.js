'use strict';

var read = require('read');
var promptly = module.exports;

promptly.prompt = function (message, opts, fn) {
    // Arguments parsing
    if (typeof opts === 'function') {
        fn = opts;
        opts = {};
    } else if (!opts) {
        opts = {};
    }

    if (opts.trim === undefined) {
        opts.trim = true;
    }
    if (opts.retry === undefined) {
        opts.retry = true;
    }

    // Setup read's options
    var readOpts = {
        prompt: message,
        input: opts.input || process.stdin,
        output: opts.output || process.stdout,
        silent: opts.silent
    };

    // Use readline question
    read(readOpts, function (err, data) {
        // Ignore the error attribute
        // It is set on SIGINT or if timeout reached (we are not using timeout)
        if (err) {
            return;
        }

        // Trim?
        if (opts.trim) {
            data = data.trim();
        }

        // Mandatory?
        if (opts['default'] == null && !data) {
            return promptly.prompt(message, opts, fn);
        } else {
            data = data || opts['default'];
        }

        // Validator verification
        if (opts.validator) {
            if (!Array.isArray(opts.validator)) {
                opts.validator = [opts.validator];
            }

            var x;
            var length = opts.validator.length;

            for (x = 0; x < length; x += 1) {
                try {
                    data = opts.validator[x](data);
                } catch (e) {
                    // Retry automatically if the retry option is enabled
                    if (opts.retry) {
                        if (e.message) {
                            readOpts.output.write(e.message + '\n');
                        }

                        return promptly.prompt(message, opts, fn);
                    }

                    e.retry = promptly.prompt.bind(promptly, message, opts, fn);

                    return fn(e);
                }
            }
        }

        // Everything ok
        fn(null, data);
    });
};

promptly.password = function (message, opts, fn) {
    // Arguments parsing
    if (typeof opts === 'function') {
        fn = opts;
        opts = {};
    } else {
        opts = opts || {};
    }

    // Set default options
    if (opts.silent === undefined) {
        opts.silent = true;
    }
    if (opts.trim === undefined) {
        opts.trim = false;
    }
    if (opts['default'] === undefined) {
        opts['default'] = '';
    }

    // Use prompt()
    promptly.prompt(message, opts, fn);
};

promptly.confirm = function (message, opts, fn) {
    // Arguments parsing
    if (typeof opts === 'function') {
        fn = opts;
        opts = {};
    } else if (!opts) {
        opts = {};
    }

    opts.validator = opts.validator || [];
    if (!Array.isArray(opts.validator)) {
        opts.validator = [opts.validator];
    }

    // Push the validator that will coerse boolean values
    var validator = function (value) {
        if (typeof value === 'string') {
            value = value.toLowerCase();
        }

        switch (value) {
        case 'y':
        case 'yes':
        case '1':
        case true:
            return true;
        case 'n':
        case 'no':
        case '0':
        case false:
            return false;
        }

        throw new Error();
    };
    opts.validator.push(validator);

    // Use choose() with true, false
    promptly.choose(message, [true, false], opts, fn);
};

promptly.choose = function (message, choices, opts, fn) {
    // Arguments parsing
    if (typeof opts === 'function') {
        fn = opts;
        opts = {};
    } else if (!opts) {
        opts = {};
    }

    opts.validator = opts.validator || [];
    if (!Array.isArray(opts.validator)) {
        opts.validator = [opts.validator];
    }

    // Push the choice validator
    var nrChoices = choices.length;
    var validator = function (value) {
        var x;

        for (x = 0; x < nrChoices; x++) {
            if (choices[x] == value) {
                return choices[x];
            }
        }

        throw new Error('Invalid choice: ' + value);
    };
    opts.validator.push(validator);

    // Use prompt()
    promptly.prompt(message, opts, fn);
};