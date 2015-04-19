var cp = require('child_process');
var path = require('path');
var Q = require('q');
var mout = require('mout');
var which = require('which');
var PThrottler = require('p-throttler');
var createError = require('./createError');

// The concurrency limit here is kind of magic. You don't really gain a lot from
// having a large number of commands spawned at once, so it isn't super
// important for this number to be large. Reports have shown that much more than 5
// or 10 cause issues for corporate networks, private repos or situations where
// internet bandwidth is limited. We're running with a concurrency of 5 until
// 1.4.X is released, at which time we'll move to what was discussed in #1262
// https://github.com/bower/bower/pull/1262
var throttler = new PThrottler(5);

var winBatchExtensions;
var winWhichCache;
var isWin = process.platform === 'win32';

if (isWin) {
    winBatchExtensions = ['.bat', '.cmd'];
    winWhichCache = {};
}

function getWindowsCommand(command) {
    var fullCommand;
    var extension;

    // Do we got the value converted in the cache?
    if (mout.object.hasOwn(winWhichCache, command)) {
        return winWhichCache[command];
    }

    // Use which to retrieve the full command, which puts the extension in the end
    try {
        fullCommand = which.sync(command);
    } catch (err) {
        return winWhichCache[command] = command;
    }

    extension = path.extname(fullCommand).toLowerCase();

    // Does it need to be converted?
    if (winBatchExtensions.indexOf(extension) === -1) {
        return winWhichCache[command] = command;
    }

    return winWhichCache[command] = fullCommand;
}

// Executes a shell command, buffering the stdout and stderr
// If an error occurs, a meaningful error is generated
// Returns a promise that gets fulfilled if the command succeeds
// or rejected if it fails
function executeCmd(command, args, options) {
    var process;
    var stderr = '';
    var stdout = '';
    var deferred = Q.defer();

    // Windows workaround for .bat and .cmd files, see #626
    if (isWin) {
        command = getWindowsCommand(command);
    }

    // Buffer output, reporting progress
    process = cp.spawn(command, args, options);
    process.stdout.on('data', function (data) {
        data = data.toString();
        deferred.notify(data);
        stdout += data;
    });
    process.stderr.on('data', function (data) {
        data = data.toString();
        deferred.notify(data);
        stderr += data;
    });

    // If there is an error spawning the command, reject the promise
    process.on('error', function (error) {
        return deferred.reject(error);
    });

    // Listen to the close event instead of exit
    // They are similar but close ensures that streams are flushed
    process.on('close', function (code) {
        var fullCommand;
        var error;

        if (code) {
            // Generate the full command to be presented in the error message
            if (!Array.isArray(args)) {
                args = [];
            }

            fullCommand = command;
            fullCommand += args.length ? ' ' + args.join(' ') : '';

            // Build the error instance
            error = createError('Failed to execute "' + fullCommand + '", exit code of #' + code + '\n' + stderr, 'ECMDERR', {
                details: stderr,
                exitCode: code
            });

            return deferred.reject(error);
        }

        return deferred.resolve([stdout, stderr]);
    });

    return deferred.promise;
}

function cmd(command, args, options) {
    return throttler.enqueue(executeCmd.bind(null, command, args, options));
}

module.exports = cmd;
