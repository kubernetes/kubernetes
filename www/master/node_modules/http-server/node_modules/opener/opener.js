#!/usr/bin/env node

"use strict";

var childProcess = require("child_process");

function opener(args, options, callback) {
    // http://stackoverflow.com/q/1480971/3191, but see below for Windows.
    var command = process.platform === "win32" ? "cmd" :
                  process.platform === "darwin" ? "open" :
                  "xdg-open";

    if (typeof args === "string") {
        args = [args];
    }

    if (typeof options === "function") {
        callback = options;
        options = {};
    }

    if (options && typeof options === "object" && options.command) {
        if (process.platform === "win32") {
            // *always* use cmd on windows
            args = [options.command].concat(args);
        } else {
            command = options.command;
        }
    }

    if (process.platform === "win32") {
        // On Windows, we really want to use the "start" command. But, the rules regarding arguments with spaces, and
        // escaping them with quotes, can get really arcane. So the easiest way to deal with this is to pass off the
        // responsibility to "cmd /c", which has that logic built in.
        //
        // Furthermore, if "cmd /c" double-quoted the first parameter, then "start" will interpret it as a window title,
        // so we need to add a dummy empty-string window title: http://stackoverflow.com/a/154090/3191
        args = ["/c", "start", '""'].concat(args);
    }

    childProcess.execFile(command, args, options, callback);
}

// Export `opener` for programmatic access.
// You might use this to e.g. open a website: `opener("http://google.com")`
module.exports = opener;

// If we're being called from the command line, just execute, using the command-line arguments.
if (require.main && require.main.id === module.id) {
    opener(process.argv.slice(2), function (error) {
        if (error) {
            throw error;
        }
    });
}
