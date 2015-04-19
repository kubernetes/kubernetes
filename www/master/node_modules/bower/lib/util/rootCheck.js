/*jshint multistr:true*/
'use strict';
var isRoot = require('is-root');
var createError = require('./createError');

var renderer;

function rootCheck(options, config) {
    var errorMsg;

    // Allow running the command as root
    if (options.allowRoot || config.allowRoot) {
        return;
    }

    errorMsg = 'Since bower is a user command, there is no need to execute it with \
superuser permissions.\nIf you\'re having permission errors when using bower without \
sudo, please spend a few minutes learning more about how your system should work and \
make any necessary repairs.\n\n\
http://www.joyent.com/blog/installing-node-and-npm\n\
https://gist.github.com/isaacs/579814\n\n\
You can however run a command with sudo using --allow-root option';

    if (isRoot()) {
        var cli = require('./cli');
        renderer = cli.getRenderer('', false, config);
        renderer.error(createError('Cannot be run with sudo', 'ESUDO', { details : errorMsg }));
        process.exit(1);
    }
}

module.exports = rootCheck;
