var fs = require('graceful-fs');
var path = require('path');
var Logger = require('bower-logger');
var Resolver = require('../../../lib/core/resolvers/Resolver');
var defaultConfig = require('../../../lib/config');

var resolver = new Resolver({ source: 'foo' }, defaultConfig(), new Logger());
resolver._createTempDir()
.then(function (dir) {
    // Need to write something to prevent tmp to automatically
    // remove the temp dir (it removes if empty)
    fs.writeFileSync(path.join(dir, 'some_file'), 'foo');
})
.done();
