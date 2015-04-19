var fs = require('fs');

var pkg = JSON.parse(fs.readFileSync(__dirname + '/../package.json').toString());

exports.VERSION = pkg.version;

exports.DEFAULT_PORT = 9876;
exports.DEFAULT_HOSTNAME = 'localhost';

// log levels
exports.LOG_DISABLE = 'OFF';
exports.LOG_ERROR   = 'ERROR';
exports.LOG_WARN    = 'WARN';
exports.LOG_INFO    = 'INFO';
exports.LOG_DEBUG   = 'DEBUG';

// Default patterns for the pattern layout.
exports.COLOR_PATTERN = '%[%p [%c]: %]%m';
exports.NO_COLOR_PATTERN = '%p [%c]: %m';

// Default console appender
exports.CONSOLE_APPENDER = {
  type: 'console',
  layout: {
    type: 'pattern',
    pattern: exports.COLOR_PATTERN
  }
};

exports.EXIT_CODE = '\x1FEXIT';
