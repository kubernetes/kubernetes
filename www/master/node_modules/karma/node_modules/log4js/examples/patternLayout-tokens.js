var log4js = require('./lib/log4js');

var config = {
    "appenders": [
      {
        "type": "console",
        "layout": {
          "type": "pattern",
          "pattern": "%[%r (%x{pid}) %p %c -%] %m%n",
          "tokens": {
            "pid" : function() { return process.pid; }
          }
        }
      }
    ]
  };

log4js.configure(config, {});

var logger = log4js.getLogger("app");
logger.info("Test log message");