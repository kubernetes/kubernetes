//Note that loggly appender needs node-loggly to work.
//If you haven't got node-loggly installed, you'll get cryptic
//"cannot find module" errors when using the loggly appender
var log4js = require('../lib/log4js');

log4js.configure({
  "appenders": [
    {
      type: "console",
      category: "test"
    },
    {
      "type"     : "loggly",
      "token"    : "12345678901234567890",
      "subdomain": "your-subdomain",
      "tags"     : ["test"],
      "category" : "loggly"
    }
  ]
});

var logger = log4js.getLogger("loggly");
logger.info("Test log message");
//logger.debug("Test log message");