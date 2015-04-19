var log4js = require('../lib/log4js')
, logger;

log4js.configure({
  appenders: [
    { type: "file", maxLogSize: 200, backups: 3, filename: "test.log" }
]
});

logger = log4js.getLogger("testing");

logger.info("this should be appended");
