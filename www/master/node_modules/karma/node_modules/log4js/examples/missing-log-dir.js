var log4js = require('../lib/log4js');
log4js.configure({
  appenders: [
    { type: "file", filename: "madeup/path/to/file.log" }
  ],
  replaceConsole: true
});

logger = log4js.getLogger();

logger.info("Does this work?");
