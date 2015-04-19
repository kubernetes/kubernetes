var log4js = require('../lib/log4js')
, log
, i = 0;
log4js.configure({
  "appenders": [
      {
          type: "console"
        , category: "console"
      },
      {
          "type": "file",
          "filename": "tmp-test.log",
          "maxLogSize": 1024,
          "backups": 3,
          "category": "test"
      }
  ]
});
log = log4js.getLogger("test");

function doTheLogging(x) {
    log.info("Logging something %d", x);
}

for ( ; i < 5000; i++) {
    doTheLogging(i);
}