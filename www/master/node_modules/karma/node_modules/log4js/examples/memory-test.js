var log4js = require('./lib/log4js')
, logger
, usage
, i;

log4js.configure(
    {
        appenders: [
            {
                category: "memory-test"
              , type: "file"
              , filename: "memory-test.log"
            },
            {
                type: "console"
              , category: "memory-usage"
            },
            {
                type: "file"
              , filename: "memory-usage.log"
              , category: "memory-usage"
              , layout: {
                  type: "messagePassThrough"
              }
            }
        ]
    }
);
logger = log4js.getLogger("memory-test");
usage = log4js.getLogger("memory-usage");

for (i=0; i < 1000000; i++) {
    if ( (i % 5000) === 0) {
        usage.info("%d %d", i, process.memoryUsage().rss);
    }
    logger.info("Doing something.");
}
