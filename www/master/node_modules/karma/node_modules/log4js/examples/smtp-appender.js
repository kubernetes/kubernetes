//Note that smtp appender needs nodemailer to work.
//If you haven't got nodemailer installed, you'll get cryptic
//"cannot find module" errors when using the smtp appender
var log4js = require('../lib/log4js')
, log
, logmailer
, i = 0;
log4js.configure({
  "appenders": [
    {
      type: "console",
      category: "test"
    },
    {
      "type": "smtp",
      "recipients": "logfilerecipient@logging.com",
      "sendInterval": 5,
      "transport": "SMTP",
      "SMTP": {
        "host": "smtp.gmail.com",
        "secureConnection": true,
        "port": 465,
        "auth": {
          "user": "someone@gmail",
          "pass": "********************"
        },
        "debug": true
      },
      "category": "mailer"
    }
  ]
});
log = log4js.getLogger("test");
logmailer = log4js.getLogger("mailer");

function doTheLogging(x) {
    log.info("Logging something %d", x);
    logmailer.info("Logging something %d", x);
}

for ( ; i < 500; i++) {
    doTheLogging(i);
}
