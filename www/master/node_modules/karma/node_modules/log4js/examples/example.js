var log4js = require('../lib/log4js');
//log the cheese logger messages to a file, and the console ones as well.
log4js.configure({
    appenders: [
        {
            type: "file",
            filename: "cheese.log",
            category: [ 'cheese','console' ]
        },
        {
            type: "console"
        }
    ],
    replaceConsole: true
});

//to add an appender programmatically, and without clearing other appenders
//loadAppender is only necessary if you haven't already configured an appender of this type
log4js.loadAppender('file');
log4js.addAppender(log4js.appenders.file('pants.log'), 'pants');
//a custom logger outside of the log4js/lib/appenders directory can be accessed like so
//log4js.loadAppender('what/you/would/put/in/require');
//log4js.addAppender(log4js.appenders['what/you/would/put/in/require'](args));
//or through configure as:
//log4js.configure({
//  appenders: [ { type: 'what/you/would/put/in/require', otherArgs: 'blah' } ]
//});

var logger = log4js.getLogger('cheese');
//only errors and above get logged.
//you can also set this log level in the config object
//via the levels field.
logger.setLevel('ERROR');

//console logging methods have been replaced with log4js ones.
//so this will get coloured output on console, and appear in cheese.log
console.error("AAArgh! Something went wrong", { some: "otherObject", useful_for: "debug purposes" });
console.log("This should appear as info output");

//these will not appear (logging level beneath error)
logger.trace('Entering cheese testing');
logger.debug('Got cheese.');
logger.info('Cheese is Gouda.');
logger.log('Something funny about cheese.');
logger.warn('Cheese is quite smelly.');
//these end up on the console and in cheese.log
logger.error('Cheese %s is too ripe!', "gouda");
logger.fatal('Cheese was breeding ground for listeria.');

//these don't end up in cheese.log, but will appear on the console
var anotherLogger = log4js.getLogger('another');
anotherLogger.debug("Just checking");

//one for pants.log
//will also go to console, since that's configured for all categories
var pantsLog = log4js.getLogger('pants');
pantsLog.debug("Something for pants");



