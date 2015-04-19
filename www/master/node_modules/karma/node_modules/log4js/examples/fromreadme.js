//remember to change the require to just 'log4js' if you've npm install'ed it
var log4js = require('./lib/log4js');
//by default the console appender is loaded
//log4js.loadAppender('console');
//you'd only need to add the console appender if you
//had previously called log4js.clearAppenders();
//log4js.addAppender(log4js.appenders.console());
log4js.loadAppender('file');
log4js.addAppender(log4js.appenders.file('cheese.log'), 'cheese');

var logger = log4js.getLogger('cheese');
logger.setLevel('ERROR');

logger.trace('Entering cheese testing');
logger.debug('Got cheese.');
logger.info('Cheese is Gouda.');
logger.warn('Cheese is quite smelly.');
logger.error('Cheese is too ripe!');
logger.fatal('Cheese was breeding ground for listeria.');
