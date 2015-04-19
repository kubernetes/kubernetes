/**
 * run this, then "ab -c 10 -n 100 localhost:4444/" to test (in
 * another shell)
 */
var log4js = require('../lib/log4js');
log4js.configure({
      appenders: [
        { type: 'file', filename: 'cheese.log', category: 'cheese' },
        { type: 'console'}
  ]
});

var logger = log4js.getLogger('cheese');
logger.setLevel('INFO');

var http=require('http');

var server = http.createServer(function(request, response){
    response.writeHead(200, {'Content-Type': 'text/plain'});
    var rd = Math.random() * 50;
    logger.info("hello " + rd);
    response.write('hello ');
    if (Math.floor(rd) == 30){
        log4js.shutdown(function() { process.exit(1); });
    }
    response.end();
}).listen(4444);
