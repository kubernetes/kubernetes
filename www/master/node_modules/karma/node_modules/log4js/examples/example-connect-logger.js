//The connect/express logger was added to log4js by danbell. This allows connect/express servers to log using log4js.
//https://github.com/nomiddlename/log4js-node/wiki/Connect-Logger

// load modules
var log4js = require('log4js');
var express = require("express");
var app = express();

//config
log4js.configure({
	appenders: [
		{ type: 'console' },
		{ type: 'file', filename: 'logs/log4jsconnect.log', category: 'log4jslog' }
	]
});

//define logger
var logger = log4js.getLogger('log4jslog');

// set at which time msg is logged print like: only on error & above
// logger.setLevel('ERROR');

//express app
app.configure(function() {
	app.use(express.favicon(''));
	// app.use(log4js.connectLogger(logger, { level: log4js.levels.INFO }));
	// app.use(log4js.connectLogger(logger, { level: 'auto', format: ':method :url :status' }));

	//### AUTO LEVEL DETECTION
	//http responses 3xx, level = WARN
	//http responses 4xx & 5xx, level = ERROR
	//else.level = INFO
	app.use(log4js.connectLogger(logger, { level: 'auto' }));
});

//route
app.get('/', function(req,res) {
	res.send('hello world');
});

//start app
app.listen(5000);

console.log('server runing at localhost:5000');
console.log('Simulation of normal response: goto localhost:5000');
console.log('Simulation of error response: goto localhost:5000/xxx');
