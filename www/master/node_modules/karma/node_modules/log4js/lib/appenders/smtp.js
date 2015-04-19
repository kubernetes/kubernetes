"use strict";
var layouts = require("../layouts")
, mailer = require("nodemailer")
, os = require('os')
, async = require('async')
, unsentCount = 0
, shutdownTimeout;

/**
* SMTP Appender. Sends logging events using SMTP protocol. 
* It can either send an email on each event or group several 
* logging events gathered during specified interval.
*
* @param config appender configuration data
*    config.sendInterval time between log emails (in seconds), if 0
*    then every event sends an email
*    config.shutdownTimeout time to give up remaining emails (in seconds; defaults to 5).
* @param layout a function that takes a logevent and returns a string (defaults to basicLayout).
*/
function smtpAppender(config, layout) {
	layout = layout || layouts.basicLayout;
	var subjectLayout = layouts.messagePassThroughLayout;
	var sendInterval = config.sendInterval*1000 || 0;
	
	var logEventBuffer = [];
	var sendTimer;
	
	shutdownTimeout = ('shutdownTimeout' in config ? config.shutdownTimeout : 5) * 1000;
	
	function sendBuffer() {
    if (logEventBuffer.length > 0) {
		
      var transport = mailer.createTransport(config.SMTP);
      var firstEvent = logEventBuffer[0];
      var body = "";
      var count = logEventBuffer.length;
      while (logEventBuffer.length > 0) {
        body += layout(logEventBuffer.shift(), config.timezoneOffset) + "\n";
      }

      var msg = {
        to: config.recipients,
        subject: config.subject || subjectLayout(firstEvent),
        headers: { "Hostname": os.hostname() }
      };
	  
      if (!config.html) {
	msg.text = body;
      } else {
    	msg.html = body;
      }
      
      if (config.sender) {
        msg.from = config.sender;
      }
      transport.sendMail(msg, function(error, success) {
        if (error) {
          console.error("log4js.smtpAppender - Error happened", error);
        }
        transport.close();
        unsentCount -= count;
      });
    }
	}
	
	function scheduleSend() {
		if (!sendTimer) {
			sendTimer = setTimeout(function() {
				sendTimer = null; 
				sendBuffer();
			}, sendInterval);
    }
	}
	
	return function(loggingEvent) {
		unsentCount++;
		logEventBuffer.push(loggingEvent);
		if (sendInterval > 0) {
			scheduleSend();
		} else {
			sendBuffer();
    }
	};
}

function configure(config) {
	var layout;
	if (config.layout) {
		layout = layouts.layout(config.layout.type, config.layout);
	}
	return smtpAppender(config, layout);
}

function shutdown(cb) {
	if (shutdownTimeout > 0) {
		setTimeout(function() { unsentCount = 0; }, shutdownTimeout);
	}
	async.whilst(function() {
		return unsentCount > 0;
	}, function(done) {
		setTimeout(done, 100);
	}, cb);
}

exports.name = "smtp";
exports.appender = smtpAppender;
exports.configure = configure;
exports.shutdown = shutdown;

