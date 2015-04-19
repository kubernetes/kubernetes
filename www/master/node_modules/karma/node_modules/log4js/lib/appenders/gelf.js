"use strict";
var zlib = require('zlib');
var layouts = require('../layouts');
var levels = require('../levels');
var dgram = require('dgram');
var util = require('util');
var debug = require('../debug')('GELF Appender');

var LOG_EMERG=0;    // system is unusable
var LOG_ALERT=1;    // action must be taken immediately
var LOG_CRIT=2;     // critical conditions
var LOG_ERR=3;      // error conditions
var LOG_ERROR=3;    // because people WILL typo
var LOG_WARNING=4;  // warning conditions
var LOG_NOTICE=5;   // normal, but significant, condition
var LOG_INFO=6;     // informational message
var LOG_DEBUG=7;    // debug-level message

var levelMapping = {};
levelMapping[levels.ALL] = LOG_DEBUG;
levelMapping[levels.TRACE] = LOG_DEBUG;
levelMapping[levels.DEBUG] = LOG_DEBUG;
levelMapping[levels.INFO] = LOG_INFO;
levelMapping[levels.WARN] = LOG_WARNING;
levelMapping[levels.ERROR] = LOG_ERR;
levelMapping[levels.FATAL] = LOG_CRIT;

/**
 * GELF appender that supports sending UDP packets to a GELF compatible server such as Graylog
 *
 * @param layout a function that takes a logevent and returns a string (defaults to none).
 * @param host - host to which to send logs (default:localhost)
 * @param port - port at which to send logs to (default:12201)
 * @param hostname - hostname of the current host (default:os hostname)
 * @param facility - facility to log to (default:nodejs-server)
 */
function gelfAppender (layout, host, port, hostname, facility) {
  var config, customFields;
  if (typeof(host) === 'object') {
    config = host;
    host = config.host;
    port = config.port;
    hostname = config.hostname;
    facility = config.facility;
    customFields = config.customFields;
  }
  
  host = host || 'localhost';
  port = port || 12201;
  hostname = hostname || require('os').hostname();
  layout = layout || layouts.messagePassThroughLayout;

  var defaultCustomFields = customFields || {};

  if(facility) {
    defaultCustomFields['_facility'] = facility;
  }

  var client = dgram.createSocket("udp4");
  
  process.on('exit', function() {
    if (client) client.close();
  });

  /**
   * Add custom fields (start with underscore ) 
   * - if the first object passed to the logger contains 'GELF' field, 
   *   copy the underscore fields to the message
   * @param loggingEvent
   * @param msg
   */
  function addCustomFields(loggingEvent, msg){

    /* append defaultCustomFields firsts */
    Object.keys(defaultCustomFields).forEach(function(key) {
      // skip _id field for graylog2, skip keys not starts with UNDERSCORE
      if (key.match(/^_/) && key !== "_id") { 
        msg[key] = defaultCustomFields[key];
      }
    });

    /* append custom fields per message */
    var data = loggingEvent.data;
    if (!Array.isArray(data) || data.length === 0) return;
    var firstData = data[0];
    
    if (!firstData.GELF) return; // identify with GELF field defined
    Object.keys(firstData).forEach(function(key) {
      // skip _id field for graylog2, skip keys not starts with UNDERSCORE
      if (key.match(/^_/) || key !== "_id") { 
        msg[key] = firstData[key];
      }
    });
    
    /* the custom field object should be removed, so it will not be looged by the later appenders */
    loggingEvent.data.shift(); 
  }
 
  function preparePacket(loggingEvent) {
    var msg = {};
    addCustomFields(loggingEvent, msg);
    msg.short_message = layout(loggingEvent);
    
    msg.version="1.1";
    msg.timestamp = msg.timestamp || new Date().getTime() / 1000; // log should use millisecond 
    msg.host = hostname;
    msg.level = levelMapping[loggingEvent.level || levels.DEBUG];
    return msg;
  }
  
  function sendPacket(packet) {
    try {
      client.send(packet, 0, packet.length, port, host);
    } catch(e) {}
  }

  return function(loggingEvent) {
    var message = preparePacket(loggingEvent);
    zlib.gzip(new Buffer(JSON.stringify(message)), function(err, packet) {
      if (err) {
        console.error(err.stack);
      } else {
        if (packet.length > 8192) {
          debug("Message packet length (" + packet.length + ") is larger than 8k. Not sending");
        } else {
          sendPacket(packet);
        }
      }
    });
  };
}

function configure(config) {
  var layout;
  if (config.layout) {
    layout = layouts.layout(config.layout.type, config.layout);
  }
  return gelfAppender(layout, config);
}

exports.appender = gelfAppender;
exports.configure = configure;
