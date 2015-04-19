
/*!
 * socket.io-node
 * Copyright(c) 2011 LearnBoost <dev@learnboost.com>
 * MIT Licensed
 */

var vm = require('vm')
  , should = require('should');

/**
 * Generates evn variables for the vm so we can `emulate` a browser.
 * @returns {Object} evn variables
 */

exports.env = function env () {
  var details = {
      location: {
          port: 8080
        , host: 'www.example.org'
        , hostname: 'www.example.org'
        , href: 'http://www.example.org/example/'
        , pathname: '/example/'
        , protocol: 'http:'
        , search: ''
        , hash: ''
      }
    , console: {
        log:   function(){},
        info:  function(){},
        warn:  function(){},
        error: function(){}
      }
    , navigator: {
          userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_7) AppleWebKit'
           + '/534.27 (KHTML, like Gecko) Chrome/12.0.716.0 Safari/534.27'
        , appName: 'socket.io'
        , platform: process.platform
        , appVersion: process.version
    , }
    , name: 'socket.io'
    , innerWidth: 1024
    , innerHeight: 768
    , length: 1
    , outerWidth: 1024
    , outerHeight: 768
    , pageXOffset: 0
    , pageYOffset: 0
    , screenX: 0
    , screenY: 0
    , screenLeft: 0
    , screenTop: 0
    , scrollX: 0
    , scrollY: 0
    , scrollTop: 0
    , scrollLeft: 0
    , screen: {
          width: 0
        , height: 0
      }
  };

  // circular references
  details.window = details.self = details.contentWindow = details;

  // callable methods
  details.Image = details.scrollTo = details.scrollBy = details.scroll = 
  details.resizeTo = details.resizeBy = details.prompt = details.print = 
  details.open = details.moveTo = details.moveBy = details.focus = 
  details.createPopup = details.confirm = details.close = details.blur = 
  details.alert = details.clearTimeout = details.clearInterval = 
  details.setInterval = details.setTimeout = details.XMLHttpRequest = 
  details.getComputedStyle = details.trigger = details.dispatchEvent = 
  details.removeEventListener = details.addEventListener = function(){};

  // frames
  details.frames = [details];

  // document
  details.document = details;
  details.document.domain = details.location.href;

  return details;
};

/**
 * Executes a script in a browser like env and returns
 * the result
 *
 * @param {String} contents The script content
 * @returns {Object} The evaluated script.
 */

exports.execute = function execute (contents) {
  var env = exports.env() 
    , script = vm.createScript(contents);

  // run the script with `browser like` globals
  script.runInNewContext(env);

  return env;
};
