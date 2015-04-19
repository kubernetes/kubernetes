var fs      = require('fs');
var path    = require('path');
var express = require('express');
var tinylr  = require('../..');
var debug   = require('debug')('tinylr:server');

process.env.DEBUG = process.env.DEBUG || 'tinylr*';

var app = module.exports = express();

function logger(fmt) {
  fmt = fmt || '%s - %s';

  return function logger(req, res, next) {
    debug(fmt, req.method, req.url);
    next();
  }
}

function throttle(delay, fn) {
  var now = Date.now();

  return function() {
    var from = Date.now();
    var interval = from - now;
    if (interval < delay) return;
    now = from;
    fn.apply(this, arguments);
  };
}

var watch = (function watch(em) {
  em = em || new (require('events').EventEmitter)();

  em.on('rename', function(file) {
    tinylr.changed(file);
  });

  fs.watch(path.join(__dirname, 'styles/site.css'), throttle(200, function(ev, filename) {
    em.emit(ev, filename);
  }));

  return watch;
})();

app
  .use(logger())
  .use('/', express.static(path.join(__dirname)))
  .use(tinylr.middleware({ app: app }));
