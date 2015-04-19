var port = process.env.LR_PORT || process.env.PORT || 35729;

var fs      = require('fs');
var path    = require('path');
var express = require('express');
var tinylr  = require('../..');
var body    = require('body-parser');
var debug   = require('debug')('tinylr:server');

process.env.DEBUG = process.env.DEBUG || 'tinylr*';

var app = require('./app');

app.listen(port, function(err) {
  if (err) throw err;
  debug('listening on %d', port);
});
