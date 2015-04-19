#!/usr/bin/env node

var express = require('express');
var optimist = require('optimist');
var util = require('util');
var path = require('path');
var env = require('../../spec/environment.js');

var testApp = express();
var DEFAULT_PORT = process.env.HTTP_PORT || env.webServerDefaultPort;
var testAppDir = path.resolve(__dirname, '..');

var argv = optimist.describe('port', 'port').
    default('port', DEFAULT_PORT).
    describe('ngversion', 'version of AngularJS to use').
    default('ngversion', '1.3.13').
    argv;

var angularDir = path.join(testAppDir, 'lib/angular_v' + argv.ngversion);

var main = function() {
  var port = argv.port;
  testApp.listen(port);
  util.puts(["Starting express web server in", testAppDir ,"on port", port].
      join(" "));
};

var storage = {};
var testMiddleware = function(req, res, next) {
  if (req.path == '/fastcall') {
    res.send(200, 'done');
  } else if (req.path == '/slowcall') {
    setTimeout(function() {
      res.send(200, 'finally done');
    }, 5000);
  } else if (req.path == '/fastTemplateUrl') {
    res.send(200, 'fast template contents');
  } else if (req.path == '/slowTemplateUrl') {
    setTimeout(function() {
      res.send(200, 'slow template contents');
    }, 5000);
  } else if (req.path == '/storage') {
    if (req.method === 'GET') {
      var value;
      if (req.query.q) {
        value = storage[req.query.q];
        res.send(200, value);
      } else {
        res.send(400, 'must specify query');
      }
    } else if (req.method === 'POST') {
      if (req.body.key && req.body.value) {
        storage[req.body.key] = req.body.value;
        res.send(200);
      } else {
        res.send(400, 'must specify key/value pair');
      }
    } else {
      res.send(400, 'only accepts GET/POST');
    }
  } else {
    return next();
  }
};

testApp.configure(function() {
  testApp.use('/lib/angular', express.static(angularDir));
  testApp.use(express.static(testAppDir));
  testApp.use(express.json());
  testApp.use(testMiddleware);
});

main();
