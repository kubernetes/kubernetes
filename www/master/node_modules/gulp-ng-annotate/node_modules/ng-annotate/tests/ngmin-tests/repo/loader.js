/*
 * Test for angular modules that are wrapped by goofy
 * 3rd party loaders like Require.js
 */

var assert = require('should');

// so we don't have to put the stuff we're testing into a string
var stringifyFunctionBody = require('./util').stringifyFunctionBody;
var annotate = function (arg) {
  return require('../main').annotate(
    stringifyFunctionBody(arg));
};


describe('annotate', function () {

  it('should annotate modules inside of loaders', function () {
    var annotated = annotate(function () {
      define(["./thing"], function(thing) {
        angular.module('myMod', []).
          controller('MyCtrl', function ($scope) {});
      });
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      define(["./thing"], function(thing) {
        angular.module('myMod', []).
          controller('MyCtrl', ['$scope', function ($scope) {}]);
      });
    }));
  });

  it('should annotate module refs inside of loaders', function () {
    var annotated = annotate(function () {


      define(["./thing"], function(thing) {
        var myMod = angular.module('myMod', []);
        myMod.controller('MyCtrl', function ($scope) {});
        return myMod;
      });

    });

    annotated.should.equal(stringifyFunctionBody(function () {
      define(["./thing"], function(thing) {
        var myMod = angular.module('myMod', []);
        myMod.controller('MyCtrl', ['$scope', function ($scope) {}]);
        return myMod;
      });
    }));
  });


});
