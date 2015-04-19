/*
 * Test cases where there's a reference to a module
 *
 *     var myMod = angular.module('myMod', []);
 *     myMod.controller( ... )
 *
 */

var assert = require('should');

// so we don't have to put the stuff we're testing into a string
var stringifyFunctionBody = require('./util').stringifyFunctionBody;
var annotate = function (arg) {
  return require('../main').annotate(
    stringifyFunctionBody(arg));
};


describe('annotate', function () {

  it('should annotate declarations on referenced modules', function () {
    var annotated = annotate(function () {
      var myMod = angular.module('myMod', []);
      myMod.controller('MyCtrl', function ($scope) {});
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      var myMod = angular.module('myMod', []);
      myMod.controller('MyCtrl', [
        '$scope',
        function ($scope) {
        }
      ]);
    }));
  });

  it('should annotate declarations on referenced modules when reference is declared then initialized', function () {
    var annotated = annotate(function () {
      var myMod;
      myMod = angular.module('myMod', []);
      myMod.controller('MyCtrl', function ($scope) {});
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      var myMod;
      myMod = angular.module('myMod', []);
      myMod.controller('MyCtrl', [
        '$scope',
        function ($scope) {
        }
      ]);
    }));
  });

  it('should annotate object-defined providers on referenced modules', function () {
    var annotated = annotate(function () {
      var myMod;
      myMod = angular.module('myMod', []);
      myMod.provider('MyService', { $get: function(service) {} });
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      var myMod;
      myMod = angular.module('myMod', []);
      myMod.provider('MyService', {
        $get: ['service', function(service) {}]
      });
    }));
  });

  // TODO: lol commenting out test cases
  /*
  it('should annotate declarations on referenced modules ad infinitum', function () {
    var annotated = annotate(function () {
      var myMod = angular.module('myMod', []);
      var myMod2 = myMod, myMod3;
      myMod3 = myMod2;
      myMod3.controller('MyCtrl', function ($scope) {});
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      var myMod = angular.module('myMod', []);
      var myMod2 = myMod, myMod3;
      myMod3 = myMod2;
      myMod3.controller('MyCtrl', ['$scope', function ($scope) {}]);
    }));
  });
  */

  // TODO: it should annotate silly assignment chains

  it('should not annotate declarations on non-module objects', function () {
    var fn = function () {
      var myMod, myOtherMod;
      myMod = angular.module('myMod', []);
      myOtherMod.controller('MyCtrl', function ($scope) {});
    };
    var annotated = annotate(fn);
    annotated.should.equal(stringifyFunctionBody(fn));
  });

  it('should keep comments', function() {
    var annotated = annotate(function () {
      var myMod = angular.module('myMod', []);
      /*! license */
      myMod.controller('MyCtrl', function ($scope) {});
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      var myMod = angular.module('myMod', []);
      /*! license */
      myMod.controller('MyCtrl', [
        '$scope',
        function ($scope) {
        }
      ]);
    }));
  });

});
