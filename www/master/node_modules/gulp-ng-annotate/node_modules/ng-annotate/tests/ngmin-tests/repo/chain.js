/*
 * Test chained declarations
 *     angular.module('myMod', []).
 *       controller( ... ).
 *       controller( ... );
 */


var assert = require('should');

// so we don't have to put the stuff we're testing into a string
var stringifyFunctionBody = require('./util').stringifyFunctionBody;
var annotate = function (arg) {
  return require('../main').annotate(
    stringifyFunctionBody(arg));
};


describe('annotate', function () {

  it('should annotate chained declarations', function () {
    var annotated = annotate(function () {
      angular.module('myMod', []).
        service('myService', function (dep) {}).
        service('MyCtrl', function ($scope) {});
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      angular.module('myMod', []).
        service('myService', ['dep', function (dep) {}]).
        service('MyCtrl', ['$scope', function ($scope) {}]);
    }));
  });

  it('should annotate multiple chained declarations', function () {
    var annotated = annotate(function () {
      angular.module('myMod', []).
        service('myService', function (dep) {}).
        service('myService2', function (dep) {}).
        service('myService3', function (dep) {}).
        service('MyCtrl', function ($scope) {});
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      angular.module('myMod', []).
        service('myService', ['dep', function (dep) {}]).
        service('myService2', ['dep', function (dep) {}]).
        service('myService3', ['dep', function (dep) {}]).
        service('MyCtrl', ['$scope', function ($scope) {}]);
    }));
  });

  it('should annotate multiple chained declarations on constants', function() {
    var annotated = annotate(function () {
      angular.module('myMod', []).
        constant('myConstant', 'someConstant').
        constant('otherConstant', 'otherConstant').
        service('myService1', function (dep) {}).
        service('MyCtrl', function ($scope) {});
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      angular.module('myMod', []).
        constant('myConstant', 'someConstant').
        constant('otherConstant', 'otherConstant').
        service('myService1', ['dep', function (dep) {}]).
        service('MyCtrl', ['$scope', function ($scope) {}]);
    }));
  });

  it('should annotate multiple chained declarations on values', function() {
    var annotated = annotate(function () {
      angular.module('myMod', []).
        value('myConstant', 'someConstant').
        value('otherConstant', 'otherConstant').
        service('myService1', function (dep) {}).
        service('MyCtrl', function ($scope) {});
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      angular.module('myMod', []).
        value('myConstant', 'someConstant').
        value('otherConstant', 'otherConstant').
        service('myService1', ['dep', function (dep) {}]).
        service('MyCtrl', ['$scope', function ($scope) {}]);
    }));
  });

  it('should annotate multiple chained declarations on constants and value regardless of order', function() {
    var annotated = annotate(function () {
      angular.module('myMod', []).
        value('myConstant', 'someConstant').
        service('myService1', function (dep) {}).
        constant('otherConstant', 'otherConstant').
        service('MyCtrl', function ($scope) {});
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      angular.module('myMod', []).
        value('myConstant', 'someConstant').
        service('myService1', ['dep', function (dep) {}]).
        constant('otherConstant', 'otherConstant').
        service('MyCtrl', ['$scope', function ($scope) {}]);
    }));
  });

  it('should annotate refs that have been chained', function () {
    var annotated = annotate(function () {
      var mod =  angular.module('chain', []);
      mod.factory('a', function ($scope){}).
        factory('b', function ($scope){});
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      var mod =  angular.module('chain', []);
      mod.factory('a', ['$scope', function($scope){}]).
        factory('b', ['$scope', function($scope){}]);
    }));
  });

  it('should annotate refs to chains', function () {
    var annotated = annotate(function () {
      var mod =  angular.module('chain', []).
        factory('a', function ($scope){});
      mod.factory('b', function ($scope){});
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      var mod =  angular.module('chain', []).
        factory('a', ['$scope', function($scope){}]);
      mod.factory('b', ['$scope', function($scope){}]);
    }));
  });

});
