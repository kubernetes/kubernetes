/*
 * Test simple cases
 *
 *     angular.module('myMod', []).controller( ... );
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

  it('should annotate controllers', function () {
    var annotated = annotate(function () {
      angular.module('myMod', []).
        controller('MyCtrl', function ($scope) {
          $scope.foo = 'bar';
        });
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      angular.module('myMod', []).controller('MyCtrl', [
        '$scope',
        function ($scope) {
          $scope.foo = 'bar';
        }
      ]);
    }));
  });


  it('should annotate directives', function () {
    var annotated = annotate(function () {
      angular.module('myMod', []).
        directive('myDirective', function ($rootScope) {
          return {
            restrict: 'E',
            template: 'sup'
          };
        });
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      angular.module('myMod', []).directive('myDirective', [
        '$rootScope',
        function ($rootScope) {
          return {
            restrict: 'E',
            template: 'sup'
          };
        }
      ]);
    }));
  });


  it('should annotate filters', function () {
    var annotated = annotate(function () {
      angular.module('myMod', []).
        filter('myFilter', function (dep) {});
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      angular.module('myMod', []).filter('myFilter', [
        'dep',
        function (dep) {
        }
      ]);
    }));
  });


  it('should annotate services', function () {
    var annotated = annotate(function () {
      angular.module('myMod', []).
        service('myService', function (dep) {});
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      angular.module('myMod', []).service('myService', [
        'dep',
        function (dep) {
        }
      ]);
    }));
  });


  it('should annotate factories', function () {
    var annotated = annotate(function () {
      angular.module('myMod', []).
        controller('factory', function (dep) {});
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      angular.module('myMod', []).controller('factory', [
        'dep',
        function (dep) {
        }
      ]);
    }));
  });


  it('should annotate decorators', function () {
    var annotated = annotate(function () {
      angular.module('myMod', []).
        decorator('myService', function (dep) {});
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      angular.module('myMod', []).decorator('myService', [
        'dep',
        function (dep) {
        }
      ]);
    }));
  });


  it('should annotate config', function () {
    var annotated = annotate(function () {
      angular.module('myMod', []).
        config(function (dep) {});
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      angular.module('myMod', []).config([
        'dep',
        function (dep) {
        }
      ]);
    }));
  });


  it('should annotate run', function () {
    var annotated = annotate(function () {
      angular.module('myMod', []).
        run(function (dep) {});
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      angular.module('myMod', []).run([
        'dep',
        function (dep) {
        }
      ]);
    }));
  });


  it('should annotate providers defined by functions', function () {
    var annotated = annotate(function () {
      angular.module('myMod', []).
        provider('myService', function (dep) {
          this.$get = function(otherDep) {};
        });
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      angular.module('myMod', []).provider('myService', [
        'dep',
        function (dep) {
          this.$get = ['otherDep', function(otherDep) {}];
        }
      ]);
    }));
  });


  it('should annotate providers defined by objects', function () {
    var annotated = annotate(function () {
      angular.module('myMod', []).
        provider('myService', {
          $get: function(otherDep) {}
        })
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      angular.module('myMod', []).
        provider('myService', {
          $get: ['otherDep', function(otherDep) {}]
        });
    }));
  });


  it('should annotate declarations on modules being referenced', function () {
    var annotated = annotate(function () {
      angular.module('myMod', []);
      angular.module('myMod').
        provider('myService', function (dep) {});
    });

    annotated.should.equal(stringifyFunctionBody(function () {
      angular.module('myMod', []);
      angular.module('myMod').provider('myService', [
        'dep',
        function (dep) {
        }
      ]);
    }));
  });


  it('should not annotate declarations with no dependencies', function () {
    var fn = function () {
      angular.module('myMod', []).
        provider('myService', function () {});
    };
    var annotated = annotate(fn);

    annotated.should.equal(stringifyFunctionBody(fn));
  });


  it('should not annotate constants', function () {
    var fn = function () {
      angular.module('myMod', []).constant('fortyTwo', 42);
    };

    var annotated = annotate(fn);
    annotated.should.equal(stringifyFunctionBody(fn));
  });


  it('should not annotate values', function () {
    var fn = function () {
      angular.module('myMod', []).value('fortyTwo', 42);
    };

    var annotated = annotate(fn);
    annotated.should.equal(stringifyFunctionBody(fn));
  });

});
