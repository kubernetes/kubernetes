function AsyncCtrl($scope, $http, $timeout, $location) {
  $scope.slowHttpStatus = 'not started';
  $scope.slowFunctionStatus = 'not started';
  $scope.slowTimeoutStatus = 'not started';
  $scope.slowAngularTimeoutStatus = 'not started';
  $scope.slowAngularTimeoutPromiseStatus = 'not started';
  $scope.slowHttpPromiseStatus = 'not started';
  $scope.routingChangeStatus = 'not started';
  $scope.templateUrl = '/fastTemplateUrl';

  $scope.slowHttp = function() {
    $scope.slowHttpStatus = 'pending...';
    $http({method: 'GET', url: '/slowcall'}).success(function() {
      $scope.slowHttpStatus = 'done';
    });
  };

  $scope.slowFunction = function() {
    $scope.slowFunctionStatus = 'pending...';
    for (var i = 0, t = 0; i < 500000000; ++i) {
      t++;
    }
    $scope.slowFunctionStatus = 'done';
  };

  $scope.slowTimeout = function() {
    $scope.slowTimeoutStatus = 'pending...';
    window.setTimeout(function() {
      $scope.$apply(function() {
        $scope.slowTimeoutStatus = 'done';
      });
    }, 5000);
  };

  $scope.slowAngularTimeout = function() {
    $scope.slowAngularTimeoutStatus = 'pending...';
    $timeout(function() {
      $scope.slowAngularTimeoutStatus = 'done';
    }, 5000);
  };

  $scope.slowAngularTimeoutPromise = function() {
    $scope.slowAngularTimeoutPromiseStatus = 'pending...';
    $timeout(function() {
      // intentionally empty
    }, 5000).then(function() {
      $scope.slowAngularTimeoutPromiseStatus = 'done';
    });
  };

  $scope.slowHttpPromise = function() {
    $scope.slowHttpPromiseStatus = 'pending...';
    $http({method: 'GET', url: '/slowcall'}).success(function() {
      // intentionally empty
    }).then(function() {
      $scope.slowHttpPromiseStatus = 'done';
    });
  };

  $scope.routingChange = function() {
    $scope.routingChangeStatus = 'pending...';
    $location.url('/slowloader');
  };

  $scope.changeTemplateUrl = function() {
    $scope.templateUrl = '/slowTemplateUrl';
  };
}

AsyncCtrl.$inject = ['$scope', '$http', '$timeout', '$location'];
