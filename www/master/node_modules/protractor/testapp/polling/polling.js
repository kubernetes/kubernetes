function PollingCtrl($scope, $timeout) {
  $scope.count = 0;

  $scope.startPolling = function() {
    function poll() {
      $timeout(function() {
        $scope.count++;
        poll();
      }, 1000);
    };

    poll();
  };
}
PollingCtrl.$inject = ['$scope', '$timeout'];
