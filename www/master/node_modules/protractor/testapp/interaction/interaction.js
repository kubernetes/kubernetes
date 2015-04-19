function InteractionCtrl($scope, $interval, $http) {

  $scope.messages = [];
  $scope.message = '';
  $scope.user = '';
  $scope.userInput = '';

  $scope.sendUser = function() {
    $scope.user = $scope.userInput;
  };

  var loadMessages = function() {
    $http.get('/storage?q=chatMessages').
      success(function(data) {
        $scope.messages = data ? data : [];
      }).
      error(function(err) {
        $scope.messages = ['server request failed with: ' + err];
      });
  };
  var saveMessages = function() {
    var data = {
      key: 'chatMessages',
      value: $scope.messages
    };
    $http.post('/storage', data);
  };

  $scope.sendMessage = function() {
    $scope.messages.push($scope.user + ': ' + $scope.message);
    $scope.message = '';
    saveMessages();
  };
  $scope.clearMessages = function() {
    $scope.messages = [];
    saveMessages();
  };

  $interval(function() {
    loadMessages();
  }, 100);
}
InteractionCtrl.$inject = ['$scope', '$interval', '$http'];
