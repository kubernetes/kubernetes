function FormCtrl($scope, $window) {
  $scope.greeting = 'Hiya';
  $scope.username = 'Anon';
  $scope.nickname = 'annie';
  $scope.aboutbox = 'This is a text box';
  $scope.color = 'blue';
  $scope.show = true;

  $scope.colors = ['red', 'green', 'blue'];
  $scope.dayColors = [{day: 'Mon', color: 'red'}, {day: 'Tue', color: 'green'}, {day: 'Wed', color: 'blue'}];

  $scope.fruit = '';
  $scope.defaultFruit = 'apple';
  $scope.fruits = ['pear', 'peach', 'banana'];

  $scope.doAlert = function() {
    $window.alert('Hello');
  };
}
FormCtrl.$inject = ['$scope', '$window'];
