function RepeaterCtrl($scope) {
  $scope.days = [
    {initial: 'M', name: 'Monday'},
    {initial: 'T', name: 'Tuesday'},
    {initial: 'W', name: 'Wednesday'},
    {initial: 'Th', name: 'Thursday'},
    {initial: 'F', name: 'Friday'}
  ];
}

RepeaterCtrl.$inject = ['$scope'];
