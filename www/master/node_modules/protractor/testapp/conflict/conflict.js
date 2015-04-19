function ConflictCtrl($scope) {
  $scope.item = {
    reusedBinding: 'outer',
    alsoReused: 'outerbarbaz'
  };

  $scope.wrapper = [{
    reusedBinding: 'inner',
    alsoReused: 'innerbarbaz'
  }];
}
ConflictCtrl.$inject = ['$scope'];
