/**=========================================================
 * Module: Replication
 * Visualizer for replication controllers
 =========================================================*/

function ReplicationController() {
}

ReplicationController.prototype.getData = function(dataId) {
  this.scope.loading = true;
  this.k8sApi.getReplicationControllers(dataId).success(angular.bind(this, function(data) {
    this.scope.replicationController = data;
    this.scope.loading = false;
  })).error(angular.bind(this, this.handleError));
};

ReplicationController.prototype.handleError = function(data, status, headers, config) {
  console.log("Error (" + status + "): " + data);
  this.scope.loading = false;
};

app.controller('ReplicationControllerCtrl', [
  '$scope',
  '$routeParams',
  'k8sApi',
  function($scope, $routeParams, k8sApi) {
    $scope.controller = new ReplicationController();
    $scope.controller.k8sApi = k8sApi;
    $scope.controller.scope = $scope;
    $scope.controller.getData($routeParams.replicationControllerId);
  }
]);
