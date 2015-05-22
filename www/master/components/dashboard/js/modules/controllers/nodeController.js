/**=========================================================
 * Module: Nodes
 * Visualizer for nodes
 =========================================================*/

app.controller('NodeCtrl', [
  '$scope',
  '$interval',
  '$routeParams',
  'k8sApi',
  '$rootScope',
  function($scope, $interval, $routeParams, k8sApi, $rootScope) {
    'use strict';
    $scope.doTheBack = function() { window.history.back(); };

    $rootScope.doTheBack = $scope.doTheBack;

    $scope.handleError = function(data, status, headers, config) {
      console.log("Error (" + status + "): " + data);
      $scope_.loading = false;
    };

    $scope.handleNode = function(nodeId) {
      $scope.loading = true;
      k8sApi.getNodes(nodeId).success(angular.bind(this, function(data) {
        $scope.node = data;
        $scope.loading = false;
      })).error($scope.handleError);
    };

    $scope.handleNode($routeParams.nodeId);
  }
]);
