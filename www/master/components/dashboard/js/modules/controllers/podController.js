/**=========================================================
 * Module: Pods
 * Visualizer for pods
 =========================================================*/

app.controller('PodCtrl', [
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

    $scope.handlePod = function(podId) {
      $scope.loading = true;
      k8sApi.getPods(podId).success(angular.bind(this, function(data) {
        $scope.pod = data;
        $scope.loading = false;
      })).error($scope.handleError);
    };

    $scope.handlePod($routeParams.podId);
  }
]);
