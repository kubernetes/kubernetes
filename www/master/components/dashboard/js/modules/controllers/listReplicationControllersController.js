/**=========================================================
 * Module: Replication Controllers
 * Visualizer for replication controllers
 =========================================================*/

app.controller('ListReplicationControllersCtrl', [
  '$scope',
  '$routeParams',
  'k8sApi',
  '$location',
  function($scope, $routeParams, k8sApi, $location) {
    'use strict';
    $scope.getData = getData;
    $scope.loading = true;
    $scope.k8sApi = k8sApi;
    $scope.pods = null;
    $scope.groupedPods = null;
    $scope.serverView = false;

    $scope.headers = [
      {name: 'Controller', field: 'controller'},
      {name: 'Containers', field: 'containers'},
      {name: 'Images', field: 'images'},
      {name: 'Selector', field: 'selector'},
      {name: 'Replicas', field: 'replicas'}
    ];

    $scope.custom = {
      controller: '',
      containers: 'grey',
      images: 'grey',
      selector: 'grey',
      replicas: 'grey'
    };
    $scope.sortable = ['controller', 'containers', 'images'];
    $scope.thumbs = 'thumb';
    $scope.count = 10;

    $scope.go = function(d) { $location.path('/dashboard/pods/' + d.id); };

    $scope.moreClick = function(d, e) {
      $location.path('/dashboard/pods/' + d.id);
      e.stopPropagation();
    };

    function handleError(data, status, headers, config) {
      console.log("Error (" + status + "): " + data);
      $scope.loading = false;
    }

    $scope.content = [];

    function getData(dataId) {
      $scope.loading = true;
      k8sApi.getReplicationControllers().success(function(data) {
        $scope.loading = false;

        var _fixComma = function(str) {
          if (str.substring(0, 1) == ',') {
            return str.substring(1);
          } else {
            return str;
          }
        };

        data.items.forEach(function(replicationController) {

          var _name = '', _image = '';

          if (replicationController.desiredState.podTemplate.desiredState.manifest.containers) {
            Object.keys(replicationController.desiredState.podTemplate.desiredState.manifest.containers)
                .forEach(function(key) {
                  _name += replicationController.desiredState.podTemplate.desiredState.manifest.containers[key].name;
                  _image += replicationController.desiredState.podTemplate.desiredState.manifest.containers[key].image;
                });
          }

          var _name_selector = '';

          if (replicationController.desiredState.replicaSelector) {
            Object.keys(replicationController.desiredState.replicaSelector)
                .forEach(function(key) { _name_selector += replicationController.desiredState.replicaSelector[key]; });
          }

          $scope.content.push({
            controller: replicationController.id,
            containers: _name,
            images: _image,
            selector: _name_selector,
            replicas: replicationController.currentState.replicas
          });

        });

      }).error($scope.handleError);
    }

    getData($routeParams.serviceId);

  }
]);
