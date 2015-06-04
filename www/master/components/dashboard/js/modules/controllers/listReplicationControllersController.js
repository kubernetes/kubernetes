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

    $scope.go = function(data) { $location.path('/dashboard/replicationcontrollers/' + data.controller); };

    function handleError(data, status, headers, config) {
      console.log("Error (" + status + "): " + data);
      $scope.loading = false;
    }

    $scope.content = [];

    function getData() {
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

          if (replicationController.spec.template.spec.containers) {
            Object.keys(replicationController.spec.template.spec.containers)
                .forEach(function(key) {
                  _name += replicationController.spec.template.spec.containers[key].name;
                  _image += replicationController.spec.template.spec.containers[key].image;
                });
          }

          var _selectors = '';

          if (replicationController.spec.selector) {
            _selectors = _.map(replicationController.spec.selector, function(v, k) { return k + '=' + v }).join(', ');
          }

          $scope.content.push({
            controller: replicationController.metadata.name,
            containers: _name,
            images: _image,
            selector: _selectors,
            replicas: replicationController.status.replicas
          });

        });

      }).error($scope.handleError);
    }

    getData();

  }
]);
