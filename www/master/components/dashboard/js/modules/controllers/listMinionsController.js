/**=========================================================
 * Module: Minions
 * Visualizer for minions
 =========================================================*/

app.controller('ListMinionsCtrl', [
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

    $scope.headers = [{name: 'Name', field: 'name'}, {name: 'IP', field: 'ip'}, {name: 'Status', field: 'status'}];

    $scope.custom = {
      name: '',
      status: 'grey',
      ip: 'grey'
    };
    $scope.sortable = ['name', 'status', 'ip'];
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
      k8sApi.getMinions().success(function(data) {
        $scope.loading = false;

        var _fixComma = function(str) {
          if (str.substring(0, 1) == ',') {
            return str.substring(1);
          } else {
            return str;
          }
        };

        data.items.forEach(function(minion) {
          var _kind = '';

          if (minion.status.conditions) {
            Object.keys(minion.status.conditions)
                .forEach(function(key) { _kind += minion.status.conditions[key].kind; });
          }

          $scope.content.push({name: minion.id, ip: minion.hostIP, status: _kind});

        });

      }).error($scope.handleError);
    }

    getData($routeParams.serviceId);

  }
]);
