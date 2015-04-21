/**=========================================================
 * Module: List Events
 * Visualizer list events
 =========================================================*/

app.controller('ListEventsCtrl', [
  '$scope',
  '$routeParams',
  'k8sApi',
  '$location',
  '$filter',
  function($scope, $routeParams, k8sApi, $location, $filter) {
    'use strict';
    $scope.getData = getData;
    $scope.loading = true;
    $scope.k8sApi = k8sApi;
    $scope.pods = null;
    $scope.groupedPods = null;
    $scope.serverView = false;

    $scope.headers = [
      {name: 'Time', field: 'time'},
      {name: 'From', field: 'from'},
      {name: 'Sub Object Path', field: 'subobject'},
      {name: 'Reason', field: 'reason'},
      {name: 'Message', field: 'message'}
    ];

    $scope.custom = {
      time: '',
      from: 'grey',
      subobject: 'grey',
      reason: 'grey',
      message: 'grey'
    };
    $scope.sortable = ['time', 'from', 'subobject'];
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
      k8sApi.getEvents().success(function(data) {
        $scope.loading = false;

        var _fixComma = function(str) {
          if (str.substring(0, 1) == ',') {
            return str.substring(1);
          } else {
            return str;
          }
        };

        data.items.forEach(function(event) {

          $scope.content.push({
            time: $filter('date')(event.timestamp, 'medium'),
            from: event.source,
            subobject: event.involvedObject.fieldPath,
            reason: event.reason,
            message: event.message
          });

        });

      }).error($scope.handleError);
    }

    getData($routeParams.serviceId);

  }
]);
