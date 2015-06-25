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
      {name: 'First Seen', field: 'firstSeen'},
      {name: 'Last Seen', field: 'lastSeen'},
      {name: 'Count', field: 'count'},
      {name: 'Name', field: 'name'},
      {name: 'Kind', field: 'kind'},
      {name: 'SubObject', field: 'subObject'},
      {name: 'Reason', field: 'reason'},
      {name: 'Source', field: 'source'},
      {name: 'Message', field: 'message'}
    ];


    $scope.sortable = ['firstSeen', 'lastSeen', 'count', 'name', 'kind', 'subObject', 'reason', 'source'];
    $scope.count = 10;
    function handleError(data, status, headers, config) {
      console.log("Error (" + status + "): " + data);
      $scope.loading = false;
    }

    $scope.content = [];

    function getData() {
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
          var _sources = '';
          if (event.source) {
            _sources = event.source.component + ' ' + event.source.host;
          }


          $scope.content.push({
            firstSeen: $filter('date')(event.firstTimestamp, 'medium'),
            lastSeen: $filter('date')(event.lastTimestamp, 'medium'),
            count: event.count,
            name: event.involvedObject.name,
            kind: event.involvedObject.kind,
            subObject: event.involvedObject.fieldPath,
            reason: event.reason,
            source: _sources,
            message: event.message
          });



        });

      }).error($scope.handleError);
    }

    getData();

  }
]);
