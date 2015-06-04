/**=========================================================
 * Module: Services
 * Visualizer for services
 =========================================================*/

app.controller('ListServicesCtrl', [
  '$scope',
  '$interval',
  '$routeParams',
  'k8sApi',
  '$rootScope',
  '$location',
  function($scope, $interval, $routeParams, k8sApi, $rootScope, $location) {
    'use strict';
    $scope.doTheBack = function() { window.history.back(); };

    $scope.headers = [
      {name: 'Name', field: 'name'},
      {name: 'Labels', field: 'labels'},
      {name: 'Selector', field: 'selector'},
      {name: 'IP', field: 'ip'},
      {name: 'Ports', field: 'port'}
    ];

    $scope.custom = {
      name: '',
      ip: 'grey',
      selector: 'grey',
      port: 'grey',
      labels: 'grey'
    };
    $scope.sortable = ['name', 'ip', 'port'];
    $scope.count = 10;

    $scope.go = function(data) { $location.path('/dashboard/services/' + data.name); };

    $scope.content = [];

    $rootScope.doTheBack = $scope.doTheBack;

    $scope.handleError = function(data, status, headers, config) {
      console.log("Error (" + status + "): " + data);
      $scope_.loading = false;
    };

    $scope.getData = function() {
      $scope.loading = true;
      k8sApi.getServices().success(angular.bind(this, function(data) {
        $scope.services = data;
        $scope.loading = false;

        var _fixComma = function(str) {
          if (str.substring(0, 1) == ',') {
            return str.substring(1);
          } else {
            return str;
          }
        };

        var addLabel = function(str, label) {
          if (str) {
            str = label + str;
          }
          return str;
        };

        if (data.items.constructor === Array) {
          data.items.forEach(function(service) {

            var _labels = '';

            if (service.metadata.labels) {
              _labels = _.map(service.metadata.labels, function(v, k) { return k + '=' + v }).join(', ');
            }

            var _selectors = '';

            if (service.spec.selector) {
              _selectors = _.map(service.spec.selector, function(v, k) { return k + '=' + v }).join(', ');
            }

            var _ports = '';

            if (service.spec.ports) {
              _ports = _.map(service.spec.ports, function(p) {
                var n = '';
                if(p.name)
                  n = p.name + ': ';
                 n = n + p.port;
                return n;
               }).join(', ');
            }

            $scope.content.push({
              name: service.metadata.name,
              ip: service.spec.portalIP,
              port: _ports,
              selector: _selectors,
              labels: _labels
            });
          });
        }
      })).error($scope.handleError);
    };

    $scope.getData();
  }
]);
