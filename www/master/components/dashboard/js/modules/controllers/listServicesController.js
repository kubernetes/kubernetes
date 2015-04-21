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
  function($scope, $interval, $routeParams, k8sApi, $rootScope) {
    'use strict';
    $scope.doTheBack = function() { window.history.back(); };

    $scope.headers = [
      {name: 'Name', field: 'name'},
      {name: 'Labels', field: 'labels'},
      {name: 'Selector', field: 'selector'},
      {name: 'IP', field: 'ip'},
      {name: 'Port', field: 'port'}
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

    $scope.content = [];

    $rootScope.doTheBack = $scope.doTheBack;

    $scope.handleError = function(data, status, headers, config) {
      console.log("Error (" + status + "): " + data);
      $scope_.loading = false;
    };

    $scope.getData = function(dataId) {
      $scope.loading = true;
      k8sApi.getServices(dataId).success(angular.bind(this, function(data) {
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

            var _name = '', _uses = '', _component = '', _provider = '';

            if (service.labels !== null && typeof service.labels === 'object') {
              Object.keys(service.labels)
                  .forEach(function(key) {
                    if (key == 'name') {
                      _name += ',' + service.labels[key];
                    }
                    if (key == 'component') {
                      _component += ',' + service.labels[key];
                    }
                    if (key == 'provider') {
                      _provider += ',' + service.labels[key];
                    }
                  });
            }

            var _selectors = '';

            if (service.selector !== null && typeof service.selector === 'object') {
              Object.keys(service.selector)
                  .forEach(function(key) {
                    if (key == 'name') {
                      _selectors += ',' + service.selector[key];
                    }
                  });
            }

            $scope.content.push({
              name: service.id,
              ip: service.portalIP,
              port: service.port,
              selector: addLabel(_fixComma(_selectors), 'name='),
              labels: addLabel(_fixComma(_name), 'name=') + ' ' + addLabel(_fixComma(_component), 'component=') + ' ' +
                          addLabel(_fixComma(_provider), 'provider=')
            });
          });
        }
      })).error($scope.handleError);
    };

    $scope.getData($routeParams.serviceId);
  }
]);
