/**=========================================================
 * Module: Group
 * Visualizer for groups
 =========================================================*/

app.controller('GroupCtrl', [
  '$scope',
  '$route',
  '$interval',
  '$routeParams',
  'k8sApi',
  '$rootScope',
  '$location',
  'lodash',
  function($scope, $route, $interval, $routeParams, k8sApi, $rootScope, $location, _) {
    'use strict';
    $scope.doTheBack = function() { window.history.back(); };

    $scope.capitalize = function(s) { return _.capitalize(s); };

    $rootScope.doTheBack = $scope.doTheBack;

    $scope.resetGroupLayout = function(group) { delete group.settings; };

    $scope.handlePath = function(path) {
      var parts = path.split("/");
      // split leaves an empty string at the beginning.
      parts = parts.slice(1);

      if (parts.length === 0) {
        return;
      }
      this.handleGroups(parts.slice(1));
    };

    $scope.getState = function(obj) { return Object.keys(obj)[0]; };

    $scope.clearSelector = function(grouping) { $location.path("/dashboard/groups/" + grouping + "/selector/"); };

    $scope.changeGroupBy = function() {
      var grouping = $scope.selectedGroupBy;

      var s = _.clone($location.search());
      if ($scope.routeParams.grouping != grouping)
        $location.path("/dashboard/groups/" + grouping + "/selector/").search(s);
    };

    $scope.createBarrier = function(count, callback) {
      var barrier = count;
      var barrierFunction = angular.bind(this, function(data) {
        // JavaScript is single threaded so this is safe.
        barrier--;
        if (barrier === 0) {
          if (callback) {
            callback();
          }
        }
      });
      return barrierFunction;
    };

    $scope.handleGroups = function(parts, selector) {
      $scope.groupBy = parts;
      $scope.loading = true;
      $scope.selector = selector;
      var args = [];
      var type = "";
      if (selector && selector.length > 0) {
        $scope.selectorPieces = selector.split(",");
        var labels = [];
        var fields = [];
        for (var i = 0; i < $scope.selectorPieces.length; i++) {
          var piece = $scope.selectorPieces[i];
          if (piece[0] == '$') {
            fields.push(piece.slice(2));
          } else {
            if (piece.indexOf("type=") === 0) {
              var labelParts = piece.split("=");
              if (labelParts.length > 1) {
                type = labelParts[1];
              }
            } else {
              labels.push(piece);
            }
          }
        }
        if (labels.length > 0) {
          args.push("labels=" + encodeURI(labels.join(",")));
        }
        if (fields.length > 0) {
          args.push("fields=" + encodeURI(fields.join(",")));
        }
      }
      var query = "?" + args.join("&");
      var list = [];
      var count = type.length > 0 ? 1 : 3;
      var barrier = $scope.createBarrier(count, function() {
        $scope.groups = $scope.groupData(list, 0);
        $scope.loading = false;
        $scope.groupByOptions = buildGroupByOptions();
        $scope.selectedGroupBy = $routeParams.grouping;
      });

      if (type === "" || type == "pod") {
        k8sApi.getPods(query).success(function(data) {
          $scope.addLabel("type", "pod", data.items);
          for (var i = 0; data.items && i < data.items.length; ++i) {
            data.items[i].labels.host = data.items[i].currentState.host;
            list.push(data.items[i]);
          }
          barrier();
        }).error($scope.handleError);
      }
      if (type === "" || type == "service") {
        k8sApi.getServices(query).success(function(data) {
          $scope.addLabel("type", "service", data.items);
          for (var i = 0; data.items && i < data.items.length; ++i) {
            list.push(data.items[i]);
          }
          barrier();
        }).error($scope.handleError);
      }
      if (type === "" || type == "replicationController") {
        k8sApi.getReplicationControllers(query).success(angular.bind(this, function(data) {
          $scope.addLabel("type", "replicationController", data.items);
          for (var i = 0; data.items && i < data.items.length; ++i) {
            list.push(data.items[i]);
          }
          barrier();
        })).error($scope.handleError);
      }
    };

    $scope.addLabel = function(key, value, items) {
      if (!items) {
        return;
      }
      for (var i = 0; i < items.length; i++) {
        if (!items[i].labels) {
          items[i].labels = [];
        }
        items[i].labels[key] = value;
      }
    };

    $scope.groupData = function(items, index) {
      var result = {
        "items": {},
        "kind": "grouping"
      };
      for (var i = 0; i < items.length; i++) {
        key = items[i].labels[$scope.groupBy[index]];
        if (!key) {
          key = "";
        }
        var list = result.items[key];
        if (!list) {
          list = [];
          result.items[key] = list;
        }
        list.push(items[i]);
      }

      if (index + 1 < $scope.groupBy.length) {
        for (var key in result.items) {
          result.items[key] = $scope.groupData(result.items[key], index + 1);
        }
      }
      return result;
    };
    $scope.getGroupColor = function(type) {
      if (type === 'pod') {
        return '#6193F0';
      } else if (type === 'replicationController') {
        return '#E008FE';
      } else if (type === 'service') {
        return '#7C43FF';
      }
    };

    var groups = $routeParams.grouping;
    if (!groups) {
      groups = '';
    }

    $scope.routeParams = $routeParams;
    $scope.route = $route;

    $scope.handleGroups(groups.split('/'), $routeParams.selector);

    $scope.handleError = function(data, status, headers, config) {
      console.log("Error (" + status + "): " + data);
      $scope_.loading = false;
    };

    function getDefaultGroupByOptions() { return [{name: 'Type', value: 'type'}, {name: 'Name', value: 'name'}]; }

    function buildGroupByOptions() {
      var g = $scope.groups;
      var options = getDefaultGroupByOptions();
      var newOptions = _.map(g.items, function(vals) { return _.map(vals, function(v) { return _.keys(v.labels); }); });
      newOptions =
          _.reject(_.uniq(_.flattenDeep(newOptions)), function(o) { return o == 'name' || o == 'type' || o == ""; });
      newOptions = _.map(newOptions, function(o) {
        return {
          name: o,
          value: o
        };
      });

      options = options.concat(newOptions);
      return options;
    }

    $scope.changeFilterBy = function(selector) {
      var grouping = $scope.selectedGroupBy;

      var s = _.clone($location.search());
      if ($scope.routeParams.selector != selector)
        $location.path("/dashboard/groups/" + $scope.routeParams.grouping + "/selector/" + selector).search(s);
    };
  }
]);
