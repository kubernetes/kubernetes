/**=========================================================
 * Module: Header
 * Visualizer for clusters
 =========================================================*/

angular.module('kubernetesApp.components.dashboard', [])
    .controller('HeaderCtrl', [
      '$scope',
      '$location',
      function($scope, $location) {
        'use strict';
        $scope.$watch('Pages', function(newValue, oldValue) {
          if (typeof newValue !== 'undefined') {
            $location.path(newValue);
          }
        });

        $scope.subPages = [
          {category: 'dashboard', name: 'Explore', value: '/dashboard/groups/type/selector/'},
          {category: 'dashboard', name: 'Pods', value: '/dashboard/pods'},
          {category: 'dashboard', name: 'Minions', value: '/dashboard/minions'},
          {category: 'dashboard', name: 'Replication Controllers', value: '/dashboard/replicationcontrollers'},
          {category: 'dashboard', name: 'Services', value: '/dashboard/services'},
          {category: 'dashboard', name: 'Events', value: '/dashboard/events'}
        ];
      }
    ]);
