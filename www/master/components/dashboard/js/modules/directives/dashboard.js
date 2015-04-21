(function() {
  'use strict';

  angular.module('kubernetesApp.components.dashboard')
      .directive(
           'dashboardHeader',
           function() {
             'use strict';
             return {
               restrict: 'A',
               replace: true,
               scope: {user: '='},
               templateUrl: "components/dashboard/pages/header.html",
               controller: [
                 '$scope',
                 '$filter',
                 '$location',
                 '$rootScope',
                 function($scope, $filter, $location, $rootScope) {
                   $scope.$watch('page', function(newValue, oldValue) {
                     if (typeof newValue !== 'undefined') {
                       $location.path(newValue);
                     }
                   });

                   $scope.subpages = [
                     {
                       category: 'dashboard',
                       name: 'Groups',
                       value: '/dashboard/groups/type/selector/',
                       id: 'groupsView'
                     },
                     {category: 'dashboard', name: 'Pods', value: '/dashboard/pods', id: 'podsView'},
                     {category: 'dashboard', name: 'Minions', value: '/dashboard/minions', id: 'minionsView'},
                     {
                       category: 'dashboard',
                       name: 'Replication Controllers',
                       value: '/dashboard/replicationcontrollers',
                       id: 'rcView'
                     },
                     {category: 'dashboard', name: 'Services', value: '/dashboard/services', id: 'servicesView'},
                     {category: 'dashboard', name: 'Events', value: '/dashboard/events', id: 'eventsView'},
                   ];
                 }
               ]
             };
           })
      .directive('dashboardFooter',
                 function() {
                   'use strict';
                   return {
                     restrict: 'A',
                     replace: true,
                     templateUrl: "components/dashboard/pages/footer.html",
                     controller: ['$scope', '$filter', function($scope, $filter) {}]
                   };
                 })
      .directive('mdTable', function() {
        'use strict';
        return {
          restrict: 'E',
          scope: {
            headers: '=',
            content: '=',
            sortable: '=',
            filters: '=',
            customClass: '=customClass',
            thumbs: '=',
            count: '='
          },
          controller: function($scope, $filter, $window, $location) {
            var orderBy = $filter('orderBy');
            $scope.currentPage = 0;
            $scope.nbOfPages = function() { return Math.ceil($scope.content.length / $scope.count); };
            $scope.handleSort = function(field) {
              if ($scope.sortable.indexOf(field) > -1) {
                return true;
              } else {
                return false;
              }
            };
            $scope.go = function(d) {
              if (d.pod) {
                $location.path('/dashboard/pods/' + d.pod);
              } else if (d.name) {
                $location.path('/dashboard/services/' + d.name);
              }
            };
            $scope.order = function(predicate, reverse) {
              $scope.content = orderBy($scope.content, predicate, reverse);
              $scope.predicate = predicate;
            };
            $scope.order($scope.sortable[0], false);
            $scope.getNumber = function(num) { return new Array(num); };
            $scope.goToPage = function(page) { $scope.currentPage = page; };
          },
          templateUrl: 'views/partials/md-table.tmpl.html'
        };
      });

}());
