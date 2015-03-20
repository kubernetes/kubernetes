'use strict';

angular.module('mean.theme')
	.controller('ThemeController', ['$scope', 'Global',
	  function($scope, Global) {
// Original scaffolded code.
      $scope.global = Global;
      $scope.package = {
        name: 'theme'
      };
    }
  ]);
