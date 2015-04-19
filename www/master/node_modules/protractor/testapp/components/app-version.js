'use strict';

angular.module('myApp.appVersion', []).
  value('version', '0.1').
  directive('appVersion', ['version', function(version) {
    return function(scope, elm, attrs) {
      elm.text(version);
    };
  }]);
