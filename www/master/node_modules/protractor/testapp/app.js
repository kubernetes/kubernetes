'use strict';


// Declare app level module which depends on filters, and services
angular.module('myApp', ['ngAnimate', 'ngRoute', 'myApp.appVersion']).
  config(['$routeProvider', function($routeProvider) {
    $routeProvider.when('/repeater', {templateUrl: 'repeater/repeater.html', controller: RepeaterCtrl});
    $routeProvider.when('/bindings', {templateUrl: 'bindings/bindings.html', controller: BindingsCtrl});
    $routeProvider.when('/form', {templateUrl: 'form/form.html', controller: FormCtrl});
    $routeProvider.when('/async', {templateUrl: 'async/async.html', controller: AsyncCtrl});
    $routeProvider.when('/conflict', {templateUrl: 'conflict/conflict.html', controller: ConflictCtrl});
    $routeProvider.when('/polling', {templateUrl: 'polling/polling.html', controller: PollingCtrl});
    $routeProvider.when('/animation', {templateUrl: 'animation/animation.html', controller: AnimationCtrl});
    $routeProvider.when('/interaction', {templateUrl: 'interaction/interaction.html', controller: InteractionCtrl});
    $routeProvider.when('/shadow', {templateUrl: 'shadow/shadow.html', controller: ShadowCtrl});
    $routeProvider.when('/slowloader', {
      templateUrl: 'polling/polling.html',
      controller: PollingCtrl,
      resolve: {
        slow: function($timeout) {
          return $timeout(function() {}, 5000);
        }
      }
    });
    $routeProvider.otherwise({redirectTo: '/form'});
  }]);
