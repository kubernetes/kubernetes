/*!
 * Angular Material Design
 * https://github.com/angular/material
 * @license MIT
 * v0.8.1
 */
goog.provide('ng.material.components.swipe');

(function() {
'use strict';


/**
 * @ngdoc module
 * @name material.components.swipe
 * @description Swipe module!
 */
/**
 * @ngdoc directive
 * @module material.components.swipe
 * @name mdSwipeLeft
 *
 * @restrict A
 *
 * @description
 * The md-swipe-left directives allows you to specify custom behavior when an element is swiped
 * left.
 *
 * @usage
 * <hljs lang="html">
 * <div md-swipe-left="onSwipeLeft()">Swipe me left!</div>
 * </hljs>
 */

/**
 * @ngdoc directive
 * @module material.components.swipe
 * @name mdSwipeRight
 *
 * @restrict A
 *
 * @description
 * The md-swipe-right directives allows you to specify custom behavior when an element is swiped
 * right.
 *
 * @usage
 * <hljs lang="html">
 * <div md-swipe-right="onSwipeRight()">Swipe me right!</div>
 * </hljs>
 */

var module = angular.module('material.components.swipe',[]);

['SwipeLeft', 'SwipeRight'].forEach(function(name) {
  var directiveName = 'md' + name;
  var eventName = '$md.' + name.toLowerCase();

  module.directive(directiveName, /*@ngInject*/ ["$parse", function($parse) {
    return {
      restrict: 'A',
      link: postLink
    };

    function postLink(scope, element, attr) {
      var fn = $parse(attr[directiveName]);

      element.on(eventName, function(ev) {
        scope.$apply(function() {
          fn(scope, {
            $event: ev
          });
        });
      });

    }
  }]);
});

})();
