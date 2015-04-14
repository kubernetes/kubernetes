/*!
 * Angular Material Design
 * https://github.com/angular/material
 * @license MIT
 * v0.8.1
 */
goog.provide('ng.material.components.bottomSheet');
goog.require('ng.material.components.backdrop');
goog.require('ng.material.core');
(function() {
'use strict';

/**
 * @ngdoc module
 * @name material.components.bottomSheet
 * @description
 * BottomSheet
 */
angular.module('material.components.bottomSheet', [
  'material.core',
  'material.components.backdrop'
])
  .directive('mdBottomSheet', MdBottomSheetDirective)
  .provider('$mdBottomSheet', MdBottomSheetProvider);

function MdBottomSheetDirective() {
  return {
    restrict: 'E'
  };
}

/**
 * @ngdoc service
 * @name $mdBottomSheet
 * @module material.components.bottomSheet
 *
 * @description
 * `$mdBottomSheet` opens a bottom sheet over the app and provides a simple promise API.
 *
 * ## Restrictions
 *
 * - The bottom sheet's template must have an outer `<md-bottom-sheet>` element.
 * - Add the `md-grid` class to the bottom sheet for a grid layout.
 * - Add the `md-list` class to the bottom sheet for a list layout.
 *
 * @usage
 * <hljs lang="html">
 * <div ng-controller="MyController">
 *   <md-button ng-click="openBottomSheet()">
 *     Open a Bottom Sheet!
 *   </md-button>
 * </div>
 * </hljs>
 * <hljs lang="js">
 * var app = angular.module('app', ['ngMaterial']);
 * app.controller('MyController', function($scope, $mdBottomSheet) {
 *   $scope.openBottomSheet = function() {
 *     $mdBottomSheet.show({
 *       template: '<md-bottom-sheet>Hello!</md-bottom-sheet>'
 *     });
 *   };
 * });
 * </hljs>
 */

 /**
 * @ngdoc method
 * @name $mdBottomSheet#show
 *
 * @description
 * Show a bottom sheet with the specified options.
 *
 * @param {object} options An options object, with the following properties:
 *
 *   - `templateUrl` - `{string=}`: The url of an html template file that will
 *   be used as the content of the bottom sheet. Restrictions: the template must
 *   have an outer `md-bottom-sheet` element.
 *   - `template` - `{string=}`: Same as templateUrl, except this is an actual
 *   template string.
 *   - `scope` - `{object=}`: the scope to link the template / controller to. If none is specified, it will create a new child scope.
 *     This scope will be destroyed when the bottom sheet is removed unless `preserveScope` is set to true.
 *   - `preserveScope` - `{boolean=}`: whether to preserve the scope when the element is removed. Default is false
 *   - `controller` - `{string=}`: The controller to associate with this bottom sheet.
 *   - `locals` - `{string=}`: An object containing key/value pairs. The keys will
 *   be used as names of values to inject into the controller. For example,
 *   `locals: {three: 3}` would inject `three` into the controller with the value
 *   of 3.
 *   - `targetEvent` - `{DOMClickEvent=}`: A click's event object. When passed in as an option,
 *   the location of the click will be used as the starting point for the opening animation
 *   of the the dialog.
 *   - `resolve` - `{object=}`: Similar to locals, except it takes promises as values
 *   and the bottom sheet will not open until the promises resolve.
 *   - `controllerAs` - `{string=}`: An alias to assign the controller to on the scope.
 *   - `parent` - `{element=}`: The element to append the bottom sheet to. Defaults to appending
 *     to the root element of the application.
 *   - `disableParentScroll` - `{boolean=}`: Whether to disable scrolling while the bottom sheet is open.
 *     Default true.
 *
 * @returns {promise} A promise that can be resolved with `$mdBottomSheet.hide()` or
 * rejected with `$mdBottomSheet.cancel()`.
 */

/**
 * @ngdoc method
 * @name $mdBottomSheet#hide
 *
 * @description
 * Hide the existing bottom sheet and resolve the promise returned from
 * `$mdBottomSheet.show()`.
 *
 * @param {*=} response An argument for the resolved promise.
 *
 */

/**
 * @ngdoc method
 * @name $mdBottomSheet#cancel
 *
 * @description
 * Hide the existing bottom sheet and reject the promise returned from
 * `$mdBottomSheet.show()`.
 *
 * @param {*=} response An argument for the rejected promise.
 *
 */

function MdBottomSheetProvider($$interimElementProvider) {
  // how fast we need to flick down to close the sheet, pixels/ms
  var CLOSING_VELOCITY = 0.5;
  var PADDING = 80; // same as css

  bottomSheetDefaults.$inject = ["$animate", "$mdConstant", "$timeout", "$$rAF", "$compile", "$mdTheming", "$mdBottomSheet", "$rootElement", "$rootScope", "$mdGesture"];
  return $$interimElementProvider('$mdBottomSheet')
    .setDefaults({
      methods: ['disableParentScroll', 'escapeToClose', 'targetEvent'],
      options: bottomSheetDefaults
    });

  /* @ngInject */
  function bottomSheetDefaults($animate, $mdConstant, $timeout, $$rAF, $compile, $mdTheming, $mdBottomSheet, $rootElement, $rootScope, $mdGesture) {
    var backdrop;

    return {
      themable: true,
      targetEvent: null,
      onShow: onShow,
      onRemove: onRemove,
      escapeToClose: true,
      disableParentScroll: true
    };

    function onShow(scope, element, options) {
      // Add a backdrop that will close on click
      backdrop = $compile('<md-backdrop class="md-opaque md-bottom-sheet-backdrop">')(scope);
      backdrop.on('click', function() {
        $timeout($mdBottomSheet.cancel);
      });

      $mdTheming.inherit(backdrop, options.parent);

      $animate.enter(backdrop, options.parent, null);

      var bottomSheet = new BottomSheet(element, options.parent);
      options.bottomSheet = bottomSheet;

      // Give up focus on calling item
      options.targetEvent && angular.element(options.targetEvent.target).blur();
      $mdTheming.inherit(bottomSheet.element, options.parent);

      if (options.disableParentScroll) {
        options.lastOverflow = options.parent.css('overflow');
        options.parent.css('overflow', 'hidden');
      }

      return $animate.enter(bottomSheet.element, options.parent)
        .then(function() {
          var focusable = angular.element(
            element[0].querySelector('button') ||
            element[0].querySelector('a') ||
            element[0].querySelector('[ng-click]')
          );
          focusable.focus();

          if (options.escapeToClose) {
            options.rootElementKeyupCallback = function(e) {
              if (e.keyCode === $mdConstant.KEY_CODE.ESCAPE) {
                $timeout($mdBottomSheet.cancel);
              }
            };
            $rootElement.on('keyup', options.rootElementKeyupCallback);
          }
        });

    }

    function onRemove(scope, element, options) {
      var bottomSheet = options.bottomSheet;


      $animate.leave(backdrop);
      return $animate.leave(bottomSheet.element).then(function() {
        if (options.disableParentScroll) {
          options.parent.css('overflow', options.lastOverflow);
          delete options.lastOverflow;
        }

        bottomSheet.cleanup();

        // Restore focus
        options.targetEvent && angular.element(options.targetEvent.target).focus();
      });
    }

    /**
     * BottomSheet class to apply bottom-sheet behavior to an element
     */
    function BottomSheet(element, parent) {
      var deregister = $mdGesture.register(parent, 'drag', { horizontal: false });
      parent.on('$md.dragstart', onDragStart)
        .on('$md.drag', onDrag)
        .on('$md.dragend', onDragEnd);

      return {
        element: element,
        cleanup: function cleanup() {
          deregister();
          parent.off('$md.dragstart', onDragStart)
            .off('$md.drag', onDrag)
            .off('$md.dragend', onDragEnd);
        }
      };

      function onDragStart(ev) {
        // Disable transitions on transform so that it feels fast
        element.css($mdConstant.CSS.TRANSITION_DURATION, '0ms');
      }

      function onDrag(ev) {
        var transform = ev.pointer.distanceY;
        if (transform < 5) {
          // Slow down drag when trying to drag up, and stop after PADDING
          transform = Math.max(-PADDING, transform / 2);
        }
        element.css($mdConstant.CSS.TRANSFORM, 'translate3d(0,' + (PADDING + transform) + 'px,0)');
      }

      function onDragEnd(ev) {
        if (ev.pointer.distanceY > 0 &&
            (ev.pointer.distanceY > 20 || Math.abs(ev.pointer.velocityY) > CLOSING_VELOCITY)) {
          var distanceRemaining = element.prop('offsetHeight') - ev.pointer.distanceY;
          var transitionDuration = Math.min(distanceRemaining / ev.pointer.velocityY * 0.75, 500);
          element.css($mdConstant.CSS.TRANSITION_DURATION, transitionDuration + 'ms');
          $timeout($mdBottomSheet.cancel);
        } else {
          element.css($mdConstant.CSS.TRANSITION_DURATION, '');
          element.css($mdConstant.CSS.TRANSFORM, '');
        }
      }
    }

  }

}
MdBottomSheetProvider.$inject = ["$$interimElementProvider"];

})();
