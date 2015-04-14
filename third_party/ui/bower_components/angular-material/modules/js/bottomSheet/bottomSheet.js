/*!
 * Angular Material Design
 * https://github.com/angular/material
 * @license MIT
 * v0.7.0-rc3
 */
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
 * ### Restrictions
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

  bottomSheetDefaults.$inject = ["$animate", "$mdConstant", "$timeout", "$$rAF", "$compile", "$mdTheming", "$mdBottomSheet", "$rootElement"];
  return $$interimElementProvider('$mdBottomSheet')
    .setDefaults({
      options: bottomSheetDefaults
    });

  /* @ngInject */
  function bottomSheetDefaults($animate, $mdConstant, $timeout, $$rAF, $compile, $mdTheming, $mdBottomSheet, $rootElement) {
    var backdrop;

    return {
      themable: true,
      targetEvent: null,
      onShow: onShow,
      onRemove: onRemove,
      escapeToClose: true
    };

    function onShow(scope, element, options) {
      // Add a backdrop that will close on click
      backdrop = $compile('<md-backdrop class="md-opaque md-bottom-sheet-backdrop">')(scope);
      backdrop.on('click touchstart', function() {
        $timeout($mdBottomSheet.cancel);
      });

      $mdTheming.inherit(backdrop, options.parent);

      $animate.enter(backdrop, options.parent, null);

      var bottomSheet = new BottomSheet(element);
      options.bottomSheet = bottomSheet;

      // Give up focus on calling item
      options.targetEvent && angular.element(options.targetEvent.target).blur();
      $mdTheming.inherit(bottomSheet.element, options.parent);

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
        bottomSheet.cleanup();

        // Restore focus
        options.targetEvent && angular.element(options.targetEvent.target).focus();
      });
    }

    /**
     * BottomSheet class to apply bottom-sheet behavior to an element
     */
    function BottomSheet(element) {
      var MAX_OFFSET = 80; // amount past the bottom of the element that we can drag down, this is same as in _bottomSheet.scss
      var WIGGLE_AMOUNT = 20; // point where it starts to get "harder" to drag
      var CLOSING_VELOCITY = 10; // how fast we need to flick down to close the sheet
      var startY, lastY, velocity, transitionDelay, startTarget;

      // coercion incase $mdCompiler returns multiple elements
      element = element.eq(0);

      element.on('touchstart', onTouchStart)
             .on('touchmove', onTouchMove)
             .on('touchend', onTouchEnd);

      return {
        element: element,
        cleanup: function cleanup() {
          element.off('touchstart', onTouchStart)
                 .off('touchmove', onTouchMove)
                 .off('touchend', onTouchEnd);
        }
      };

      function onTouchStart(e) {
        e.preventDefault();
        startTarget = e.target;
        startY = getY(e);

        // Disable transitions on transform so that it feels fast
        transitionDelay = element.css($mdConstant.CSS.TRANSITION_DURATION);
        element.css($mdConstant.CSS.TRANSITION_DURATION, '0s');
      }

      function onTouchEnd(e) {
        // Re-enable the transitions on transforms
        element.css($mdConstant.CSS.TRANSITION_DURATION, transitionDelay);

        var currentY = getY(e);
        // If we didn't scroll much, and we didn't change targets, assume its a click
        if ( Math.abs(currentY - startY) < 5  && e.target == startTarget) {
          angular.element(e.target).triggerHandler('click');
        } else {
          // If they went fast enough, trigger a close.
          if (velocity > CLOSING_VELOCITY) {
            $timeout($mdBottomSheet.cancel);

          // Otherwise, untransform so that we go back to our normal position
          } else {
            setTransformY(undefined);
          }
        }
      }

      function onTouchMove(e) {
        var currentY = getY(e);
        var delta = currentY - startY;

        velocity = currentY - lastY;
        lastY = currentY;

        // Do some conversion on delta to get a friction-like effect
        delta = adjustedDelta(delta);
        setTransformY(delta + MAX_OFFSET);
      }

      /**
       * Helper function to find the Y aspect of various touch events.
       **/
      function getY(e) {
        var touch = e.touches && e.touches.length ? e.touches[0] : e.changedTouches[0];
        return touch.clientY;
      }

      /**
       * Transform the element along the y-axis
       **/
      function setTransformY(amt) {
        if (amt === null || amt === undefined) {
          element.css($mdConstant.CSS.TRANSFORM, '');
        } else {
          element.css($mdConstant.CSS.TRANSFORM, 'translate3d(0, ' + amt + 'px, 0)');
        }
      }

      // Returns a new value for delta that will never exceed MAX_OFFSET_AMOUNT
      // Will get harder to exceed it as you get closer to it
      function adjustedDelta(delta) {
        if ( delta < 0  && delta < -MAX_OFFSET + WIGGLE_AMOUNT) {
          delta = -delta;
          var base = MAX_OFFSET - WIGGLE_AMOUNT;
          delta = Math.max(-MAX_OFFSET, -Math.min(MAX_OFFSET - 5, base + ( WIGGLE_AMOUNT * (delta - base)) / MAX_OFFSET) - delta / 50);
        }

        return delta;
      }
    }

  }

}
MdBottomSheetProvider.$inject = ["$$interimElementProvider"];

})();
