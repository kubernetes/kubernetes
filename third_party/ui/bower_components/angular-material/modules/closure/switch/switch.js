/*!
 * Angular Material Design
 * https://github.com/angular/material
 * @license MIT
 * v0.8.1
 */
goog.provide('ng.material.components.switch');
goog.require('ng.material.components.checkbox');
goog.require('ng.material.core');
(function() {
'use strict';

/**
 * @private
 * @ngdoc module
 * @name material.components.switch
 */

angular.module('material.components.switch', [
  'material.core',
  'material.components.checkbox'
])
  .directive('mdSwitch', MdSwitch);

/**
 * @private
 * @ngdoc directive
 * @module material.components.switch
 * @name mdSwitch
 * @restrict E
 *
 * The switch directive is used very much like the normal [angular checkbox](https://docs.angularjs.org/api/ng/input/input%5Bcheckbox%5D).
 *
 * As per the [material design spec](http://www.google.com/design/spec/style/color.html#color-ui-color-application)
 * the switch is in the accent color by default. The primary color palette may be used with
 * the `md-primary` class.
 *
 * @param {string} ng-model Assignable angular expression to data-bind to.
 * @param {string=} name Property name of the form under which the control is published.
 * @param {expression=} ng-true-value The value to which the expression should be set when selected.
 * @param {expression=} ng-false-value The value to which the expression should be set when not selected.
 * @param {string=} ng-change Angular expression to be executed when input changes due to user interaction with the input element.
 * @param {boolean=} md-no-ink Use of attribute indicates use of ripple ink effects.
 * @param {string=} aria-label Publish the button label used by screen-readers for accessibility. Defaults to the switch's text.
 *
 * @usage
 * <hljs lang="html">
 * <md-switch ng-model="isActive" aria-label="Finished?">
 *   Finished ?
 * </md-switch>
 *
 * <md-switch md-no-ink ng-model="hasInk" aria-label="No Ink Effects">
 *   No Ink Effects
 * </md-switch>
 *
 * <md-switch ng-disabled="true" ng-model="isDisabled" aria-label="Disabled">
 *   Disabled
 * </md-switch>
 *
 * </hljs>
 */
function MdSwitch(mdCheckboxDirective, $mdTheming, $mdUtil, $document, $mdConstant, $parse, $$rAF, $mdGesture) {
  var checkboxDirective = mdCheckboxDirective[0];

  return {
    restrict: 'E',
    transclude: true,
    template:
      '<div class="md-container">' +
        '<div class="md-bar"></div>' +
        '<div class="md-thumb-container">' +
          '<div class="md-thumb" md-ink-ripple md-ink-ripple-checkbox></div>' +
        '</div>'+
      '</div>' +
      '<div ng-transclude class="md-label">' +
      '</div>',
    require: '?ngModel',
    compile: compile
  };

  function compile(element, attr) {
    var checkboxLink = checkboxDirective.compile(element, attr);
    // no transition on initial load
    element.addClass('md-dragging');

    return function (scope, element, attr, ngModel) {
      ngModel = ngModel || $mdUtil.fakeNgModel();
      var disabledGetter = $parse(attr.ngDisabled);
      var thumbContainer = angular.element(element[0].querySelector('.md-thumb-container'));
      var switchContainer = angular.element(element[0].querySelector('.md-container'));

      // no transition on initial load
      $$rAF(function() {
        element.removeClass('md-dragging');
      });

      checkboxLink(scope, element, attr, ngModel);

      if (angular.isDefined(attr.ngDisabled)) {
        scope.$watch(disabledGetter, function(isDisabled) {
          element.attr('tabindex', isDisabled ? -1 : 0);
        });
      }

      // These events are triggered by setup drag
      $mdGesture.register(switchContainer, 'drag');
      switchContainer
        .on('$md.dragstart', onDragStart)
        .on('$md.drag', onDrag)
        .on('$md.dragend', onDragEnd);

      var drag;
      function onDragStart(ev) {
        // Don't go if ng-disabled===true
        if (disabledGetter(scope)) return;
        ev.stopPropagation();

        element.addClass('md-dragging');
        drag = {
          width: thumbContainer.prop('offsetWidth')
        };
        element.removeClass('transition');
      }

      function onDrag(ev) {
        if (!drag) return;
        ev.stopPropagation();
        ev.srcEvent && ev.srcEvent.preventDefault();

        var percent = ev.pointer.distanceX / drag.width;

        //if checked, start from right. else, start from left
        var translate = ngModel.$viewValue ?  1 + percent : percent;
        // Make sure the switch stays inside its bounds, 0-1%
        translate = Math.max(0, Math.min(1, translate));

        thumbContainer.css($mdConstant.CSS.TRANSFORM, 'translate3d(' + (100*translate) + '%,0,0)');
        drag.translate = translate;
      }

      function onDragEnd(ev) {
        if (!drag) return;
        ev.stopPropagation();

        element.removeClass('md-dragging');
        thumbContainer.css($mdConstant.CSS.TRANSFORM, '');

        // We changed if there is no distance (this is a click a click),
        // or if the drag distance is >50% of the total.
        var isChanged = ngModel.$viewValue ? drag.translate < 0.5 : drag.translate > 0.5;
        if (isChanged) {
          applyModelValue(!ngModel.$viewValue);
        }
        drag = null;
      }

      function applyModelValue(newValue) {
        scope.$apply(function() {
          ngModel.$setViewValue(newValue);
          ngModel.$render();
        });
      }

    };
  }


}
MdSwitch.$inject = ["mdCheckboxDirective", "$mdTheming", "$mdUtil", "$document", "$mdConstant", "$parse", "$$rAF", "$mdGesture"];

})();
