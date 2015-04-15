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
 * @name material.components.slider
 */
angular.module('material.components.slider', [
  'material.core'
])
  .directive('mdSlider', SliderDirective);

/**
 * @ngdoc directive
 * @name mdSlider
 * @module material.components.slider
 * @restrict E
 * @description
 * The `<md-slider>` component allows the user to choose from a range of
 * values.
 *
 * It has two modes: 'normal' mode, where the user slides between a wide range
 * of values, and 'discrete' mode, where the user slides between only a few
 * select values.
 *
 * To enable discrete mode, add the `md-discrete` attribute to a slider,
 * and use the `step` attribute to change the distance between
 * values the user is allowed to pick.
 *
 * @usage
 * <h4>Normal Mode</h4>
 * <hljs lang="html">
 * <md-slider ng-model="myValue" min="5" max="500">
 * </md-slider>
 * </hljs>
 * <h4>Discrete Mode</h4>
 * <hljs lang="html">
 * <md-slider md-discrete ng-model="myDiscreteValue" step="10" min="10" max="130">
 * </md-slider>
 * </hljs>
 *
 * @param {boolean=} md-discrete Whether to enable discrete mode.
 * @param {number=} step The distance between values the user is allowed to pick. Default 1.
 * @param {number=} min The minimum value the user is allowed to pick. Default 0.
 * @param {number=} max The maximum value the user is allowed to pick. Default 100.
 */
function SliderDirective($mdTheming) {
  return {
    scope: {},
    require: ['?ngModel', 'mdSlider'],
    controller: SliderController,
    template:
      '<div class="md-track-container">' +
        '<div class="md-track"></div>' +
        '<div class="md-track md-track-fill"></div>' +
        '<div class="md-track-ticks"></div>' +
      '</div>' +
      '<div class="md-thumb-container">' +
        '<div class="md-thumb"></div>' +
        '<div class="md-focus-thumb"></div>' +
        '<div class="md-focus-ring"></div>' +
        '<div class="md-sign">' +
          '<span class="md-thumb-text"></span>' +
        '</div>' +
        '<div class="md-disabled-thumb"></div>' +
      '</div>',
    link: postLink
  };

  function postLink(scope, element, attr, ctrls) {
    $mdTheming(element);
    var ngModelCtrl = ctrls[0] || {
      // Mock ngModelController if it doesn't exist to give us
      // the minimum functionality needed
      $setViewValue: function(val) {
        this.$viewValue = val;
        this.$viewChangeListeners.forEach(function(cb) { cb(); });
      },
      $parsers: [],
      $formatters: [],
      $viewChangeListeners: []
    };

    var sliderCtrl = ctrls[1];
    sliderCtrl.init(ngModelCtrl);
  }
}
SliderDirective.$inject = ["$mdTheming"];

/**
 * We use a controller for all the logic so that we can expose a few
 * things to unit tests
 */
function SliderController($scope, $element, $attrs, $$rAF, $window, $mdAria, $mdUtil, $mdConstant) {

  this.init = function init(ngModelCtrl) {
    var thumb = angular.element($element[0].querySelector('.md-thumb'));
    var thumbText = angular.element($element[0].querySelector('.md-thumb-text'));
    var thumbContainer = thumb.parent();
    var trackContainer = angular.element($element[0].querySelector('.md-track-container'));
    var activeTrack = angular.element($element[0].querySelector('.md-track-fill'));
    var tickContainer = angular.element($element[0].querySelector('.md-track-ticks'));
    var throttledRefreshDimensions = $mdUtil.throttle(refreshSliderDimensions, 5000);

    // Default values, overridable by $attrss
    $attrs.min ? $attrs.$observe('min', updateMin) : updateMin(0);
    $attrs.max ? $attrs.$observe('max', updateMax) : updateMax(100);
    $attrs.step ? $attrs.$observe('step', updateStep) : updateStep(1);

    // We have to manually stop the $watch on ngDisabled because it exists
    // on the parent $scope, and won't be automatically destroyed when
    // the component is destroyed.
    var stopDisabledWatch = angular.noop;
    if ($attrs.ngDisabled) {
      stopDisabledWatch = $scope.$parent.$watch($attrs.ngDisabled, updateAriaDisabled);
    }

    $mdAria.expect($element, 'aria-label');

    $element.attr('tabIndex', 0);
    $element.attr('role', 'slider');
    $element.on('keydown', keydownListener);

    var hammertime = new Hammer($element[0], {
      recognizers: [
        [Hammer.Pan, { direction: Hammer.DIRECTION_HORIZONTAL }]
      ]
    });
    hammertime.on('hammer.input', onInput);
    hammertime.on('panstart', onPanStart);
    hammertime.on('pan', onPan);
    hammertime.on('panend', onPanEnd);

    // On resize, recalculate the slider's dimensions and re-render
    function updateAll() {
      refreshSliderDimensions();
      ngModelRender();
      redrawTicks();
    }
    setTimeout(updateAll);

    var debouncedUpdateAll = $$rAF.debounce(updateAll);
    angular.element($window).on('resize', debouncedUpdateAll);

    $scope.$on('$destroy', function() {
      angular.element($window).off('resize', debouncedUpdateAll);
      hammertime.destroy();
      stopDisabledWatch();
    });

    ngModelCtrl.$render = ngModelRender;
    ngModelCtrl.$viewChangeListeners.push(ngModelRender);
    ngModelCtrl.$formatters.push(minMaxValidator);
    ngModelCtrl.$formatters.push(stepValidator);

    /**
     * Attributes
     */
    var min;
    var max;
    var step;
    function updateMin(value) {
      min = parseFloat(value);
      $element.attr('aria-valuemin', value);
      updateAll();
    }
    function updateMax(value) {
      max = parseFloat(value);
      $element.attr('aria-valuemax', value);
      updateAll();
    }
    function updateStep(value) {
      step = parseFloat(value);
      redrawTicks();
    }
    function updateAriaDisabled(isDisabled) {
      $element.attr('aria-disabled', !!isDisabled);
    }

    // Draw the ticks with canvas.
    // The alternative to drawing ticks with canvas is to draw one $element for each tick,
    // which could quickly become a performance bottleneck.
    var tickCanvas, tickCtx;
    function redrawTicks() {
      if (!angular.isDefined($attrs.mdDiscrete)) return;

      var numSteps = Math.floor( (max - min) / step );
      if (!tickCanvas) {
        var trackTicksStyle = $window.getComputedStyle(tickContainer[0]);
        tickCanvas = angular.element('<canvas style="position:absolute;">');
        tickCtx = tickCanvas[0].getContext('2d');
        tickCtx.fillStyle = trackTicksStyle.backgroundColor || 'black';
        tickContainer.append(tickCanvas);
      }
      var dimensions = getSliderDimensions();
      tickCanvas[0].width = dimensions.width;
      tickCanvas[0].height = dimensions.height;

      var distance;
      for (var i = 0; i <= numSteps; i++) {
        distance = Math.floor(dimensions.width * (i / numSteps));
        tickCtx.fillRect(distance - 1, 0, 2, dimensions.height);
      }
    }


    /**
     * Refreshing Dimensions
     */
    var sliderDimensions = {};
    refreshSliderDimensions();
    function refreshSliderDimensions() {
      sliderDimensions = trackContainer[0].getBoundingClientRect();
    }
    function getSliderDimensions() {
      throttledRefreshDimensions();
      return sliderDimensions;
    }

    /**
     * left/right arrow listener
     */
    function keydownListener(ev) {
      if($element[0].hasAttribute('disabled')) {
        return;
      }

      var changeAmount;
      if (ev.keyCode === $mdConstant.KEY_CODE.LEFT_ARROW) {
        changeAmount = -step;
      } else if (ev.keyCode === $mdConstant.KEY_CODE.RIGHT_ARROW) {
        changeAmount = step;
      }
      if (changeAmount) {
        if (ev.metaKey || ev.ctrlKey || ev.altKey) {
          changeAmount *= 4;
        }
        ev.preventDefault();
        ev.stopPropagation();
        $scope.$evalAsync(function() {
          setModelValue(ngModelCtrl.$viewValue + changeAmount);
        });
      }
    }

    /**
     * ngModel setters and validators
     */
    function setModelValue(value) {
      ngModelCtrl.$setViewValue( minMaxValidator(stepValidator(value)) );
    }
    function ngModelRender() {

      if (isNaN(ngModelCtrl.$viewValue)) {
        ngModelCtrl.$viewValue = ngModelCtrl.$modelValue;
      }

      var percent = (ngModelCtrl.$viewValue - min) / (max - min);
      $scope.modelValue = ngModelCtrl.$viewValue;
      $element.attr('aria-valuenow', ngModelCtrl.$viewValue);
      setSliderPercent(percent);
      thumbText.text( ngModelCtrl.$viewValue );
    }

    function minMaxValidator(value) {
      if (angular.isNumber(value)) {
        return Math.max(min, Math.min(max, value));
      }
    }
    function stepValidator(value) {
      if (angular.isNumber(value)) {
        return Math.round(value / step) * step;
      }
    }

    /**
     * @param percent 0-1
     */
    function setSliderPercent(percent) {
      activeTrack.css('width', (percent * 100) + '%');
      thumbContainer.css(
        $mdConstant.CSS.TRANSFORM,
        'translate3d(' + getSliderDimensions().width * percent + 'px,0,0)'
      );
      $element.toggleClass('md-min', percent === 0);
    }


    /**
     * Slide listeners
     */
    var isSliding = false;
    var isDiscrete = angular.isDefined($attrs.mdDiscrete);

    function onInput(ev) {
      if (!isSliding && ev.eventType === Hammer.INPUT_START &&
          !$element[0].hasAttribute('disabled')) {

        isSliding = true;

        $element.addClass('active');
        $element[0].focus();
        refreshSliderDimensions();

        onPan(ev);

        ev.srcEvent.stopPropagation();

      } else if (isSliding && ev.eventType === Hammer.INPUT_END) {

        if ( isSliding && isDiscrete ) onPanEnd(ev);
        isSliding = false;

        $element.removeClass('panning active');
      }
    }
    function onPanStart() {
      if (!isSliding) return;
      $element.addClass('panning');
    }
    function onPan(ev) {
      if (!isSliding) return;

      // While panning discrete, update only the
      // visual positioning but not the model value.

      if ( isDiscrete ) adjustThumbPosition( ev.center.x );
      else              doSlide( ev.center.x );

      ev.preventDefault();
      ev.srcEvent.stopPropagation();
    }

    function onPanEnd(ev) {
      if ( isDiscrete && !$element[0].hasAttribute('disabled') ) {
        // Convert exact to closest discrete value.
        // Slide animate the thumb... and then update the model value.

        var exactVal = percentToValue( positionToPercent( ev.center.x ));
        var closestVal = minMaxValidator( stepValidator(exactVal) );

        setSliderPercent( valueToPercent(closestVal));
        $$rAF(function(){
          setModelValue( closestVal );
        });

        ev.preventDefault();
        ev.srcEvent.stopPropagation();
      }
    }

    /**
     * Expose for testing
     */
    this._onInput = onInput;
    this._onPanStart = onPanStart;
    this._onPan = onPan;

    /**
     * Slide the UI by changing the model value
     * @param x
     */
    function doSlide( x ) {
      $scope.$evalAsync( function() {
        setModelValue( percentToValue( positionToPercent(x) ));
      });
    }

    /**
     * Slide the UI without changing the model (while dragging/panning)
     * @param x
     */
    function adjustThumbPosition( x ) {
      var exactVal = percentToValue( positionToPercent( x ));
      var closestVal = minMaxValidator( stepValidator(exactVal) );
      setSliderPercent( positionToPercent(x) );
      thumbText.text( closestVal );
    }

    /**
     * Convert horizontal position on slider to percentage value of offset from beginning...
     * @param x
     * @returns {number}
     */
    function positionToPercent( x ) {
      return Math.max(0, Math.min(1, (x - sliderDimensions.left) / (sliderDimensions.width)));
    }

    /**
     * Convert percentage offset on slide to equivalent model value
     * @param percent
     * @returns {*}
     */
    function percentToValue( percent ) {
      return (min + percent * (max - min));
    }

    function valueToPercent( val ) {
      return (val - min)/(max - min);
    }

  };
}
SliderController.$inject = ["$scope", "$element", "$attrs", "$$rAF", "$window", "$mdAria", "$mdUtil", "$mdConstant"];
})();
