/**
 * @license AngularJS v1.3.15
 * (c) 2010-2014 Google, Inc. http://angularjs.org
 * License: MIT
 */
(function(window, angular, undefined) {'use strict';

/**
 * @ngdoc module
 * @name ngAria
 * @description
 *
 * The `ngAria` module provides support for common
 * [<abbr title="Accessible Rich Internet Applications">ARIA</abbr>](http://www.w3.org/TR/wai-aria/)
 * attributes that convey state or semantic information about the application for users
 * of assistive technologies, such as screen readers.
 *
 * <div doc-module-components="ngAria"></div>
 *
 * ## Usage
 *
 * For ngAria to do its magic, simply include the module as a dependency. The directives supported
 * by ngAria are:
 * `ngModel`, `ngDisabled`, `ngShow`, `ngHide`, `ngClick`, `ngDblClick`, and `ngMessages`.
 *
 * Below is a more detailed breakdown of the attributes handled by ngAria:
 *
 * | Directive                                   | Supported Attributes                                                                   |
 * |---------------------------------------------|----------------------------------------------------------------------------------------|
 * | {@link ng.directive:ngDisabled ngDisabled}  | aria-disabled                                                                          |
 * | {@link ng.directive:ngShow ngShow}          | aria-hidden                                                                            |
 * | {@link ng.directive:ngHide ngHide}          | aria-hidden                                                                            |
 * | {@link ng.directive:ngDblclick ngDblclick}  | tabindex                                                                               |
 * | {@link module:ngMessages ngMessages}        | aria-live                                                                              |
 * | {@link ng.directive:ngModel ngModel}        | aria-checked, aria-valuemin, aria-valuemax, aria-valuenow, aria-invalid, aria-required, input roles |
 * | {@link ng.directive:ngClick ngClick}        | tabindex, keypress event, button role                                                               |
 *
 * Find out more information about each directive by reading the
 * {@link guide/accessibility ngAria Developer Guide}.
 *
 * ##Example
 * Using ngDisabled with ngAria:
 * ```html
 * <md-checkbox ng-disabled="disabled">
 * ```
 * Becomes:
 * ```html
 * <md-checkbox ng-disabled="disabled" aria-disabled="true">
 * ```
 *
 * ##Disabling Attributes
 * It's possible to disable individual attributes added by ngAria with the
 * {@link ngAria.$ariaProvider#config config} method. For more details, see the
 * {@link guide/accessibility Developer Guide}.
 */
 /* global -ngAriaModule */
var ngAriaModule = angular.module('ngAria', ['ng']).
                        provider('$aria', $AriaProvider);

/**
 * @ngdoc provider
 * @name $ariaProvider
 *
 * @description
 *
 * Used for configuring the ARIA attributes injected and managed by ngAria.
 *
 * ```js
 * angular.module('myApp', ['ngAria'], function config($ariaProvider) {
 *   $ariaProvider.config({
 *     ariaValue: true,
 *     tabindex: false
 *   });
 * });
 *```
 *
 * ## Dependencies
 * Requires the {@link ngAria} module to be installed.
 *
 */
function $AriaProvider() {
  var config = {
    ariaHidden: true,
    ariaChecked: true,
    ariaDisabled: true,
    ariaRequired: true,
    ariaInvalid: true,
    ariaMultiline: true,
    ariaValue: true,
    tabindex: true,
    bindKeypress: true
  };

  /**
   * @ngdoc method
   * @name $ariaProvider#config
   *
   * @param {object} config object to enable/disable specific ARIA attributes
   *
   *  - **ariaHidden** – `{boolean}` – Enables/disables aria-hidden tags
   *  - **ariaChecked** – `{boolean}` – Enables/disables aria-checked tags
   *  - **ariaDisabled** – `{boolean}` – Enables/disables aria-disabled tags
   *  - **ariaRequired** – `{boolean}` – Enables/disables aria-required tags
   *  - **ariaInvalid** – `{boolean}` – Enables/disables aria-invalid tags
   *  - **ariaMultiline** – `{boolean}` – Enables/disables aria-multiline tags
   *  - **ariaValue** – `{boolean}` – Enables/disables aria-valuemin, aria-valuemax and aria-valuenow tags
   *  - **tabindex** – `{boolean}` – Enables/disables tabindex tags
   *  - **bindKeypress** – `{boolean}` – Enables/disables keypress event binding on `&lt;div&gt;` and
   *    `&lt;li&gt;` elements with ng-click
   *
   * @description
   * Enables/disables various ARIA attributes
   */
  this.config = function(newConfig) {
    config = angular.extend(config, newConfig);
  };

  function watchExpr(attrName, ariaAttr, negate) {
    return function(scope, elem, attr) {
      var ariaCamelName = attr.$normalize(ariaAttr);
      if (config[ariaCamelName] && !attr[ariaCamelName]) {
        scope.$watch(attr[attrName], function(boolVal) {
          if (negate) {
            boolVal = !boolVal;
          }
          elem.attr(ariaAttr, boolVal);
        });
      }
    };
  }

  /**
   * @ngdoc service
   * @name $aria
   *
   * @description
   * @priority 200
   *
   * The $aria service contains helper methods for applying common
   * [ARIA](http://www.w3.org/TR/wai-aria/) attributes to HTML directives.
   *
   * ngAria injects common accessibility attributes that tell assistive technologies when HTML
   * elements are enabled, selected, hidden, and more. To see how this is performed with ngAria,
   * let's review a code snippet from ngAria itself:
   *
   *```js
   * ngAriaModule.directive('ngDisabled', ['$aria', function($aria) {
   *   return $aria.$$watchExpr('ngDisabled', 'aria-disabled');
   * }])
   *```
   * Shown above, the ngAria module creates a directive with the same signature as the
   * traditional `ng-disabled` directive. But this ngAria version is dedicated to
   * solely managing accessibility attributes. The internal `$aria` service is used to watch the
   * boolean attribute `ngDisabled`. If it has not been explicitly set by the developer,
   * `aria-disabled` is injected as an attribute with its value synchronized to the value in
   * `ngDisabled`.
   *
   * Because ngAria hooks into the `ng-disabled` directive, developers do not have to do
   * anything to enable this feature. The `aria-disabled` attribute is automatically managed
   * simply as a silent side-effect of using `ng-disabled` with the ngAria module.
   *
   * The full list of directives that interface with ngAria:
   * * **ngModel**
   * * **ngShow**
   * * **ngHide**
   * * **ngClick**
   * * **ngDblclick**
   * * **ngMessages**
   * * **ngDisabled**
   *
   * Read the {@link guide/accessibility ngAria Developer Guide} for a thorough explanation of each
   * directive.
   *
   *
   * ## Dependencies
   * Requires the {@link ngAria} module to be installed.
   */
  this.$get = function() {
    return {
      config: function(key) {
        return config[key];
      },
      $$watchExpr: watchExpr
    };
  };
}


ngAriaModule.directive('ngShow', ['$aria', function($aria) {
  return $aria.$$watchExpr('ngShow', 'aria-hidden', true);
}])
.directive('ngHide', ['$aria', function($aria) {
  return $aria.$$watchExpr('ngHide', 'aria-hidden', false);
}])
.directive('ngModel', ['$aria', function($aria) {

  function shouldAttachAttr(attr, normalizedAttr, elem) {
    return $aria.config(normalizedAttr) && !elem.attr(attr);
  }

  function shouldAttachRole(role, elem) {
    return !elem.attr('role') && (elem.attr('type') === role) && (elem[0].nodeName !== 'INPUT');
  }

  function getShape(attr, elem) {
    var type = attr.type,
        role = attr.role;

    return ((type || role) === 'checkbox' || role === 'menuitemcheckbox') ? 'checkbox' :
           ((type || role) === 'radio'    || role === 'menuitemradio') ? 'radio' :
           (type === 'range'              || role === 'progressbar' || role === 'slider') ? 'range' :
           (type || role) === 'textbox'   || elem[0].nodeName === 'TEXTAREA' ? 'multiline' : '';
  }

  return {
    restrict: 'A',
    require: '?ngModel',
    priority: 200, //Make sure watches are fired after any other directives that affect the ngModel value
    link: function(scope, elem, attr, ngModel) {
      var shape = getShape(attr, elem);
      var needsTabIndex = shouldAttachAttr('tabindex', 'tabindex', elem);

      function ngAriaWatchModelValue() {
        return ngModel.$modelValue;
      }

      function getRadioReaction() {
        if (needsTabIndex) {
          needsTabIndex = false;
          return function ngAriaRadioReaction(newVal) {
            var boolVal = (attr.value == ngModel.$viewValue);
            elem.attr('aria-checked', boolVal);
            elem.attr('tabindex', 0 - !boolVal);
          };
        } else {
          return function ngAriaRadioReaction(newVal) {
            elem.attr('aria-checked', (attr.value == ngModel.$viewValue));
          };
        }
      }

      function ngAriaCheckboxReaction(newVal) {
        elem.attr('aria-checked', !ngModel.$isEmpty(ngModel.$viewValue));
      }

      switch (shape) {
        case 'radio':
        case 'checkbox':
          if (shouldAttachRole(shape, elem)) {
            elem.attr('role', shape);
          }
          if (shouldAttachAttr('aria-checked', 'ariaChecked', elem)) {
            scope.$watch(ngAriaWatchModelValue, shape === 'radio' ?
                getRadioReaction() : ngAriaCheckboxReaction);
          }
          break;
        case 'range':
          if (shouldAttachRole(shape, elem)) {
            elem.attr('role', 'slider');
          }
          if ($aria.config('ariaValue')) {
            if (attr.min && !elem.attr('aria-valuemin')) {
              elem.attr('aria-valuemin', attr.min);
            }
            if (attr.max && !elem.attr('aria-valuemax')) {
              elem.attr('aria-valuemax', attr.max);
            }
            if (!elem.attr('aria-valuenow')) {
              scope.$watch(ngAriaWatchModelValue, function ngAriaValueNowReaction(newVal) {
                elem.attr('aria-valuenow', newVal);
              });
            }
          }
          break;
        case 'multiline':
          if (shouldAttachAttr('aria-multiline', 'ariaMultiline', elem)) {
            elem.attr('aria-multiline', true);
          }
          break;
      }

      if (needsTabIndex) {
        elem.attr('tabindex', 0);
      }

      if (ngModel.$validators.required && shouldAttachAttr('aria-required', 'ariaRequired', elem)) {
        scope.$watch(function ngAriaRequiredWatch() {
          return ngModel.$error.required;
        }, function ngAriaRequiredReaction(newVal) {
          elem.attr('aria-required', !!newVal);
        });
      }

      if (shouldAttachAttr('aria-invalid', 'ariaInvalid', elem)) {
        scope.$watch(function ngAriaInvalidWatch() {
          return ngModel.$invalid;
        }, function ngAriaInvalidReaction(newVal) {
          elem.attr('aria-invalid', !!newVal);
        });
      }
    }
  };
}])
.directive('ngDisabled', ['$aria', function($aria) {
  return $aria.$$watchExpr('ngDisabled', 'aria-disabled');
}])
.directive('ngMessages', function() {
  return {
    restrict: 'A',
    require: '?ngMessages',
    link: function(scope, elem, attr, ngMessages) {
      if (!elem.attr('aria-live')) {
        elem.attr('aria-live', 'assertive');
      }
    }
  };
})
.directive('ngClick',['$aria', '$parse', function($aria, $parse) {
  return {
    restrict: 'A',
    compile: function(elem, attr) {
      var fn = $parse(attr.ngClick, /* interceptorFn */ null, /* expensiveChecks */ true);
      return function(scope, elem, attr) {

        var nodeBlackList = ['BUTTON', 'A', 'INPUT', 'TEXTAREA'];

        function isNodeOneOf(elem, nodeTypeArray) {
          if (nodeTypeArray.indexOf(elem[0].nodeName) !== -1) {
            return true;
          }
        }
        if (!elem.attr('role') && !isNodeOneOf(elem, nodeBlackList)) {
          elem.attr('role', 'button');
        }

        if ($aria.config('tabindex') && !elem.attr('tabindex')) {
          elem.attr('tabindex', 0);
        }

        if ($aria.config('bindKeypress') && !attr.ngKeypress && !isNodeOneOf(elem, nodeBlackList)) {
          elem.on('keypress', function(event) {
            if (event.keyCode === 32 || event.keyCode === 13) {
              scope.$apply(callback);
            }

            function callback() {
              fn(scope, { $event: event });
            }
          });
        }
      };
    }
  };
}])
.directive('ngDblclick', ['$aria', function($aria) {
  return function(scope, elem, attr) {
    if ($aria.config('tabindex') && !elem.attr('tabindex')) {
      elem.attr('tabindex', 0);
    }
  };
}]);


})(window, window.angular);
