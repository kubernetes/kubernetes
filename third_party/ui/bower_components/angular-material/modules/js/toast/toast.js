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
 * @name material.components.toast
 * @description
 * Toast
 */
angular.module('material.components.toast', [
  'material.core',
  'material.components.swipe',
  'material.components.button'
])
  .directive('mdToast', MdToastDirective)
  .provider('$mdToast', MdToastProvider);

function MdToastDirective() {
  return {
    restrict: 'E'
  };
}

/**
 * @ngdoc service
 * @name $mdToast
 * @module material.components.toast
 *
 * @description
 * `$mdToast` is a service to build a toast nofication on any position
 * on the screen with an optional duration, and provides a simple promise API.
 *
 *
 * ### Restrictions on custom toasts
 * - The toast's template must have an outer `<md-toast>` element.
 * - For a toast action, use element with class `md-action`.
 * - Add the class `md-capsule` for curved corners.
 *
 * @usage
 * <hljs lang="html">
 * <div ng-controller="MyController">
 *   <md-button ng-click="openToast()">
 *     Open a Toast!
 *   </md-button>
 * </div>
 * </hljs>
 *
 * <hljs lang="js">
 * var app = angular.module('app', ['ngMaterial']);
 * app.controller('MyController', function($scope, $mdToast) {
 *   $scope.openToast = function($event) {
 *     $mdToast.show($mdToast.simple().content('Hello!'));
 *     // Could also do $mdtoast.showSimple('Hello');
 *   };
 * });
 * </hljs>
 */

/**
 * @ngdoc method
 * @name $mdToast#showSimple
 * 
 * @description
 * Convenience method which builds and shows a simple toast.
 *
 * @returns {promise} A promise that can be resolved with `$mdToast.hide()` or
 * rejected with `$mdToast.cancel()`.
 *
 */

 /**
 * @ngdoc method
 * @name $mdToast#simple
 *
 * @description
 * Builds a preconfigured toast.
 *
 * @returns {obj} a `$mdToastPreset` with the chainable configuration methods:
 *
 * - $mdToastPreset#content(string) - sets toast content to string
 * - $mdToastPreset#action(string) - adds an action button, which resolves the promise returned from `show()` if clicked.
 * - $mdToastPreset#highlightAction(boolean) - sets action button to be highlighted
 * - $mdToastPreset#capsule(boolean) - adds 'md-capsule' class to the toast (curved corners)
 */

 /**
 * @ngdoc method
 * @name $mdToast#build
 *
 * @description
 * Creates a custom `$mdToastPreset` that you can configure.
 *
 * @returns {obj} a `$mdToastPreset` with the chainable configuration methods for shows' options (see below).
 */

 /**
 * @ngdoc method
 * @name $mdToast#show
 *
 * @description Shows the toast.
 *
 * @param {object} optionsOrPreset Either provide an `$mdToastPreset` returned from `simple()`
 * and `build()`, or an options object with the following properties:
 *
 *   - `templateUrl` - `{string=}`: The url of an html template file that will
 *     be used as the content of the toast. Restrictions: the template must
 *     have an outer `md-toast` element.
 *   - `template` - `{string=}`: Same as templateUrl, except this is an actual
 *     template string.
 *   - `hideDelay` - `{number=}`: How many milliseconds the toast should stay
 *     active before automatically closing.  Set to 0 or false to have the toast stay open until
 *     closed manually. Default: 3000.
 *   - `position` - `{string=}`: Where to place the toast. Available: any combination
 *     of 'bottom', 'left', 'top', 'right', 'fit'. Default: 'bottom left'.
 *   - `controller` - `{string=}`: The controller to associate with this toast.
 *     The controller will be injected the local `$hideToast`, which is a function
 *     used to hide the toast.
 *   - `locals` - `{string=}`: An object containing key/value pairs. The keys will
 *     be used as names of values to inject into the controller. For example,
 *     `locals: {three: 3}` would inject `three` into the controller with the value
 *     of 3.
 *   - `resolve` - `{object=}`: Similar to locals, except it takes promises as values
 *     and the toast will not open until the promises resolve.
 *   - `controllerAs` - `{string=}`: An alias to assign the controller to on the scope.
 *   - `parent` - `{element=}`: The element to append the toast to. Defaults to appending
 *     to the root element of the application.
 *
 * @returns {promise} A promise that can be resolved with `$mdToast.hide()` or
 * rejected with `$mdToast.cancel()`.
 */

/**
 * @ngdoc method
 * @name $mdToast#hide
 *
 * @description
 * Hide an existing toast and resolve the promise returned from `$mdToast.show()`.
 *
 * @param {*=} response An argument for the resolved promise.
 *
 */

/**
 * @ngdoc method
 * @name $mdToast#cancel
 *
 * @description
 * Hide the existing toast and reject the promise returned from
 * `$mdToast.show()`.
 *
 * @param {*=} response An argument for the rejected promise.
 *
 */

function MdToastProvider($$interimElementProvider) {

  toastDefaultOptions.$inject = ["$timeout", "$animate", "$mdSwipe", "$mdTheming", "$mdToast"];
  return $$interimElementProvider('$mdToast')
    .setDefaults({
      methods: ['position', 'hideDelay', 'capsule'],
      options: toastDefaultOptions
    })
    .addPreset('simple', {
      argOption: 'content',
      methods: ['content', 'action', 'highlightAction'],
      options: /* @ngInject */ ["$mdToast", function($mdToast) {
        return {
          template: [
            '<md-toast ng-class="{\'md-capsule\': toast.capsule}">',
              '<span flex>{{ toast.content }}</span>',
              '<md-button ng-if="toast.action" ng-click="toast.resolve()" ng-class="{\'md-action\': toast.highlightAction}">',
                '{{ toast.action }}',
              '</md-button>',
            '</md-toast>'
          ].join(''),
          controller: function mdToastCtrl() {
            this.resolve = function() {
              $mdToast.hide();
            };
          },
          controllerAs: 'toast',
          bindToController: true
        };
      }]
    });

  /* @ngInject */
  function toastDefaultOptions($timeout, $animate, $mdSwipe, $mdTheming, $mdToast) {
    return {
      onShow: onShow,
      onRemove: onRemove,
      position: 'bottom left',
      themable: true,
      hideDelay: 3000
    };

    function onShow(scope, element, options) {
      // 'top left' -> 'md-top md-left'
      element.addClass(options.position.split(' ').map(function(pos) {
        return 'md-' + pos;
      }).join(' '));
      options.parent.addClass(toastOpenClass(options.position));

      var configureSwipe = $mdSwipe(scope, 'swipeleft swiperight');
      options.detachSwipe = configureSwipe(element, function(ev) {
        //Add swipeleft/swiperight class to element so it can animate correctly
        element.addClass('md-' + ev.type);
        $timeout($mdToast.cancel);
      });

      return $animate.enter(element, options.parent);
    }

    function onRemove(scope, element, options) {
      options.detachSwipe();
      options.parent.removeClass(toastOpenClass(options.position));
      return $animate.leave(element);
    }

    function toastOpenClass(position) {
      return 'md-toast-open-' +
        (position.indexOf('top') > -1 ? 'top' : 'bottom');
    }
  }

}
MdToastProvider.$inject = ["$$interimElementProvider"];

})();
