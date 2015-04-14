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
 * @name material.components.dialog
 */
angular.module('material.components.dialog', [
  'material.core',
  'material.components.backdrop'
])
  .directive('mdDialog', MdDialogDirective)
  .provider('$mdDialog', MdDialogProvider);

function MdDialogDirective($$rAF, $mdTheming) {
  return {
    restrict: 'E',
    link: function(scope, element, attr) {
      $mdTheming(element);
      $$rAF(function() {
        var content = element[0].querySelector('md-content');
        if (content && content.scrollHeight > content.clientHeight) {
          element.addClass('md-content-overflow');
        }
      });
    }
  };
}
MdDialogDirective.$inject = ["$$rAF", "$mdTheming"];

/**
 * @ngdoc service
 * @name $mdDialog
 * @module material.components.dialog
 *
 * @description
 * `$mdDialog` opens a dialog over the app and provides a simple promise API.
 *
 * ### Restrictions
 *
 * - The dialog is always given an isolate scope.
 * - The dialog's template must have an outer `<md-dialog>` element.
 *   Inside, use an `<md-content>` element for the dialog's content, and use
 *   an element with class `md-actions` for the dialog's actions.
 *
 * @usage
 * ##### HTML
 *
 * <hljs lang="html">
 * <div  ng-app="demoApp" ng-controller="EmployeeController">
 *   <md-button ng-click="showAlert()" class="md-raised md-warn">
 *     Employee Alert!
 *   </md-button>
 *   <md-button ng-click="closeAlert()" ng-disabled="!hasAlert()" class="md-raised">
 *     Close Alert
 *   </md-button>
 *   <md-button ng-click="showGreeting($event)" class="md-raised md-primary" >
 *     Greet Employee
 *   </md-button>
 * </div>
 * </hljs>
 *
 * ##### JavaScript
 *
 * <hljs lang="js">
 * (function(angular, undefined){
 *   "use strict";
 *
 *   angular
 *     .module('demoApp', ['ngMaterial'])
 *     .controller('EmployeeController', EmployeeEditor)
 *     .controller('GreetingController', GreetingController);
 *
 *   // Fictitious Employee Editor to show how to use simple and complex dialogs.
 *
 *   function EmployeeEditor($scope, $mdDialog) {
 *     var alert;
 *
 *     $scope.showAlert = showAlert;
 *     $scope.closeAlert = closeAlert;
 *     $scope.showGreeting = showCustomGreeting;
 *
 *     $scope.hasAlert = function() { return !!alert };
 *     $scope.userName = $scope.userName || 'Bobby';
 *
 *     // Dialog #1 - Show simple alert dialog and cache
 *     // reference to dialog instance
 *
 *     function showAlert() {
 *       alert = $mdDialog.alert()
 *         .title('Attention, ' + $scope.userName)
 *         .content('This is an example of how easy dialogs can be!')
 *         .ok('Close');
 *
 *       $mdDialog
 *           .show( alert )
 *           .finally(function() {
 *             alert = undefined;
 *           });
 *     }
 *
 *     // Close the specified dialog instance and resolve with 'finished' flag
 *     // Normally this is not needed, just use '$mdDialog.hide()' to close
 *     // the most recent dialog popup.
 *
 *     function closeAlert() {
 *       $mdDialog.hide( alert, "finished" );
 *       alert = undefined;
 *     }
 *
 *     // Dialog #2 - Demonstrate more complex dialogs construction and popup.
 *
 *     function showCustomGreeting($event) {
 *         $mdDialog.show({
 *           targetEvent: $event,
 *           template:
 *             '<md-dialog>' +
 *
 *             '  <md-content>Hello {{ employee }}!</md-content>' +
 *
 *             '  <div class="md-actions">' +
 *             '    <md-button ng-click="closeDialog()">' +
 *             '      Close Greeting' +
 *
 *             '    </md-button>' +
 *             '  </div>' +
 *             '</md-dialog>',
 *           controller: 'GreetingController',
 *           onComplete: afterShowAnimation,
 *           locals: { employee: $scope.userName }
 *         });
 *
 *         // When the 'enter' animation finishes...
 *
 *         function afterShowAnimation(scope, element, options) {
 *            // post-show code here: DOM element focus, etc.
 *         }
 *     }
 *   }
 *
 *   // Greeting controller used with the more complex 'showCustomGreeting()' custom dialog
 *
 *   function GreetingController($scope, $mdDialog, employee) {
 *     // Assigned from construction <code>locals</code> options...
 *     $scope.employee = employee;
 *
 *     $scope.closeDialog = function() {
 *       // Easily hides most recent dialog shown...
 *       // no specific instance reference is needed.
 *       $mdDialog.hide();
 *     };
 *   }
 *
 * })(angular);
 * </hljs>
 */

 /**
 * @ngdoc method
 * @name $mdDialog#alert
 *
 * @description
 * Builds a preconfigured dialog with the specified message.
 *
 * @returns {obj} an `$mdDialogPreset` with the chainable configuration methods:
 *
 * - $mdDialogPreset#title(string) - sets title to string
 * - $mdDialogPreset#content(string) - sets content / message to string
 * - $mdDialogPreset#ok(string) - sets okay button text to string
 *
 */

 /**
 * @ngdoc method
 * @name $mdDialog#confirm
 *
 * @description
 * Builds a preconfigured dialog with the specified message. You can call show and the promise returned
 * will be resolved only if the user clicks the confirm action on the dialog.
 *
 * @returns {obj} an `$mdDialogPreset` with the chainable configuration methods:
 *
 * Additionally, it supports the following methods:
 *
 * - $mdDialogPreset#title(string) - sets title to string
 * - $mdDialogPreset#content(string) - sets content / message to string
 * - $mdDialogPreset#ok(string) - sets okay button text to string
 * - $mdDialogPreset#cancel(string) - sets cancel button text to string
 *
 */

/**
 * @ngdoc method
 * @name $mdDialog#show
 *
 * @description
 * Show a dialog with the specified options.
 *
 * @param {object} optionsOrPreset Either provide an `$mdDialogPreset` returned from `alert()`,
 * `confirm()` or an options object with the following properties:
 *   - `templateUrl` - `{string=}`: The url of a template that will be used as the content
 *   of the dialog.
 *   - `template` - `{string=}`: Same as templateUrl, except this is an actual template string.
 *   - `targetEvent` - `{DOMClickEvent=}`: A click's event object. When passed in as an option,
 *     the location of the click will be used as the starting point for the opening animation
 *     of the the dialog.
 *   - `disableParentScroll` - `{boolean=}`: Whether to disable scrolling while the dialog is open.
 *     Default true.
 *   - `hasBackdrop` - `{boolean=}`: Whether there should be an opaque backdrop behind the dialog.
 *     Default true.
 *   - `clickOutsideToClose` - `{boolean=}`: Whether the user can click outside the dialog to
 *     close it. Default true.
 *   - `escapeToClose` - `{boolean=}`: Whether the user can press escape to close the dialog.
 *     Default true.
 *   - `controller` - `{string=}`: The controller to associate with the dialog. The controller
 *     will be injected with the local `$hideDialog`, which is a function used to hide the dialog.
 *   - `locals` - `{object=}`: An object containing key/value pairs. The keys will be used as names
 *     of values to inject into the controller. For example, `locals: {three: 3}` would inject
 *     `three` into the controller, with the value 3. If `bindToController` is true, they will be
 *     copied to the controller instead.
 *   - `bindToController` - `bool`: bind the locals to the controller, instead of passing them in
 *   - `resolve` - `{object=}`: Similar to locals, except it takes promises as values, and the
 *     dialog will not open until all of the promises resolve.
 *   - `controllerAs` - `{string=}`: An alias to assign the controller to on the scope.
 *   - `parent` - `{element=}`: The element to append the dialog to. Defaults to appending
 *     to the root element of the application.
 *   - `onComplete` `{function=}`: Callback function used to announce when the show() action is
 *     finished.
 *
 * @returns {promise} A promise that can be resolved with `$mdDialog.hide()` or
 * rejected with `mdDialog.cancel()`.
 */

/**
 * @ngdoc method
 * @name $mdDialog#hide
 *
 * @description
 * Hide an existing dialog and resolve the promise returned from `$mdDialog.show()`.
 *
 * @param {*=} response An argument for the resolved promise.
 */

/**
 * @ngdoc method
 * @name $mdDialog#cancel
 *
 * @description
 * Hide an existing dialog and reject the promise returned from `$mdDialog.show()`.
 *
 * @param {*=} response An argument for the rejected promise.
 */

function MdDialogProvider($$interimElementProvider) {

  var alertDialogMethods = ['title', 'content', 'ariaLabel', 'ok'];

  advancedDialogOptions.$inject = ["$mdDialog"];
  dialogDefaultOptions.$inject = ["$timeout", "$rootElement", "$compile", "$animate", "$mdAria", "$document", "$mdUtil", "$mdConstant", "$mdTheming", "$$rAF", "$q", "$mdDialog"];
  return $$interimElementProvider('$mdDialog')
    .setDefaults({
      methods: ['disableParentScroll', 'hasBackdrop', 'clickOutsideToClose', 'escapeToClose', 'targetEvent'],
      options: dialogDefaultOptions
    })
    .addPreset('alert', {
      methods: ['title', 'content', 'ariaLabel', 'ok'],
      options: advancedDialogOptions
    })
    .addPreset('confirm', {
      methods: ['title', 'content', 'ariaLabel', 'ok', 'cancel'],
      options: advancedDialogOptions
    });

  /* @ngInject */
  function advancedDialogOptions($mdDialog) {
    return {
      template: [
        '<md-dialog aria-label="{{ dialog.ariaLabel }}">',
          '<md-content>',
            '<h2>{{ dialog.title }}</h2>',
            '<p>{{ dialog.content }}</p>',
          '</md-content>',
          '<div class="md-actions">',
            '<md-button ng-if="dialog.$type == \'confirm\'" ng-click="dialog.abort()">',
              '{{ dialog.cancel }}',
            '</md-button>',
            '<md-button ng-click="dialog.hide()" class="md-primary">',
              '{{ dialog.ok }}',
            '</md-button>',
          '</div>',
        '</md-dialog>'
      ].join(''),
      controller: function mdDialogCtrl() {
        this.hide = function() {
          $mdDialog.hide(true);
        };
        this.abort = function() {
          $mdDialog.cancel();
        };
      },
      controllerAs: 'dialog',
      bindToController: true
    };
  }

  /* @ngInject */
  function dialogDefaultOptions($timeout, $rootElement, $compile, $animate, $mdAria, $document,
                                $mdUtil, $mdConstant, $mdTheming, $$rAF, $q, $mdDialog) {
    return {
      hasBackdrop: true,
      isolateScope: true,
      onShow: onShow,
      onRemove: onRemove,
      clickOutsideToClose: true,
      escapeToClose: true,
      targetEvent: null,
      disableParentScroll: true,
      transformTemplate: function(template) {
        return '<div class="md-dialog-container">' + template + '</div>';
      }
    };


    // On show method for dialogs
    function onShow(scope, element, options) {
      // Incase the user provides a raw dom element, always wrap it in jqLite
      options.parent = angular.element(options.parent);

      options.popInTarget = angular.element((options.targetEvent || {}).target);
      var closeButton = findCloseButton();

      configureAria(element.find('md-dialog'));

      if (options.hasBackdrop) {
        options.backdrop = angular.element('<md-backdrop class="md-dialog-backdrop md-opaque">');
        $mdTheming.inherit(options.backdrop, options.parent);
        $animate.enter(options.backdrop, options.parent);
      }

      if (options.disableParentScroll) {
        options.oldOverflowStyle = options.parent.css('overflow');
        options.parent.css('overflow', 'hidden');
      }

      return dialogPopIn(
        element,
        options.parent,
        options.popInTarget && options.popInTarget.length && options.popInTarget
      )
      .then(function() {
        if (options.escapeToClose) {
          options.rootElementKeyupCallback = function(e) {
            if (e.keyCode === $mdConstant.KEY_CODE.ESCAPE) {
              $timeout($mdDialog.cancel);
            }
          };
          $rootElement.on('keyup', options.rootElementKeyupCallback);
        }

        if (options.clickOutsideToClose) {
          options.dialogClickOutsideCallback = function(e) {
            // Only close if we click the flex container outside the backdrop
            if (e.target === element[0]) {
              $timeout($mdDialog.cancel);
            }
          };
          element.on('click', options.dialogClickOutsideCallback);
        }
        closeButton.focus();
      });


      function findCloseButton() {
        //If no element with class dialog-close, try to find the last
        //button child in md-actions and assume it is a close button
        var closeButton = element[0].querySelector('.dialog-close');
        if (!closeButton) {
          var actionButtons = element[0].querySelectorAll('.md-actions button');
          closeButton = actionButtons[ actionButtons.length - 1 ];
        }
        return angular.element(closeButton);
      }

    }

    // On remove function for all dialogs
    function onRemove(scope, element, options) {

      if (options.backdrop) {
        $animate.leave(options.backdrop);
      }
      if (options.disableParentScroll) {
        options.parent.css('overflow', options.oldOverflowStyle);
        $document[0].removeEventListener('scroll', options.captureScroll, true);
      }
      if (options.escapeToClose) {
        $rootElement.off('keyup', options.rootElementKeyupCallback);
      }
      if (options.clickOutsideToClose) {
        element.off('click', options.dialogClickOutsideCallback);
      }
      return dialogPopOut(
        element,
        options.parent,
        options.popInTarget && options.popInTarget.length && options.popInTarget
      ).then(function() {
        options.scope.$destroy();
        element.remove();
        options.popInTarget && options.popInTarget.focus();
      });

    }

    /**
     * Inject ARIA-specific attributes appropriate for Dialogs
     */
    function configureAria(element) {
      element.attr({
        'role': 'dialog'
      });

      var dialogContent = element.find('md-content');
      if (dialogContent.length === 0){
        dialogContent = element;
      }
      $mdAria.expectAsync(element, 'aria-label', function() {
        var words = dialogContent.text().split(/\s+/);
        if (words.length > 3) words = words.slice(0,3).concat('...');
        return words.join(' ');
      });
    }

    function dialogPopIn(container, parentElement, clickElement) {
      var dialogEl = container.find('md-dialog');

      parentElement.append(container);
      transformToClickElement(dialogEl, clickElement);

      $$rAF(function() {
        dialogEl.addClass('transition-in')
          .css($mdConstant.CSS.TRANSFORM, '');
      });

      return dialogTransitionEnd(dialogEl);
    }

    function dialogPopOut(container, parentElement, clickElement) {
      var dialogEl = container.find('md-dialog');

      dialogEl.addClass('transition-out').removeClass('transition-in');
      transformToClickElement(dialogEl, clickElement);

      return dialogTransitionEnd(dialogEl);
    }

    function transformToClickElement(dialogEl, clickElement) {
      if (clickElement) {
        var clickRect = clickElement[0].getBoundingClientRect();
        var dialogRect = dialogEl[0].getBoundingClientRect();

        var scaleX = Math.min(0.5, clickRect.width / dialogRect.width);
        var scaleY = Math.min(0.5, clickRect.height / dialogRect.height);

        dialogEl.css($mdConstant.CSS.TRANSFORM, 'translate3d(' +
          (-dialogRect.left + clickRect.left + clickRect.width/2 - dialogRect.width/2) + 'px,' +
          (-dialogRect.top + clickRect.top + clickRect.height/2 - dialogRect.height/2) + 'px,' +
          '0) scale(' + scaleX + ',' + scaleY + ')'
        );
      }
    }

    function dialogTransitionEnd(dialogEl) {
      var deferred = $q.defer();
      dialogEl.on($mdConstant.CSS.TRANSITIONEND, finished);
      function finished(ev) {
        //Make sure this transitionend didn't bubble up from a child
        if (ev.target === dialogEl[0]) {
          dialogEl.off($mdConstant.CSS.TRANSITIONEND, finished);
          deferred.resolve();
        }
      }
      return deferred.promise;
    }

  }
}
MdDialogProvider.$inject = ["$$interimElementProvider"];

})();
