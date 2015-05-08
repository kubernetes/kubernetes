/**
 * @license AngularJS v1.3.15
 * (c) 2010-2014 Google, Inc. http://angularjs.org
 * License: MIT
 */
(function(window, angular, undefined) {'use strict';

/* jshint maxlen: false */

/**
 * @ngdoc module
 * @name ngAnimate
 * @description
 *
 * The `ngAnimate` module provides support for JavaScript, CSS3 transition and CSS3 keyframe animation hooks within existing core and custom directives.
 *
 * <div doc-module-components="ngAnimate"></div>
 *
 * # Usage
 *
 * To see animations in action, all that is required is to define the appropriate CSS classes
 * or to register a JavaScript animation via the `myModule.animation()` function. The directives that support animation automatically are:
 * `ngRepeat`, `ngInclude`, `ngIf`, `ngSwitch`, `ngShow`, `ngHide`, `ngView` and `ngClass`. Custom directives can take advantage of animation
 * by using the `$animate` service.
 *
 * Below is a more detailed breakdown of the supported animation events provided by pre-existing ng directives:
 *
 * | Directive                                                                                                | Supported Animations                                                     |
 * |----------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
 * | {@link ng.directive:ngRepeat#animations ngRepeat}                                                        | enter, leave and move                                                    |
 * | {@link ngRoute.directive:ngView#animations ngView}                                                       | enter and leave                                                          |
 * | {@link ng.directive:ngInclude#animations ngInclude}                                                      | enter and leave                                                          |
 * | {@link ng.directive:ngSwitch#animations ngSwitch}                                                        | enter and leave                                                          |
 * | {@link ng.directive:ngIf#animations ngIf}                                                                | enter and leave                                                          |
 * | {@link ng.directive:ngClass#animations ngClass}                                                          | add and remove (the CSS class(es) present)                               |
 * | {@link ng.directive:ngShow#animations ngShow} & {@link ng.directive:ngHide#animations ngHide}            | add and remove (the ng-hide class value)                                 |
 * | {@link ng.directive:form#animation-hooks form} & {@link ng.directive:ngModel#animation-hooks ngModel}    | add and remove (dirty, pristine, valid, invalid & all other validations) |
 * | {@link module:ngMessages#animations ngMessages}                                                          | add and remove (ng-active & ng-inactive)                                 |
 * | {@link module:ngMessages#animations ngMessage}                                                           | enter and leave                                                          |
 *
 * You can find out more information about animations upon visiting each directive page.
 *
 * Below is an example of how to apply animations to a directive that supports animation hooks:
 *
 * ```html
 * <style type="text/css">
 * .slide.ng-enter, .slide.ng-leave {
 *   -webkit-transition:0.5s linear all;
 *   transition:0.5s linear all;
 * }
 *
 * .slide.ng-enter { }        /&#42; starting animations for enter &#42;/
 * .slide.ng-enter.ng-enter-active { } /&#42; terminal animations for enter &#42;/
 * .slide.ng-leave { }        /&#42; starting animations for leave &#42;/
 * .slide.ng-leave.ng-leave-active { } /&#42; terminal animations for leave &#42;/
 * </style>
 *
 * <!--
 * the animate service will automatically add .ng-enter and .ng-leave to the element
 * to trigger the CSS transition/animations
 * -->
 * <ANY class="slide" ng-include="..."></ANY>
 * ```
 *
 * Keep in mind that, by default, if an animation is running, any child elements cannot be animated
 * until the parent element's animation has completed. This blocking feature can be overridden by
 * placing the `ng-animate-children` attribute on a parent container tag.
 *
 * ```html
 * <div class="slide-animation" ng-if="on" ng-animate-children>
 *   <div class="fade-animation" ng-if="on">
 *     <div class="explode-animation" ng-if="on">
 *        ...
 *     </div>
 *   </div>
 * </div>
 * ```
 *
 * When the `on` expression value changes and an animation is triggered then each of the elements within
 * will all animate without the block being applied to child elements.
 *
 * ## Are animations run when the application starts?
 * No they are not. When an application is bootstrapped Angular will disable animations from running to avoid
 * a frenzy of animations from being triggered as soon as the browser has rendered the screen. For this to work,
 * Angular will wait for two digest cycles until enabling animations. From there on, any animation-triggering
 * layout changes in the application will trigger animations as normal.
 *
 * In addition, upon bootstrap, if the routing system or any directives or load remote data (via $http) then Angular
 * will automatically extend the wait time to enable animations once **all** of the outbound HTTP requests
 * are complete.
 *
 * ## CSS-defined Animations
 * The animate service will automatically apply two CSS classes to the animated element and these two CSS classes
 * are designed to contain the start and end CSS styling. Both CSS transitions and keyframe animations are supported
 * and can be used to play along with this naming structure.
 *
 * The following code below demonstrates how to perform animations using **CSS transitions** with Angular:
 *
 * ```html
 * <style type="text/css">
 * /&#42;
 *  The animate class is apart of the element and the ng-enter class
 *  is attached to the element once the enter animation event is triggered
 * &#42;/
 * .reveal-animation.ng-enter {
 *  -webkit-transition: 1s linear all; /&#42; Safari/Chrome &#42;/
 *  transition: 1s linear all; /&#42; All other modern browsers and IE10+ &#42;/
 *
 *  /&#42; The animation preparation code &#42;/
 *  opacity: 0;
 * }
 *
 * /&#42;
 *  Keep in mind that you want to combine both CSS
 *  classes together to avoid any CSS-specificity
 *  conflicts
 * &#42;/
 * .reveal-animation.ng-enter.ng-enter-active {
 *  /&#42; The animation code itself &#42;/
 *  opacity: 1;
 * }
 * </style>
 *
 * <div class="view-container">
 *   <div ng-view class="reveal-animation"></div>
 * </div>
 * ```
 *
 * The following code below demonstrates how to perform animations using **CSS animations** with Angular:
 *
 * ```html
 * <style type="text/css">
 * .reveal-animation.ng-enter {
 *   -webkit-animation: enter_sequence 1s linear; /&#42; Safari/Chrome &#42;/
 *   animation: enter_sequence 1s linear; /&#42; IE10+ and Future Browsers &#42;/
 * }
 * @-webkit-keyframes enter_sequence {
 *   from { opacity:0; }
 *   to { opacity:1; }
 * }
 * @keyframes enter_sequence {
 *   from { opacity:0; }
 *   to { opacity:1; }
 * }
 * </style>
 *
 * <div class="view-container">
 *   <div ng-view class="reveal-animation"></div>
 * </div>
 * ```
 *
 * Both CSS3 animations and transitions can be used together and the animate service will figure out the correct duration and delay timing.
 *
 * Upon DOM mutation, the event class is added first (something like `ng-enter`), then the browser prepares itself to add
 * the active class (in this case `ng-enter-active`) which then triggers the animation. The animation module will automatically
 * detect the CSS code to determine when the animation ends. Once the animation is over then both CSS classes will be
 * removed from the DOM. If a browser does not support CSS transitions or CSS animations then the animation will start and end
 * immediately resulting in a DOM element that is at its final state. This final state is when the DOM element
 * has no CSS transition/animation classes applied to it.
 *
 * ### Structural transition animations
 *
 * Structural transitions (such as enter, leave and move) will always apply a `0s none` transition
 * value to force the browser into rendering the styles defined in the setup (`.ng-enter`, `.ng-leave`
 * or `.ng-move`) class. This means that any active transition animations operating on the element
 * will be cut off to make way for the enter, leave or move animation.
 *
 * ### Class-based transition animations
 *
 * Class-based transitions refer to transition animations that are triggered when a CSS class is
 * added to or removed from the element (via `$animate.addClass`, `$animate.removeClass`,
 * `$animate.setClass`, or by directives such as `ngClass`, `ngModel` and `form`).
 * They are different when compared to structural animations since they **do not cancel existing
 * animations** nor do they **block successive transitions** from rendering on the same element.
 * This distinction allows for **multiple class-based transitions** to be performed on the same element.
 *
 * In addition to ngAnimate supporting the default (natural) functionality of class-based transition
 * animations, ngAnimate also decorates the element with starting and ending CSS classes to aid the
 * developer in further styling the element throughout the transition animation. Earlier versions
 * of ngAnimate may have caused natural CSS transitions to break and not render properly due to
 * $animate temporarily blocking transitions using `0s none` in order to allow the setup CSS class
 * (the `-add` or `-remove` class) to be applied without triggering an animation. However, as of
 * **version 1.3**, this workaround has been removed with ngAnimate and all non-ngAnimate CSS
 * class transitions are compatible with ngAnimate.
 *
 * There is, however, one special case when dealing with class-based transitions in ngAnimate.
 * When rendering class-based transitions that make use of the setup and active CSS classes
 * (e.g. `.fade-add` and `.fade-add-active` for when `.fade` is added) be sure to define
 * the transition value **on the active CSS class** and not the setup class.
 *
 * ```css
 * .fade-add {
 *   /&#42; remember to place a 0s transition here
 *      to ensure that the styles are applied instantly
 *      even if the element already has a transition style &#42;/
 *   transition:0s linear all;
 *
 *   /&#42; starting CSS styles &#42;/
 *   opacity:1;
 * }
 * .fade-add.fade-add-active {
 *   /&#42; this will be the length of the animation &#42;/
 *   transition:1s linear all;
 *   opacity:0;
 * }
 * ```
 *
 * The setup CSS class (in this case `.fade-add`) also has a transition style property, however, it
 * has a duration of zero. This may not be required, however, incase the browser is unable to render
 * the styling present in this CSS class instantly then it could be that the browser is attempting
 * to perform an unnecessary transition.
 *
 * This workaround, however, does not apply to  standard class-based transitions that are rendered
 * when a CSS class containing a transition is applied to an element:
 *
 * ```css
 * /&#42; this works as expected &#42;/
 * .fade {
 *   transition:1s linear all;
 *   opacity:0;
 * }
 * ```
 *
 * Please keep this in mind when coding the CSS markup that will be used within class-based transitions.
 * Also, try not to mix the two class-based animation flavors together since the CSS code may become
 * overly complex.
 *
 *
 * ### Preventing Collisions With Third Party Libraries
 *
 * Some third-party frameworks place animation duration defaults across many element or className
 * selectors in order to make their code small and reuseable. This can lead to issues with ngAnimate, which
 * is expecting actual animations on these elements and has to wait for their completion.
 *
 * You can prevent this unwanted behavior by using a prefix on all your animation classes:
 *
 * ```css
 * /&#42; prefixed with animate- &#42;/
 * .animate-fade-add.animate-fade-add-active {
 *   transition:1s linear all;
 *   opacity:0;
 * }
 * ```
 *
 * You then configure `$animate` to enforce this prefix:
 *
 * ```js
 * $animateProvider.classNameFilter(/animate-/);
 * ```
 * </div>
 *
 * ### CSS Staggering Animations
 * A Staggering animation is a collection of animations that are issued with a slight delay in between each successive operation resulting in a
 * curtain-like effect. The ngAnimate module (versions >=1.2) supports staggering animations and the stagger effect can be
 * performed by creating a **ng-EVENT-stagger** CSS class and attaching that class to the base CSS class used for
 * the animation. The style property expected within the stagger class can either be a **transition-delay** or an
 * **animation-delay** property (or both if your animation contains both transitions and keyframe animations).
 *
 * ```css
 * .my-animation.ng-enter {
 *   /&#42; standard transition code &#42;/
 *   -webkit-transition: 1s linear all;
 *   transition: 1s linear all;
 *   opacity:0;
 * }
 * .my-animation.ng-enter-stagger {
 *   /&#42; this will have a 100ms delay between each successive leave animation &#42;/
 *   -webkit-transition-delay: 0.1s;
 *   transition-delay: 0.1s;
 *
 *   /&#42; in case the stagger doesn't work then these two values
 *    must be set to 0 to avoid an accidental CSS inheritance &#42;/
 *   -webkit-transition-duration: 0s;
 *   transition-duration: 0s;
 * }
 * .my-animation.ng-enter.ng-enter-active {
 *   /&#42; standard transition styles &#42;/
 *   opacity:1;
 * }
 * ```
 *
 * Staggering animations work by default in ngRepeat (so long as the CSS class is defined). Outside of ngRepeat, to use staggering animations
 * on your own, they can be triggered by firing multiple calls to the same event on $animate. However, the restrictions surrounding this
 * are that each of the elements must have the same CSS className value as well as the same parent element. A stagger operation
 * will also be reset if more than 10ms has passed after the last animation has been fired.
 *
 * The following code will issue the **ng-leave-stagger** event on the element provided:
 *
 * ```js
 * var kids = parent.children();
 *
 * $animate.leave(kids[0]); //stagger index=0
 * $animate.leave(kids[1]); //stagger index=1
 * $animate.leave(kids[2]); //stagger index=2
 * $animate.leave(kids[3]); //stagger index=3
 * $animate.leave(kids[4]); //stagger index=4
 *
 * $timeout(function() {
 *   //stagger has reset itself
 *   $animate.leave(kids[5]); //stagger index=0
 *   $animate.leave(kids[6]); //stagger index=1
 * }, 100, false);
 * ```
 *
 * Stagger animations are currently only supported within CSS-defined animations.
 *
 * ## JavaScript-defined Animations
 * In the event that you do not want to use CSS3 transitions or CSS3 animations or if you wish to offer animations on browsers that do not
 * yet support CSS transitions/animations, then you can make use of JavaScript animations defined inside of your AngularJS module.
 *
 * ```js
 * //!annotate="YourApp" Your AngularJS Module|Replace this or ngModule with the module that you used to define your application.
 * var ngModule = angular.module('YourApp', ['ngAnimate']);
 * ngModule.animation('.my-crazy-animation', function() {
 *   return {
 *     enter: function(element, done) {
 *       //run the animation here and call done when the animation is complete
 *       return function(cancelled) {
 *         //this (optional) function will be called when the animation
 *         //completes or when the animation is cancelled (the cancelled
 *         //flag will be set to true if cancelled).
 *       };
 *     },
 *     leave: function(element, done) { },
 *     move: function(element, done) { },
 *
 *     //animation that can be triggered before the class is added
 *     beforeAddClass: function(element, className, done) { },
 *
 *     //animation that can be triggered after the class is added
 *     addClass: function(element, className, done) { },
 *
 *     //animation that can be triggered before the class is removed
 *     beforeRemoveClass: function(element, className, done) { },
 *
 *     //animation that can be triggered after the class is removed
 *     removeClass: function(element, className, done) { }
 *   };
 * });
 * ```
 *
 * JavaScript-defined animations are created with a CSS-like class selector and a collection of events which are set to run
 * a javascript callback function. When an animation is triggered, $animate will look for a matching animation which fits
 * the element's CSS class attribute value and then run the matching animation event function (if found).
 * In other words, if the CSS classes present on the animated element match any of the JavaScript animations then the callback function will
 * be executed. It should be also noted that only simple, single class selectors are allowed (compound class selectors are not supported).
 *
 * Within a JavaScript animation, an object containing various event callback animation functions is expected to be returned.
 * As explained above, these callbacks are triggered based on the animation event. Therefore if an enter animation is run,
 * and the JavaScript animation is found, then the enter callback will handle that animation (in addition to the CSS keyframe animation
 * or transition code that is defined via a stylesheet).
 *
 *
 * ### Applying Directive-specific Styles to an Animation
 * In some cases a directive or service may want to provide `$animate` with extra details that the animation will
 * include into its animation. Let's say for example we wanted to render an animation that animates an element
 * towards the mouse coordinates as to where the user clicked last. By collecting the X/Y coordinates of the click
 * (via the event parameter) we can set the `top` and `left` styles into an object and pass that into our function
 * call to `$animate.addClass`.
 *
 * ```js
 * canvas.on('click', function(e) {
 *   $animate.addClass(element, 'on', {
 *     to: {
 *       left : e.client.x + 'px',
 *       top : e.client.y + 'px'
 *     }
 *   }):
 * });
 * ```
 *
 * Now when the animation runs, and a transition or keyframe animation is picked up, then the animation itself will
 * also include and transition the styling of the `left` and `top` properties into its running animation. If we want
 * to provide some starting animation values then we can do so by placing the starting animations styles into an object
 * called `from` in the same object as the `to` animations.
 *
 * ```js
 * canvas.on('click', function(e) {
 *   $animate.addClass(element, 'on', {
 *     from: {
 *        position: 'absolute',
 *        left: '0px',
 *        top: '0px'
 *     },
 *     to: {
 *       left : e.client.x + 'px',
 *       top : e.client.y + 'px'
 *     }
 *   }):
 * });
 * ```
 *
 * Once the animation is complete or cancelled then the union of both the before and after styles are applied to the
 * element. If `ngAnimate` is not present then the styles will be applied immediately.
 *
 */

angular.module('ngAnimate', ['ng'])

  /**
   * @ngdoc provider
   * @name $animateProvider
   * @description
   *
   * The `$animateProvider` allows developers to register JavaScript animation event handlers directly inside of a module.
   * When an animation is triggered, the $animate service will query the $animate service to find any animations that match
   * the provided name value.
   *
   * Requires the {@link ngAnimate `ngAnimate`} module to be installed.
   *
   * Please visit the {@link ngAnimate `ngAnimate`} module overview page learn more about how to use animations in your application.
   *
   */
  .directive('ngAnimateChildren', function() {
    var NG_ANIMATE_CHILDREN = '$$ngAnimateChildren';
    return function(scope, element, attrs) {
      var val = attrs.ngAnimateChildren;
      if (angular.isString(val) && val.length === 0) { //empty attribute
        element.data(NG_ANIMATE_CHILDREN, true);
      } else {
        scope.$watch(val, function(value) {
          element.data(NG_ANIMATE_CHILDREN, !!value);
        });
      }
    };
  })

  //this private service is only used within CSS-enabled animations
  //IE8 + IE9 do not support rAF natively, but that is fine since they
  //also don't support transitions and keyframes which means that the code
  //below will never be used by the two browsers.
  .factory('$$animateReflow', ['$$rAF', '$document', function($$rAF, $document) {
    var bod = $document[0].body;
    return function(fn) {
      //the returned function acts as the cancellation function
      return $$rAF(function() {
        //the line below will force the browser to perform a repaint
        //so that all the animated elements within the animation frame
        //will be properly updated and drawn on screen. This is
        //required to perform multi-class CSS based animations with
        //Firefox. DO NOT REMOVE THIS LINE.
        var a = bod.offsetWidth + 1;
        fn();
      });
    };
  }])

  .config(['$provide', '$animateProvider', function($provide, $animateProvider) {
    var noop = angular.noop;
    var forEach = angular.forEach;
    var selectors = $animateProvider.$$selectors;
    var isArray = angular.isArray;
    var isString = angular.isString;
    var isObject = angular.isObject;

    var ELEMENT_NODE = 1;
    var NG_ANIMATE_STATE = '$$ngAnimateState';
    var NG_ANIMATE_CHILDREN = '$$ngAnimateChildren';
    var NG_ANIMATE_CLASS_NAME = 'ng-animate';
    var rootAnimateState = {running: true};

    function extractElementNode(element) {
      for (var i = 0; i < element.length; i++) {
        var elm = element[i];
        if (elm.nodeType == ELEMENT_NODE) {
          return elm;
        }
      }
    }

    function prepareElement(element) {
      return element && angular.element(element);
    }

    function stripCommentsFromElement(element) {
      return angular.element(extractElementNode(element));
    }

    function isMatchingElement(elm1, elm2) {
      return extractElementNode(elm1) == extractElementNode(elm2);
    }
    var $$jqLite;
    $provide.decorator('$animate',
        ['$delegate', '$$q', '$injector', '$sniffer', '$rootElement', '$$asyncCallback', '$rootScope', '$document', '$templateRequest', '$$jqLite',
 function($delegate,   $$q,   $injector,   $sniffer,   $rootElement,   $$asyncCallback,   $rootScope,   $document,   $templateRequest,   $$$jqLite) {

      $$jqLite = $$$jqLite;
      $rootElement.data(NG_ANIMATE_STATE, rootAnimateState);

      // Wait until all directive and route-related templates are downloaded and
      // compiled. The $templateRequest.totalPendingRequests variable keeps track of
      // all of the remote templates being currently downloaded. If there are no
      // templates currently downloading then the watcher will still fire anyway.
      var deregisterWatch = $rootScope.$watch(
        function() { return $templateRequest.totalPendingRequests; },
        function(val, oldVal) {
          if (val !== 0) return;
          deregisterWatch();

          // Now that all templates have been downloaded, $animate will wait until
          // the post digest queue is empty before enabling animations. By having two
          // calls to $postDigest calls we can ensure that the flag is enabled at the
          // very end of the post digest queue. Since all of the animations in $animate
          // use $postDigest, it's important that the code below executes at the end.
          // This basically means that the page is fully downloaded and compiled before
          // any animations are triggered.
          $rootScope.$$postDigest(function() {
            $rootScope.$$postDigest(function() {
              rootAnimateState.running = false;
            });
          });
        }
      );

      var globalAnimationCounter = 0;
      var classNameFilter = $animateProvider.classNameFilter();
      var isAnimatableClassName = !classNameFilter
              ? function() { return true; }
              : function(className) {
                return classNameFilter.test(className);
              };

      function classBasedAnimationsBlocked(element, setter) {
        var data = element.data(NG_ANIMATE_STATE) || {};
        if (setter) {
          data.running = true;
          data.structural = true;
          element.data(NG_ANIMATE_STATE, data);
        }
        return data.disabled || (data.running && data.structural);
      }

      function runAnimationPostDigest(fn) {
        var cancelFn, defer = $$q.defer();
        defer.promise.$$cancelFn = function() {
          cancelFn && cancelFn();
        };
        $rootScope.$$postDigest(function() {
          cancelFn = fn(function() {
            defer.resolve();
          });
        });
        return defer.promise;
      }

      function parseAnimateOptions(options) {
        // some plugin code may still be passing in the callback
        // function as the last param for the $animate methods so
        // it's best to only allow string or array values for now
        if (isObject(options)) {
          if (options.tempClasses && isString(options.tempClasses)) {
            options.tempClasses = options.tempClasses.split(/\s+/);
          }
          return options;
        }
      }

      function resolveElementClasses(element, cache, runningAnimations) {
        runningAnimations = runningAnimations || {};

        var lookup = {};
        forEach(runningAnimations, function(data, selector) {
          forEach(selector.split(' '), function(s) {
            lookup[s]=data;
          });
        });

        var hasClasses = Object.create(null);
        forEach((element.attr('class') || '').split(/\s+/), function(className) {
          hasClasses[className] = true;
        });

        var toAdd = [], toRemove = [];
        forEach((cache && cache.classes) || [], function(status, className) {
          var hasClass = hasClasses[className];
          var matchingAnimation = lookup[className] || {};

          // When addClass and removeClass is called then $animate will check to
          // see if addClass and removeClass cancel each other out. When there are
          // more calls to removeClass than addClass then the count falls below 0
          // and then the removeClass animation will be allowed. Otherwise if the
          // count is above 0 then that means an addClass animation will commence.
          // Once an animation is allowed then the code will also check to see if
          // there exists any on-going animation that is already adding or remvoing
          // the matching CSS class.
          if (status === false) {
            //does it have the class or will it have the class
            if (hasClass || matchingAnimation.event == 'addClass') {
              toRemove.push(className);
            }
          } else if (status === true) {
            //is the class missing or will it be removed?
            if (!hasClass || matchingAnimation.event == 'removeClass') {
              toAdd.push(className);
            }
          }
        });

        return (toAdd.length + toRemove.length) > 0 && [toAdd.join(' '), toRemove.join(' ')];
      }

      function lookup(name) {
        if (name) {
          var matches = [],
              flagMap = {},
              classes = name.substr(1).split('.');

          //the empty string value is the default animation
          //operation which performs CSS transition and keyframe
          //animations sniffing. This is always included for each
          //element animation procedure if the browser supports
          //transitions and/or keyframe animations. The default
          //animation is added to the top of the list to prevent
          //any previous animations from affecting the element styling
          //prior to the element being animated.
          if ($sniffer.transitions || $sniffer.animations) {
            matches.push($injector.get(selectors['']));
          }

          for (var i=0; i < classes.length; i++) {
            var klass = classes[i],
                selectorFactoryName = selectors[klass];
            if (selectorFactoryName && !flagMap[klass]) {
              matches.push($injector.get(selectorFactoryName));
              flagMap[klass] = true;
            }
          }
          return matches;
        }
      }

      function animationRunner(element, animationEvent, className, options) {
        //transcluded directives may sometimes fire an animation using only comment nodes
        //best to catch this early on to prevent any animation operations from occurring
        var node = element[0];
        if (!node) {
          return;
        }

        if (options) {
          options.to = options.to || {};
          options.from = options.from || {};
        }

        var classNameAdd;
        var classNameRemove;
        if (isArray(className)) {
          classNameAdd = className[0];
          classNameRemove = className[1];
          if (!classNameAdd) {
            className = classNameRemove;
            animationEvent = 'removeClass';
          } else if (!classNameRemove) {
            className = classNameAdd;
            animationEvent = 'addClass';
          } else {
            className = classNameAdd + ' ' + classNameRemove;
          }
        }

        var isSetClassOperation = animationEvent == 'setClass';
        var isClassBased = isSetClassOperation
                           || animationEvent == 'addClass'
                           || animationEvent == 'removeClass'
                           || animationEvent == 'animate';

        var currentClassName = element.attr('class');
        var classes = currentClassName + ' ' + className;
        if (!isAnimatableClassName(classes)) {
          return;
        }

        var beforeComplete = noop,
            beforeCancel = [],
            before = [],
            afterComplete = noop,
            afterCancel = [],
            after = [];

        var animationLookup = (' ' + classes).replace(/\s+/g,'.');
        forEach(lookup(animationLookup), function(animationFactory) {
          var created = registerAnimation(animationFactory, animationEvent);
          if (!created && isSetClassOperation) {
            registerAnimation(animationFactory, 'addClass');
            registerAnimation(animationFactory, 'removeClass');
          }
        });

        function registerAnimation(animationFactory, event) {
          var afterFn = animationFactory[event];
          var beforeFn = animationFactory['before' + event.charAt(0).toUpperCase() + event.substr(1)];
          if (afterFn || beforeFn) {
            if (event == 'leave') {
              beforeFn = afterFn;
              //when set as null then animation knows to skip this phase
              afterFn = null;
            }
            after.push({
              event: event, fn: afterFn
            });
            before.push({
              event: event, fn: beforeFn
            });
            return true;
          }
        }

        function run(fns, cancellations, allCompleteFn) {
          var animations = [];
          forEach(fns, function(animation) {
            animation.fn && animations.push(animation);
          });

          var count = 0;
          function afterAnimationComplete(index) {
            if (cancellations) {
              (cancellations[index] || noop)();
              if (++count < animations.length) return;
              cancellations = null;
            }
            allCompleteFn();
          }

          //The code below adds directly to the array in order to work with
          //both sync and async animations. Sync animations are when the done()
          //operation is called right away. DO NOT REFACTOR!
          forEach(animations, function(animation, index) {
            var progress = function() {
              afterAnimationComplete(index);
            };
            switch (animation.event) {
              case 'setClass':
                cancellations.push(animation.fn(element, classNameAdd, classNameRemove, progress, options));
                break;
              case 'animate':
                cancellations.push(animation.fn(element, className, options.from, options.to, progress));
                break;
              case 'addClass':
                cancellations.push(animation.fn(element, classNameAdd || className,     progress, options));
                break;
              case 'removeClass':
                cancellations.push(animation.fn(element, classNameRemove || className,  progress, options));
                break;
              default:
                cancellations.push(animation.fn(element, progress, options));
                break;
            }
          });

          if (cancellations && cancellations.length === 0) {
            allCompleteFn();
          }
        }

        return {
          node: node,
          event: animationEvent,
          className: className,
          isClassBased: isClassBased,
          isSetClassOperation: isSetClassOperation,
          applyStyles: function() {
            if (options) {
              element.css(angular.extend(options.from || {}, options.to || {}));
            }
          },
          before: function(allCompleteFn) {
            beforeComplete = allCompleteFn;
            run(before, beforeCancel, function() {
              beforeComplete = noop;
              allCompleteFn();
            });
          },
          after: function(allCompleteFn) {
            afterComplete = allCompleteFn;
            run(after, afterCancel, function() {
              afterComplete = noop;
              allCompleteFn();
            });
          },
          cancel: function() {
            if (beforeCancel) {
              forEach(beforeCancel, function(cancelFn) {
                (cancelFn || noop)(true);
              });
              beforeComplete(true);
            }
            if (afterCancel) {
              forEach(afterCancel, function(cancelFn) {
                (cancelFn || noop)(true);
              });
              afterComplete(true);
            }
          }
        };
      }

      /**
       * @ngdoc service
       * @name $animate
       * @kind object
       *
       * @description
       * The `$animate` service provides animation detection support while performing DOM operations (enter, leave and move) as well as during addClass and removeClass operations.
       * When any of these operations are run, the $animate service
       * will examine any JavaScript-defined animations (which are defined by using the $animateProvider provider object)
       * as well as any CSS-defined animations against the CSS classes present on the element once the DOM operation is run.
       *
       * The `$animate` service is used behind the scenes with pre-existing directives and animation with these directives
       * will work out of the box without any extra configuration.
       *
       * Requires the {@link ngAnimate `ngAnimate`} module to be installed.
       *
       * Please visit the {@link ngAnimate `ngAnimate`} module overview page learn more about how to use animations in your application.
       * ## Callback Promises
       * With AngularJS 1.3, each of the animation methods, on the `$animate` service, return a promise when called. The
       * promise itself is then resolved once the animation has completed itself, has been cancelled or has been
       * skipped due to animations being disabled. (Note that even if the animation is cancelled it will still
       * call the resolve function of the animation.)
       *
       * ```js
       * $animate.enter(element, container).then(function() {
       *   //...this is called once the animation is complete...
       * });
       * ```
       *
       * Also note that, due to the nature of the callback promise, if any Angular-specific code (like changing the scope,
       * location of the page, etc...) is executed within the callback promise then be sure to wrap the code using
       * `$scope.$apply(...)`;
       *
       * ```js
       * $animate.leave(element).then(function() {
       *   $scope.$apply(function() {
       *     $location.path('/new-page');
       *   });
       * });
       * ```
       *
       * An animation can also be cancelled by calling the `$animate.cancel(promise)` method with the provided
       * promise that was returned when the animation was started.
       *
       * ```js
       * var promise = $animate.addClass(element, 'super-long-animation');
       * promise.then(function() {
       *   //this will still be called even if cancelled
       * });
       *
       * element.on('click', function() {
       *   //tooo lazy to wait for the animation to end
       *   $animate.cancel(promise);
       * });
       * ```
       *
       * (Keep in mind that the promise cancellation is unique to `$animate` since promises in
       * general cannot be cancelled.)
       *
       */
      return {
        /**
         * @ngdoc method
         * @name $animate#animate
         * @kind function
         *
         * @description
         * Performs an inline animation on the element which applies the provided `to` and `from` CSS styles to the element.
         * If any detected CSS transition, keyframe or JavaScript matches the provided `className` value then the animation
         * will take on the provided styles. For example, if a transition animation is set for the given className then the
         * provided `from` and `to` styles will be applied alongside the given transition. If a JavaScript animation is
         * detected then the provided styles will be given in as function paramters.
         *
         * ```js
         * ngModule.animation('.my-inline-animation', function() {
         *   return {
         *     animate : function(element, className, from, to, done) {
         *       //styles
         *     }
         *   }
         * });
         * ```
         *
         * Below is a breakdown of each step that occurs during the `animate` animation:
         *
         * | Animation Step                                                                                                        | What the element class attribute looks like                  |
         * |-----------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
         * | 1. `$animate.animate(...)` is called                                                                                  | `class="my-animation"`                                       |
         * | 2. `$animate` waits for the next digest to start the animation                                                        | `class="my-animation ng-animate"`                            |
         * | 3. `$animate` runs the JavaScript-defined animations detected on the element                                          | `class="my-animation ng-animate"`                            |
         * | 4. the `className` class value is added to the element                                                                | `class="my-animation ng-animate className"`                  |
         * | 5. `$animate` scans the element styles to get the CSS transition/animation duration and delay                         | `class="my-animation ng-animate className"`                  |
         * | 6. `$animate` blocks all CSS transitions on the element to ensure the `.className` class styling is applied right away| `class="my-animation ng-animate className"`                  |
         * | 7. `$animate` applies the provided collection of `from` CSS styles to the element                                     | `class="my-animation ng-animate className"`                  |
         * | 8. `$animate` waits for a single animation frame (this performs a reflow)                                             | `class="my-animation ng-animate className"`                  |
         * | 9. `$animate` removes the CSS transition block placed on the element                                                  | `class="my-animation ng-animate className"`                  |
         * | 10. the `className-active` class is added (this triggers the CSS transition/animation)                                | `class="my-animation ng-animate className className-active"` |
         * | 11. `$animate` applies the collection of `to` CSS styles to the element which are then handled by the transition      | `class="my-animation ng-animate className className-active"` |
         * | 12. `$animate` waits for the animation to complete (via events and timeout)                                           | `class="my-animation ng-animate className className-active"` |
         * | 13. The animation ends and all generated CSS classes are removed from the element                                     | `class="my-animation"`                                       |
         * | 14. The returned promise is resolved.                                                                                 | `class="my-animation"`                                       |
         *
         * @param {DOMElement} element the element that will be the focus of the enter animation
         * @param {object} from a collection of CSS styles that will be applied to the element at the start of the animation
         * @param {object} to a collection of CSS styles that the element will animate towards
         * @param {string=} className an optional CSS class that will be added to the element for the duration of the animation (the default class is `ng-inline-animate`)
         * @param {object=} options an optional collection of options that will be picked up by the CSS transition/animation
         * @return {Promise} the animation callback promise
        */
        animate: function(element, from, to, className, options) {
          className = className || 'ng-inline-animate';
          options = parseAnimateOptions(options) || {};
          options.from = to ? from : null;
          options.to   = to ? to : from;

          return runAnimationPostDigest(function(done) {
            return performAnimation('animate', className, stripCommentsFromElement(element), null, null, noop, options, done);
          });
        },

        /**
         * @ngdoc method
         * @name $animate#enter
         * @kind function
         *
         * @description
         * Appends the element to the parentElement element that resides in the document and then runs the enter animation. Once
         * the animation is started, the following CSS classes will be present on the element for the duration of the animation:
         *
         * Below is a breakdown of each step that occurs during enter animation:
         *
         * | Animation Step                                                                                                        | What the element class attribute looks like                |
         * |-----------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
         * | 1. `$animate.enter(...)` is called                                                                                    | `class="my-animation"`                                     |
         * | 2. element is inserted into the `parentElement` element or beside the `afterElement` element                          | `class="my-animation"`                                     |
         * | 3. `$animate` waits for the next digest to start the animation                                                        | `class="my-animation ng-animate"`                          |
         * | 4. `$animate` runs the JavaScript-defined animations detected on the element                                          | `class="my-animation ng-animate"`                          |
         * | 5. the `.ng-enter` class is added to the element                                                                      | `class="my-animation ng-animate ng-enter"`                 |
         * | 6. `$animate` scans the element styles to get the CSS transition/animation duration and delay                         | `class="my-animation ng-animate ng-enter"`                 |
         * | 7. `$animate` blocks all CSS transitions on the element to ensure the `.ng-enter` class styling is applied right away | `class="my-animation ng-animate ng-enter"`                 |
         * | 8. `$animate` waits for a single animation frame (this performs a reflow)                                             | `class="my-animation ng-animate ng-enter"`                 |
         * | 9. `$animate` removes the CSS transition block placed on the element                                                  | `class="my-animation ng-animate ng-enter"`                 |
         * | 10. the `.ng-enter-active` class is added (this triggers the CSS transition/animation)                                | `class="my-animation ng-animate ng-enter ng-enter-active"` |
         * | 11. `$animate` waits for the animation to complete (via events and timeout)                                           | `class="my-animation ng-animate ng-enter ng-enter-active"` |
         * | 12. The animation ends and all generated CSS classes are removed from the element                                     | `class="my-animation"`                                     |
         * | 13. The returned promise is resolved.                                                                                 | `class="my-animation"`                                     |
         *
         * @param {DOMElement} element the element that will be the focus of the enter animation
         * @param {DOMElement} parentElement the parent element of the element that will be the focus of the enter animation
         * @param {DOMElement} afterElement the sibling element (which is the previous element) of the element that will be the focus of the enter animation
         * @param {object=} options an optional collection of options that will be picked up by the CSS transition/animation
         * @return {Promise} the animation callback promise
        */
        enter: function(element, parentElement, afterElement, options) {
          options = parseAnimateOptions(options);
          element = angular.element(element);
          parentElement = prepareElement(parentElement);
          afterElement = prepareElement(afterElement);

          classBasedAnimationsBlocked(element, true);
          $delegate.enter(element, parentElement, afterElement);
          return runAnimationPostDigest(function(done) {
            return performAnimation('enter', 'ng-enter', stripCommentsFromElement(element), parentElement, afterElement, noop, options, done);
          });
        },

        /**
         * @ngdoc method
         * @name $animate#leave
         * @kind function
         *
         * @description
         * Runs the leave animation operation and, upon completion, removes the element from the DOM. Once
         * the animation is started, the following CSS classes will be added for the duration of the animation:
         *
         * Below is a breakdown of each step that occurs during leave animation:
         *
         * | Animation Step                                                                                                        | What the element class attribute looks like                |
         * |-----------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
         * | 1. `$animate.leave(...)` is called                                                                                    | `class="my-animation"`                                     |
         * | 2. `$animate` runs the JavaScript-defined animations detected on the element                                          | `class="my-animation ng-animate"`                          |
         * | 3. `$animate` waits for the next digest to start the animation                                                        | `class="my-animation ng-animate"`                          |
         * | 4. the `.ng-leave` class is added to the element                                                                      | `class="my-animation ng-animate ng-leave"`                 |
         * | 5. `$animate` scans the element styles to get the CSS transition/animation duration and delay                         | `class="my-animation ng-animate ng-leave"`                 |
         * | 6. `$animate` blocks all CSS transitions on the element to ensure the `.ng-leave` class styling is applied right away | `class="my-animation ng-animate ng-leave"`                 |
         * | 7. `$animate` waits for a single animation frame (this performs a reflow)                                             | `class="my-animation ng-animate ng-leave"`                 |
         * | 8. `$animate` removes the CSS transition block placed on the element                                                  | `class="my-animation ng-animate ng-leave"`                 |
         * | 9. the `.ng-leave-active` class is added (this triggers the CSS transition/animation)                                 | `class="my-animation ng-animate ng-leave ng-leave-active"` |
         * | 10. `$animate` waits for the animation to complete (via events and timeout)                                           | `class="my-animation ng-animate ng-leave ng-leave-active"` |
         * | 11. The animation ends and all generated CSS classes are removed from the element                                     | `class="my-animation"`                                     |
         * | 12. The element is removed from the DOM                                                                               | ...                                                        |
         * | 13. The returned promise is resolved.                                                                                 | ...                                                        |
         *
         * @param {DOMElement} element the element that will be the focus of the leave animation
         * @param {object=} options an optional collection of styles that will be picked up by the CSS transition/animation
         * @return {Promise} the animation callback promise
        */
        leave: function(element, options) {
          options = parseAnimateOptions(options);
          element = angular.element(element);

          cancelChildAnimations(element);
          classBasedAnimationsBlocked(element, true);
          return runAnimationPostDigest(function(done) {
            return performAnimation('leave', 'ng-leave', stripCommentsFromElement(element), null, null, function() {
              $delegate.leave(element);
            }, options, done);
          });
        },

        /**
         * @ngdoc method
         * @name $animate#move
         * @kind function
         *
         * @description
         * Fires the move DOM operation. Just before the animation starts, the animate service will either append it into the parentElement container or
         * add the element directly after the afterElement element if present. Then the move animation will be run. Once
         * the animation is started, the following CSS classes will be added for the duration of the animation:
         *
         * Below is a breakdown of each step that occurs during move animation:
         *
         * | Animation Step                                                                                                       | What the element class attribute looks like              |
         * |----------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
         * | 1. `$animate.move(...)` is called                                                                                    | `class="my-animation"`                                   |
         * | 2. element is moved into the parentElement element or beside the afterElement element                                | `class="my-animation"`                                   |
         * | 3. `$animate` waits for the next digest to start the animation                                                       | `class="my-animation ng-animate"`                        |
         * | 4. `$animate` runs the JavaScript-defined animations detected on the element                                         | `class="my-animation ng-animate"`                        |
         * | 5. the `.ng-move` class is added to the element                                                                      | `class="my-animation ng-animate ng-move"`                |
         * | 6. `$animate` scans the element styles to get the CSS transition/animation duration and delay                        | `class="my-animation ng-animate ng-move"`                |
         * | 7. `$animate` blocks all CSS transitions on the element to ensure the `.ng-move` class styling is applied right away | `class="my-animation ng-animate ng-move"`                |
         * | 8. `$animate` waits for a single animation frame (this performs a reflow)                                            | `class="my-animation ng-animate ng-move"`                |
         * | 9. `$animate` removes the CSS transition block placed on the element                                                 | `class="my-animation ng-animate ng-move"`                |
         * | 10. the `.ng-move-active` class is added (this triggers the CSS transition/animation)                                | `class="my-animation ng-animate ng-move ng-move-active"` |
         * | 11. `$animate` waits for the animation to complete (via events and timeout)                                          | `class="my-animation ng-animate ng-move ng-move-active"` |
         * | 12. The animation ends and all generated CSS classes are removed from the element                                    | `class="my-animation"`                                   |
         * | 13. The returned promise is resolved.                                                                                | `class="my-animation"`                                   |
         *
         * @param {DOMElement} element the element that will be the focus of the move animation
         * @param {DOMElement} parentElement the parentElement element of the element that will be the focus of the move animation
         * @param {DOMElement} afterElement the sibling element (which is the previous element) of the element that will be the focus of the move animation
         * @param {object=} options an optional collection of styles that will be picked up by the CSS transition/animation
         * @return {Promise} the animation callback promise
        */
        move: function(element, parentElement, afterElement, options) {
          options = parseAnimateOptions(options);
          element = angular.element(element);
          parentElement = prepareElement(parentElement);
          afterElement = prepareElement(afterElement);

          cancelChildAnimations(element);
          classBasedAnimationsBlocked(element, true);
          $delegate.move(element, parentElement, afterElement);
          return runAnimationPostDigest(function(done) {
            return performAnimation('move', 'ng-move', stripCommentsFromElement(element), parentElement, afterElement, noop, options, done);
          });
        },

        /**
         * @ngdoc method
         * @name $animate#addClass
         *
         * @description
         * Triggers a custom animation event based off the className variable and then attaches the className value to the element as a CSS class.
         * Unlike the other animation methods, the animate service will suffix the className value with {@type -add} in order to provide
         * the animate service the setup and active CSS classes in order to trigger the animation (this will be skipped if no CSS transitions
         * or keyframes are defined on the -add-active or base CSS class).
         *
         * Below is a breakdown of each step that occurs during addClass animation:
         *
         * | Animation Step                                                                                         | What the element class attribute looks like                        |
         * |--------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
         * | 1. `$animate.addClass(element, 'super')` is called                                                     | `class="my-animation"`                                             |
         * | 2. `$animate` runs the JavaScript-defined animations detected on the element                           | `class="my-animation ng-animate"`                                  |
         * | 3. the `.super-add` class is added to the element                                                      | `class="my-animation ng-animate super-add"`                        |
         * | 4. `$animate` waits for a single animation frame (this performs a reflow)                              | `class="my-animation ng-animate super-add"`                        |
         * | 5. the `.super` and `.super-add-active` classes are added (this triggers the CSS transition/animation) | `class="my-animation ng-animate super super-add super-add-active"` |
         * | 6. `$animate` scans the element styles to get the CSS transition/animation duration and delay          | `class="my-animation ng-animate super super-add super-add-active"` |
         * | 7. `$animate` waits for the animation to complete (via events and timeout)                             | `class="my-animation ng-animate super super-add super-add-active"` |
         * | 8. The animation ends and all generated CSS classes are removed from the element                       | `class="my-animation super"`                                       |
         * | 9. The super class is kept on the element                                                              | `class="my-animation super"`                                       |
         * | 10. The returned promise is resolved.                                                                  | `class="my-animation super"`                                       |
         *
         * @param {DOMElement} element the element that will be animated
         * @param {string} className the CSS class that will be added to the element and then animated
         * @param {object=} options an optional collection of styles that will be picked up by the CSS transition/animation
         * @return {Promise} the animation callback promise
        */
        addClass: function(element, className, options) {
          return this.setClass(element, className, [], options);
        },

        /**
         * @ngdoc method
         * @name $animate#removeClass
         *
         * @description
         * Triggers a custom animation event based off the className variable and then removes the CSS class provided by the className value
         * from the element. Unlike the other animation methods, the animate service will suffix the className value with {@type -remove} in
         * order to provide the animate service the setup and active CSS classes in order to trigger the animation (this will be skipped if
         * no CSS transitions or keyframes are defined on the -remove or base CSS classes).
         *
         * Below is a breakdown of each step that occurs during removeClass animation:
         *
         * | Animation Step                                                                                                       | What the element class attribute looks like                        |
         * |----------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
         * | 1. `$animate.removeClass(element, 'super')` is called                                                                | `class="my-animation super"`                                       |
         * | 2. `$animate` runs the JavaScript-defined animations detected on the element                                         | `class="my-animation super ng-animate"`                            |
         * | 3. the `.super-remove` class is added to the element                                                                 | `class="my-animation super ng-animate super-remove"`               |
         * | 4. `$animate` waits for a single animation frame (this performs a reflow)                                            | `class="my-animation super ng-animate super-remove"`               |
         * | 5. the `.super-remove-active` classes are added and `.super` is removed (this triggers the CSS transition/animation) | `class="my-animation ng-animate super-remove super-remove-active"` |
         * | 6. `$animate` scans the element styles to get the CSS transition/animation duration and delay                        | `class="my-animation ng-animate super-remove super-remove-active"` |
         * | 7. `$animate` waits for the animation to complete (via events and timeout)                                           | `class="my-animation ng-animate super-remove super-remove-active"` |
         * | 8. The animation ends and all generated CSS classes are removed from the element                                     | `class="my-animation"`                                             |
         * | 9. The returned promise is resolved.                                                                                 | `class="my-animation"`                                             |
         *
         *
         * @param {DOMElement} element the element that will be animated
         * @param {string} className the CSS class that will be animated and then removed from the element
         * @param {object=} options an optional collection of styles that will be picked up by the CSS transition/animation
         * @return {Promise} the animation callback promise
        */
        removeClass: function(element, className, options) {
          return this.setClass(element, [], className, options);
        },

        /**
         *
         * @ngdoc method
         * @name $animate#setClass
         *
         * @description Adds and/or removes the given CSS classes to and from the element.
         * Once complete, the `done()` callback will be fired (if provided).
         *
         * | Animation Step                                                                                                                               | What the element class attribute looks like                                            |
         * |----------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
         * | 1. `$animate.setClass(element, 'on', 'off')` is called                                                                                       | `class="my-animation off"`                                                             |
         * | 2. `$animate` runs the JavaScript-defined animations detected on the element                                                                 | `class="my-animation ng-animate off"`                                                  |
         * | 3. the `.on-add` and `.off-remove` classes are added to the element                                                                          | `class="my-animation ng-animate on-add off-remove off"`                                |
         * | 4. `$animate` waits for a single animation frame (this performs a reflow)                                                                    | `class="my-animation ng-animate on-add off-remove off"`                                |
         * | 5. the `.on`, `.on-add-active` and `.off-remove-active` classes are added and `.off` is removed (this triggers the CSS transition/animation) | `class="my-animation ng-animate on on-add on-add-active off-remove off-remove-active"` |
         * | 6. `$animate` scans the element styles to get the CSS transition/animation duration and delay                                                | `class="my-animation ng-animate on on-add on-add-active off-remove off-remove-active"` |
         * | 7. `$animate` waits for the animation to complete (via events and timeout)                                                                   | `class="my-animation ng-animate on on-add on-add-active off-remove off-remove-active"` |
         * | 8. The animation ends and all generated CSS classes are removed from the element                                                             | `class="my-animation on"`                                                              |
         * | 9. The returned promise is resolved.                                                                                                         | `class="my-animation on"`                                                              |
         *
         * @param {DOMElement} element the element which will have its CSS classes changed
         *   removed from it
         * @param {string} add the CSS classes which will be added to the element
         * @param {string} remove the CSS class which will be removed from the element
         *   CSS classes have been set on the element
         * @param {object=} options an optional collection of styles that will be picked up by the CSS transition/animation
         * @return {Promise} the animation callback promise
         */
        setClass: function(element, add, remove, options) {
          options = parseAnimateOptions(options);

          var STORAGE_KEY = '$$animateClasses';
          element = angular.element(element);
          element = stripCommentsFromElement(element);

          if (classBasedAnimationsBlocked(element)) {
            return $delegate.$$setClassImmediately(element, add, remove, options);
          }

          // we're using a combined array for both the add and remove
          // operations since the ORDER OF addClass and removeClass matters
          var classes, cache = element.data(STORAGE_KEY);
          var hasCache = !!cache;
          if (!cache) {
            cache = {};
            cache.classes = {};
          }
          classes = cache.classes;

          add = isArray(add) ? add : add.split(' ');
          forEach(add, function(c) {
            if (c && c.length) {
              classes[c] = true;
            }
          });

          remove = isArray(remove) ? remove : remove.split(' ');
          forEach(remove, function(c) {
            if (c && c.length) {
              classes[c] = false;
            }
          });

          if (hasCache) {
            if (options && cache.options) {
              cache.options = angular.extend(cache.options || {}, options);
            }

            //the digest cycle will combine all the animations into one function
            return cache.promise;
          } else {
            element.data(STORAGE_KEY, cache = {
              classes: classes,
              options: options
            });
          }

          return cache.promise = runAnimationPostDigest(function(done) {
            var parentElement = element.parent();
            var elementNode = extractElementNode(element);
            var parentNode = elementNode.parentNode;
            // TODO(matsko): move this code into the animationsDisabled() function once #8092 is fixed
            if (!parentNode || parentNode['$$NG_REMOVED'] || elementNode['$$NG_REMOVED']) {
              done();
              return;
            }

            var cache = element.data(STORAGE_KEY);
            element.removeData(STORAGE_KEY);

            var state = element.data(NG_ANIMATE_STATE) || {};
            var classes = resolveElementClasses(element, cache, state.active);
            return !classes
              ? done()
              : performAnimation('setClass', classes, element, parentElement, null, function() {
                  if (classes[0]) $delegate.$$addClassImmediately(element, classes[0]);
                  if (classes[1]) $delegate.$$removeClassImmediately(element, classes[1]);
                }, cache.options, done);
          });
        },

        /**
         * @ngdoc method
         * @name $animate#cancel
         * @kind function
         *
         * @param {Promise} animationPromise The animation promise that is returned when an animation is started.
         *
         * @description
         * Cancels the provided animation.
        */
        cancel: function(promise) {
          promise.$$cancelFn();
        },

        /**
         * @ngdoc method
         * @name $animate#enabled
         * @kind function
         *
         * @param {boolean=} value If provided then set the animation on or off.
         * @param {DOMElement=} element If provided then the element will be used to represent the enable/disable operation
         * @return {boolean} Current animation state.
         *
         * @description
         * Globally enables/disables animations.
         *
        */
        enabled: function(value, element) {
          switch (arguments.length) {
            case 2:
              if (value) {
                cleanup(element);
              } else {
                var data = element.data(NG_ANIMATE_STATE) || {};
                data.disabled = true;
                element.data(NG_ANIMATE_STATE, data);
              }
            break;

            case 1:
              rootAnimateState.disabled = !value;
            break;

            default:
              value = !rootAnimateState.disabled;
            break;
          }
          return !!value;
         }
      };

      /*
        all animations call this shared animation triggering function internally.
        The animationEvent variable refers to the JavaScript animation event that will be triggered
        and the className value is the name of the animation that will be applied within the
        CSS code. Element, `parentElement` and `afterElement` are provided DOM elements for the animation
        and the onComplete callback will be fired once the animation is fully complete.
      */
      function performAnimation(animationEvent, className, element, parentElement, afterElement, domOperation, options, doneCallback) {
        var noopCancel = noop;
        var runner = animationRunner(element, animationEvent, className, options);
        if (!runner) {
          fireDOMOperation();
          fireBeforeCallbackAsync();
          fireAfterCallbackAsync();
          closeAnimation();
          return noopCancel;
        }

        animationEvent = runner.event;
        className = runner.className;
        var elementEvents = angular.element._data(runner.node);
        elementEvents = elementEvents && elementEvents.events;

        if (!parentElement) {
          parentElement = afterElement ? afterElement.parent() : element.parent();
        }

        //skip the animation if animations are disabled, a parent is already being animated,
        //the element is not currently attached to the document body or then completely close
        //the animation if any matching animations are not found at all.
        //NOTE: IE8 + IE9 should close properly (run closeAnimation()) in case an animation was found.
        if (animationsDisabled(element, parentElement)) {
          fireDOMOperation();
          fireBeforeCallbackAsync();
          fireAfterCallbackAsync();
          closeAnimation();
          return noopCancel;
        }

        var ngAnimateState  = element.data(NG_ANIMATE_STATE) || {};
        var runningAnimations     = ngAnimateState.active || {};
        var totalActiveAnimations = ngAnimateState.totalActive || 0;
        var lastAnimation         = ngAnimateState.last;
        var skipAnimation = false;

        if (totalActiveAnimations > 0) {
          var animationsToCancel = [];
          if (!runner.isClassBased) {
            if (animationEvent == 'leave' && runningAnimations['ng-leave']) {
              skipAnimation = true;
            } else {
              //cancel all animations when a structural animation takes place
              for (var klass in runningAnimations) {
                animationsToCancel.push(runningAnimations[klass]);
              }
              ngAnimateState = {};
              cleanup(element, true);
            }
          } else if (lastAnimation.event == 'setClass') {
            animationsToCancel.push(lastAnimation);
            cleanup(element, className);
          } else if (runningAnimations[className]) {
            var current = runningAnimations[className];
            if (current.event == animationEvent) {
              skipAnimation = true;
            } else {
              animationsToCancel.push(current);
              cleanup(element, className);
            }
          }

          if (animationsToCancel.length > 0) {
            forEach(animationsToCancel, function(operation) {
              operation.cancel();
            });
          }
        }

        if (runner.isClassBased
            && !runner.isSetClassOperation
            && animationEvent != 'animate'
            && !skipAnimation) {
          skipAnimation = (animationEvent == 'addClass') == element.hasClass(className); //opposite of XOR
        }

        if (skipAnimation) {
          fireDOMOperation();
          fireBeforeCallbackAsync();
          fireAfterCallbackAsync();
          fireDoneCallbackAsync();
          return noopCancel;
        }

        runningAnimations     = ngAnimateState.active || {};
        totalActiveAnimations = ngAnimateState.totalActive || 0;

        if (animationEvent == 'leave') {
          //there's no need to ever remove the listener since the element
          //will be removed (destroyed) after the leave animation ends or
          //is cancelled midway
          element.one('$destroy', function(e) {
            var element = angular.element(this);
            var state = element.data(NG_ANIMATE_STATE);
            if (state) {
              var activeLeaveAnimation = state.active['ng-leave'];
              if (activeLeaveAnimation) {
                activeLeaveAnimation.cancel();
                cleanup(element, 'ng-leave');
              }
            }
          });
        }

        //the ng-animate class does nothing, but it's here to allow for
        //parent animations to find and cancel child animations when needed
        $$jqLite.addClass(element, NG_ANIMATE_CLASS_NAME);
        if (options && options.tempClasses) {
          forEach(options.tempClasses, function(className) {
            $$jqLite.addClass(element, className);
          });
        }

        var localAnimationCount = globalAnimationCounter++;
        totalActiveAnimations++;
        runningAnimations[className] = runner;

        element.data(NG_ANIMATE_STATE, {
          last: runner,
          active: runningAnimations,
          index: localAnimationCount,
          totalActive: totalActiveAnimations
        });

        //first we run the before animations and when all of those are complete
        //then we perform the DOM operation and run the next set of animations
        fireBeforeCallbackAsync();
        runner.before(function(cancelled) {
          var data = element.data(NG_ANIMATE_STATE);
          cancelled = cancelled ||
                        !data || !data.active[className] ||
                        (runner.isClassBased && data.active[className].event != animationEvent);

          fireDOMOperation();
          if (cancelled === true) {
            closeAnimation();
          } else {
            fireAfterCallbackAsync();
            runner.after(closeAnimation);
          }
        });

        return runner.cancel;

        function fireDOMCallback(animationPhase) {
          var eventName = '$animate:' + animationPhase;
          if (elementEvents && elementEvents[eventName] && elementEvents[eventName].length > 0) {
            $$asyncCallback(function() {
              element.triggerHandler(eventName, {
                event: animationEvent,
                className: className
              });
            });
          }
        }

        function fireBeforeCallbackAsync() {
          fireDOMCallback('before');
        }

        function fireAfterCallbackAsync() {
          fireDOMCallback('after');
        }

        function fireDoneCallbackAsync() {
          fireDOMCallback('close');
          doneCallback();
        }

        //it is less complicated to use a flag than managing and canceling
        //timeouts containing multiple callbacks.
        function fireDOMOperation() {
          if (!fireDOMOperation.hasBeenRun) {
            fireDOMOperation.hasBeenRun = true;
            domOperation();
          }
        }

        function closeAnimation() {
          if (!closeAnimation.hasBeenRun) {
            if (runner) { //the runner doesn't exist if it fails to instantiate
              runner.applyStyles();
            }

            closeAnimation.hasBeenRun = true;
            if (options && options.tempClasses) {
              forEach(options.tempClasses, function(className) {
                $$jqLite.removeClass(element, className);
              });
            }

            var data = element.data(NG_ANIMATE_STATE);
            if (data) {

              /* only structural animations wait for reflow before removing an
                 animation, but class-based animations don't. An example of this
                 failing would be when a parent HTML tag has a ng-class attribute
                 causing ALL directives below to skip animations during the digest */
              if (runner && runner.isClassBased) {
                cleanup(element, className);
              } else {
                $$asyncCallback(function() {
                  var data = element.data(NG_ANIMATE_STATE) || {};
                  if (localAnimationCount == data.index) {
                    cleanup(element, className, animationEvent);
                  }
                });
                element.data(NG_ANIMATE_STATE, data);
              }
            }
            fireDoneCallbackAsync();
          }
        }
      }

      function cancelChildAnimations(element) {
        var node = extractElementNode(element);
        if (node) {
          var nodes = angular.isFunction(node.getElementsByClassName) ?
            node.getElementsByClassName(NG_ANIMATE_CLASS_NAME) :
            node.querySelectorAll('.' + NG_ANIMATE_CLASS_NAME);
          forEach(nodes, function(element) {
            element = angular.element(element);
            var data = element.data(NG_ANIMATE_STATE);
            if (data && data.active) {
              forEach(data.active, function(runner) {
                runner.cancel();
              });
            }
          });
        }
      }

      function cleanup(element, className) {
        if (isMatchingElement(element, $rootElement)) {
          if (!rootAnimateState.disabled) {
            rootAnimateState.running = false;
            rootAnimateState.structural = false;
          }
        } else if (className) {
          var data = element.data(NG_ANIMATE_STATE) || {};

          var removeAnimations = className === true;
          if (!removeAnimations && data.active && data.active[className]) {
            data.totalActive--;
            delete data.active[className];
          }

          if (removeAnimations || !data.totalActive) {
            $$jqLite.removeClass(element, NG_ANIMATE_CLASS_NAME);
            element.removeData(NG_ANIMATE_STATE);
          }
        }
      }

      function animationsDisabled(element, parentElement) {
        if (rootAnimateState.disabled) {
          return true;
        }

        if (isMatchingElement(element, $rootElement)) {
          return rootAnimateState.running;
        }

        var allowChildAnimations, parentRunningAnimation, hasParent;
        do {
          //the element did not reach the root element which means that it
          //is not apart of the DOM. Therefore there is no reason to do
          //any animations on it
          if (parentElement.length === 0) break;

          var isRoot = isMatchingElement(parentElement, $rootElement);
          var state = isRoot ? rootAnimateState : (parentElement.data(NG_ANIMATE_STATE) || {});
          if (state.disabled) {
            return true;
          }

          //no matter what, for an animation to work it must reach the root element
          //this implies that the element is attached to the DOM when the animation is run
          if (isRoot) {
            hasParent = true;
          }

          //once a flag is found that is strictly false then everything before
          //it will be discarded and all child animations will be restricted
          if (allowChildAnimations !== false) {
            var animateChildrenFlag = parentElement.data(NG_ANIMATE_CHILDREN);
            if (angular.isDefined(animateChildrenFlag)) {
              allowChildAnimations = animateChildrenFlag;
            }
          }

          parentRunningAnimation = parentRunningAnimation ||
                                   state.running ||
                                   (state.last && !state.last.isClassBased);
        }
        while (parentElement = parentElement.parent());

        return !hasParent || (!allowChildAnimations && parentRunningAnimation);
      }
    }]);

    $animateProvider.register('', ['$window', '$sniffer', '$timeout', '$$animateReflow',
                           function($window,   $sniffer,   $timeout,   $$animateReflow) {
      // Detect proper transitionend/animationend event names.
      var CSS_PREFIX = '', TRANSITION_PROP, TRANSITIONEND_EVENT, ANIMATION_PROP, ANIMATIONEND_EVENT;

      // If unprefixed events are not supported but webkit-prefixed are, use the latter.
      // Otherwise, just use W3C names, browsers not supporting them at all will just ignore them.
      // Note: Chrome implements `window.onwebkitanimationend` and doesn't implement `window.onanimationend`
      // but at the same time dispatches the `animationend` event and not `webkitAnimationEnd`.
      // Register both events in case `window.onanimationend` is not supported because of that,
      // do the same for `transitionend` as Safari is likely to exhibit similar behavior.
      // Also, the only modern browser that uses vendor prefixes for transitions/keyframes is webkit
      // therefore there is no reason to test anymore for other vendor prefixes: http://caniuse.com/#search=transition
      if (window.ontransitionend === undefined && window.onwebkittransitionend !== undefined) {
        CSS_PREFIX = '-webkit-';
        TRANSITION_PROP = 'WebkitTransition';
        TRANSITIONEND_EVENT = 'webkitTransitionEnd transitionend';
      } else {
        TRANSITION_PROP = 'transition';
        TRANSITIONEND_EVENT = 'transitionend';
      }

      if (window.onanimationend === undefined && window.onwebkitanimationend !== undefined) {
        CSS_PREFIX = '-webkit-';
        ANIMATION_PROP = 'WebkitAnimation';
        ANIMATIONEND_EVENT = 'webkitAnimationEnd animationend';
      } else {
        ANIMATION_PROP = 'animation';
        ANIMATIONEND_EVENT = 'animationend';
      }

      var DURATION_KEY = 'Duration';
      var PROPERTY_KEY = 'Property';
      var DELAY_KEY = 'Delay';
      var ANIMATION_ITERATION_COUNT_KEY = 'IterationCount';
      var ANIMATION_PLAYSTATE_KEY = 'PlayState';
      var NG_ANIMATE_PARENT_KEY = '$$ngAnimateKey';
      var NG_ANIMATE_CSS_DATA_KEY = '$$ngAnimateCSS3Data';
      var ELAPSED_TIME_MAX_DECIMAL_PLACES = 3;
      var CLOSING_TIME_BUFFER = 1.5;
      var ONE_SECOND = 1000;

      var lookupCache = {};
      var parentCounter = 0;
      var animationReflowQueue = [];
      var cancelAnimationReflow;
      function clearCacheAfterReflow() {
        if (!cancelAnimationReflow) {
          cancelAnimationReflow = $$animateReflow(function() {
            animationReflowQueue = [];
            cancelAnimationReflow = null;
            lookupCache = {};
          });
        }
      }

      function afterReflow(element, callback) {
        if (cancelAnimationReflow) {
          cancelAnimationReflow();
        }
        animationReflowQueue.push(callback);
        cancelAnimationReflow = $$animateReflow(function() {
          forEach(animationReflowQueue, function(fn) {
            fn();
          });

          animationReflowQueue = [];
          cancelAnimationReflow = null;
          lookupCache = {};
        });
      }

      var closingTimer = null;
      var closingTimestamp = 0;
      var animationElementQueue = [];
      function animationCloseHandler(element, totalTime) {
        var node = extractElementNode(element);
        element = angular.element(node);

        //this item will be garbage collected by the closing
        //animation timeout
        animationElementQueue.push(element);

        //but it may not need to cancel out the existing timeout
        //if the timestamp is less than the previous one
        var futureTimestamp = Date.now() + totalTime;
        if (futureTimestamp <= closingTimestamp) {
          return;
        }

        $timeout.cancel(closingTimer);

        closingTimestamp = futureTimestamp;
        closingTimer = $timeout(function() {
          closeAllAnimations(animationElementQueue);
          animationElementQueue = [];
        }, totalTime, false);
      }

      function closeAllAnimations(elements) {
        forEach(elements, function(element) {
          var elementData = element.data(NG_ANIMATE_CSS_DATA_KEY);
          if (elementData) {
            forEach(elementData.closeAnimationFns, function(fn) {
              fn();
            });
          }
        });
      }

      function getElementAnimationDetails(element, cacheKey) {
        var data = cacheKey ? lookupCache[cacheKey] : null;
        if (!data) {
          var transitionDuration = 0;
          var transitionDelay = 0;
          var animationDuration = 0;
          var animationDelay = 0;

          //we want all the styles defined before and after
          forEach(element, function(element) {
            if (element.nodeType == ELEMENT_NODE) {
              var elementStyles = $window.getComputedStyle(element) || {};

              var transitionDurationStyle = elementStyles[TRANSITION_PROP + DURATION_KEY];
              transitionDuration = Math.max(parseMaxTime(transitionDurationStyle), transitionDuration);

              var transitionDelayStyle = elementStyles[TRANSITION_PROP + DELAY_KEY];
              transitionDelay  = Math.max(parseMaxTime(transitionDelayStyle), transitionDelay);

              var animationDelayStyle = elementStyles[ANIMATION_PROP + DELAY_KEY];
              animationDelay   = Math.max(parseMaxTime(elementStyles[ANIMATION_PROP + DELAY_KEY]), animationDelay);

              var aDuration  = parseMaxTime(elementStyles[ANIMATION_PROP + DURATION_KEY]);

              if (aDuration > 0) {
                aDuration *= parseInt(elementStyles[ANIMATION_PROP + ANIMATION_ITERATION_COUNT_KEY], 10) || 1;
              }
              animationDuration = Math.max(aDuration, animationDuration);
            }
          });
          data = {
            total: 0,
            transitionDelay: transitionDelay,
            transitionDuration: transitionDuration,
            animationDelay: animationDelay,
            animationDuration: animationDuration
          };
          if (cacheKey) {
            lookupCache[cacheKey] = data;
          }
        }
        return data;
      }

      function parseMaxTime(str) {
        var maxValue = 0;
        var values = isString(str) ?
          str.split(/\s*,\s*/) :
          [];
        forEach(values, function(value) {
          maxValue = Math.max(parseFloat(value) || 0, maxValue);
        });
        return maxValue;
      }

      function getCacheKey(element) {
        var parentElement = element.parent();
        var parentID = parentElement.data(NG_ANIMATE_PARENT_KEY);
        if (!parentID) {
          parentElement.data(NG_ANIMATE_PARENT_KEY, ++parentCounter);
          parentID = parentCounter;
        }
        return parentID + '-' + extractElementNode(element).getAttribute('class');
      }

      function animateSetup(animationEvent, element, className, styles) {
        var structural = ['ng-enter','ng-leave','ng-move'].indexOf(className) >= 0;

        var cacheKey = getCacheKey(element);
        var eventCacheKey = cacheKey + ' ' + className;
        var itemIndex = lookupCache[eventCacheKey] ? ++lookupCache[eventCacheKey].total : 0;

        var stagger = {};
        if (itemIndex > 0) {
          var staggerClassName = className + '-stagger';
          var staggerCacheKey = cacheKey + ' ' + staggerClassName;
          var applyClasses = !lookupCache[staggerCacheKey];

          applyClasses && $$jqLite.addClass(element, staggerClassName);

          stagger = getElementAnimationDetails(element, staggerCacheKey);

          applyClasses && $$jqLite.removeClass(element, staggerClassName);
        }

        $$jqLite.addClass(element, className);

        var formerData = element.data(NG_ANIMATE_CSS_DATA_KEY) || {};
        var timings = getElementAnimationDetails(element, eventCacheKey);
        var transitionDuration = timings.transitionDuration;
        var animationDuration = timings.animationDuration;

        if (structural && transitionDuration === 0 && animationDuration === 0) {
          $$jqLite.removeClass(element, className);
          return false;
        }

        var blockTransition = styles || (structural && transitionDuration > 0);
        var blockAnimation = animationDuration > 0 &&
                             stagger.animationDelay > 0 &&
                             stagger.animationDuration === 0;

        var closeAnimationFns = formerData.closeAnimationFns || [];
        element.data(NG_ANIMATE_CSS_DATA_KEY, {
          stagger: stagger,
          cacheKey: eventCacheKey,
          running: formerData.running || 0,
          itemIndex: itemIndex,
          blockTransition: blockTransition,
          closeAnimationFns: closeAnimationFns
        });

        var node = extractElementNode(element);

        if (blockTransition) {
          blockTransitions(node, true);
          if (styles) {
            element.css(styles);
          }
        }

        if (blockAnimation) {
          blockAnimations(node, true);
        }

        return true;
      }

      function animateRun(animationEvent, element, className, activeAnimationComplete, styles) {
        var node = extractElementNode(element);
        var elementData = element.data(NG_ANIMATE_CSS_DATA_KEY);
        if (node.getAttribute('class').indexOf(className) == -1 || !elementData) {
          activeAnimationComplete();
          return;
        }

        var activeClassName = '';
        var pendingClassName = '';
        forEach(className.split(' '), function(klass, i) {
          var prefix = (i > 0 ? ' ' : '') + klass;
          activeClassName += prefix + '-active';
          pendingClassName += prefix + '-pending';
        });

        var style = '';
        var appliedStyles = [];
        var itemIndex = elementData.itemIndex;
        var stagger = elementData.stagger;
        var staggerTime = 0;
        if (itemIndex > 0) {
          var transitionStaggerDelay = 0;
          if (stagger.transitionDelay > 0 && stagger.transitionDuration === 0) {
            transitionStaggerDelay = stagger.transitionDelay * itemIndex;
          }

          var animationStaggerDelay = 0;
          if (stagger.animationDelay > 0 && stagger.animationDuration === 0) {
            animationStaggerDelay = stagger.animationDelay * itemIndex;
            appliedStyles.push(CSS_PREFIX + 'animation-play-state');
          }

          staggerTime = Math.round(Math.max(transitionStaggerDelay, animationStaggerDelay) * 100) / 100;
        }

        if (!staggerTime) {
          $$jqLite.addClass(element, activeClassName);
          if (elementData.blockTransition) {
            blockTransitions(node, false);
          }
        }

        var eventCacheKey = elementData.cacheKey + ' ' + activeClassName;
        var timings = getElementAnimationDetails(element, eventCacheKey);
        var maxDuration = Math.max(timings.transitionDuration, timings.animationDuration);
        if (maxDuration === 0) {
          $$jqLite.removeClass(element, activeClassName);
          animateClose(element, className);
          activeAnimationComplete();
          return;
        }

        if (!staggerTime && styles && Object.keys(styles).length > 0) {
          if (!timings.transitionDuration) {
            element.css('transition', timings.animationDuration + 's linear all');
            appliedStyles.push('transition');
          }
          element.css(styles);
        }

        var maxDelay = Math.max(timings.transitionDelay, timings.animationDelay);
        var maxDelayTime = maxDelay * ONE_SECOND;

        if (appliedStyles.length > 0) {
          //the element being animated may sometimes contain comment nodes in
          //the jqLite object, so we're safe to use a single variable to house
          //the styles since there is always only one element being animated
          var oldStyle = node.getAttribute('style') || '';
          if (oldStyle.charAt(oldStyle.length - 1) !== ';') {
            oldStyle += ';';
          }
          node.setAttribute('style', oldStyle + ' ' + style);
        }

        var startTime = Date.now();
        var css3AnimationEvents = ANIMATIONEND_EVENT + ' ' + TRANSITIONEND_EVENT;
        var animationTime     = (maxDelay + maxDuration) * CLOSING_TIME_BUFFER;
        var totalTime         = (staggerTime + animationTime) * ONE_SECOND;

        var staggerTimeout;
        if (staggerTime > 0) {
          $$jqLite.addClass(element, pendingClassName);
          staggerTimeout = $timeout(function() {
            staggerTimeout = null;

            if (timings.transitionDuration > 0) {
              blockTransitions(node, false);
            }
            if (timings.animationDuration > 0) {
              blockAnimations(node, false);
            }

            $$jqLite.addClass(element, activeClassName);
            $$jqLite.removeClass(element, pendingClassName);

            if (styles) {
              if (timings.transitionDuration === 0) {
                element.css('transition', timings.animationDuration + 's linear all');
              }
              element.css(styles);
              appliedStyles.push('transition');
            }
          }, staggerTime * ONE_SECOND, false);
        }

        element.on(css3AnimationEvents, onAnimationProgress);
        elementData.closeAnimationFns.push(function() {
          onEnd();
          activeAnimationComplete();
        });

        elementData.running++;
        animationCloseHandler(element, totalTime);
        return onEnd;

        // This will automatically be called by $animate so
        // there is no need to attach this internally to the
        // timeout done method.
        function onEnd() {
          element.off(css3AnimationEvents, onAnimationProgress);
          $$jqLite.removeClass(element, activeClassName);
          $$jqLite.removeClass(element, pendingClassName);
          if (staggerTimeout) {
            $timeout.cancel(staggerTimeout);
          }
          animateClose(element, className);
          var node = extractElementNode(element);
          for (var i in appliedStyles) {
            node.style.removeProperty(appliedStyles[i]);
          }
        }

        function onAnimationProgress(event) {
          event.stopPropagation();
          var ev = event.originalEvent || event;
          var timeStamp = ev.$manualTimeStamp || ev.timeStamp || Date.now();

          /* Firefox (or possibly just Gecko) likes to not round values up
           * when a ms measurement is used for the animation */
          var elapsedTime = parseFloat(ev.elapsedTime.toFixed(ELAPSED_TIME_MAX_DECIMAL_PLACES));

          /* $manualTimeStamp is a mocked timeStamp value which is set
           * within browserTrigger(). This is only here so that tests can
           * mock animations properly. Real events fallback to event.timeStamp,
           * or, if they don't, then a timeStamp is automatically created for them.
           * We're checking to see if the timeStamp surpasses the expected delay,
           * but we're using elapsedTime instead of the timeStamp on the 2nd
           * pre-condition since animations sometimes close off early */
          if (Math.max(timeStamp - startTime, 0) >= maxDelayTime && elapsedTime >= maxDuration) {
            activeAnimationComplete();
          }
        }
      }

      function blockTransitions(node, bool) {
        node.style[TRANSITION_PROP + PROPERTY_KEY] = bool ? 'none' : '';
      }

      function blockAnimations(node, bool) {
        node.style[ANIMATION_PROP + ANIMATION_PLAYSTATE_KEY] = bool ? 'paused' : '';
      }

      function animateBefore(animationEvent, element, className, styles) {
        if (animateSetup(animationEvent, element, className, styles)) {
          return function(cancelled) {
            cancelled && animateClose(element, className);
          };
        }
      }

      function animateAfter(animationEvent, element, className, afterAnimationComplete, styles) {
        if (element.data(NG_ANIMATE_CSS_DATA_KEY)) {
          return animateRun(animationEvent, element, className, afterAnimationComplete, styles);
        } else {
          animateClose(element, className);
          afterAnimationComplete();
        }
      }

      function animate(animationEvent, element, className, animationComplete, options) {
        //If the animateSetup function doesn't bother returning a
        //cancellation function then it means that there is no animation
        //to perform at all
        var preReflowCancellation = animateBefore(animationEvent, element, className, options.from);
        if (!preReflowCancellation) {
          clearCacheAfterReflow();
          animationComplete();
          return;
        }

        //There are two cancellation functions: one is before the first
        //reflow animation and the second is during the active state
        //animation. The first function will take care of removing the
        //data from the element which will not make the 2nd animation
        //happen in the first place
        var cancel = preReflowCancellation;
        afterReflow(element, function() {
          //once the reflow is complete then we point cancel to
          //the new cancellation function which will remove all of the
          //animation properties from the active animation
          cancel = animateAfter(animationEvent, element, className, animationComplete, options.to);
        });

        return function(cancelled) {
          (cancel || noop)(cancelled);
        };
      }

      function animateClose(element, className) {
        $$jqLite.removeClass(element, className);
        var data = element.data(NG_ANIMATE_CSS_DATA_KEY);
        if (data) {
          if (data.running) {
            data.running--;
          }
          if (!data.running || data.running === 0) {
            element.removeData(NG_ANIMATE_CSS_DATA_KEY);
          }
        }
      }

      return {
        animate: function(element, className, from, to, animationCompleted, options) {
          options = options || {};
          options.from = from;
          options.to = to;
          return animate('animate', element, className, animationCompleted, options);
        },

        enter: function(element, animationCompleted, options) {
          options = options || {};
          return animate('enter', element, 'ng-enter', animationCompleted, options);
        },

        leave: function(element, animationCompleted, options) {
          options = options || {};
          return animate('leave', element, 'ng-leave', animationCompleted, options);
        },

        move: function(element, animationCompleted, options) {
          options = options || {};
          return animate('move', element, 'ng-move', animationCompleted, options);
        },

        beforeSetClass: function(element, add, remove, animationCompleted, options) {
          options = options || {};
          var className = suffixClasses(remove, '-remove') + ' ' +
                          suffixClasses(add, '-add');
          var cancellationMethod = animateBefore('setClass', element, className, options.from);
          if (cancellationMethod) {
            afterReflow(element, animationCompleted);
            return cancellationMethod;
          }
          clearCacheAfterReflow();
          animationCompleted();
        },

        beforeAddClass: function(element, className, animationCompleted, options) {
          options = options || {};
          var cancellationMethod = animateBefore('addClass', element, suffixClasses(className, '-add'), options.from);
          if (cancellationMethod) {
            afterReflow(element, animationCompleted);
            return cancellationMethod;
          }
          clearCacheAfterReflow();
          animationCompleted();
        },

        beforeRemoveClass: function(element, className, animationCompleted, options) {
          options = options || {};
          var cancellationMethod = animateBefore('removeClass', element, suffixClasses(className, '-remove'), options.from);
          if (cancellationMethod) {
            afterReflow(element, animationCompleted);
            return cancellationMethod;
          }
          clearCacheAfterReflow();
          animationCompleted();
        },

        setClass: function(element, add, remove, animationCompleted, options) {
          options = options || {};
          remove = suffixClasses(remove, '-remove');
          add = suffixClasses(add, '-add');
          var className = remove + ' ' + add;
          return animateAfter('setClass', element, className, animationCompleted, options.to);
        },

        addClass: function(element, className, animationCompleted, options) {
          options = options || {};
          return animateAfter('addClass', element, suffixClasses(className, '-add'), animationCompleted, options.to);
        },

        removeClass: function(element, className, animationCompleted, options) {
          options = options || {};
          return animateAfter('removeClass', element, suffixClasses(className, '-remove'), animationCompleted, options.to);
        }
      };

      function suffixClasses(classes, suffix) {
        var className = '';
        classes = isArray(classes) ? classes : classes.split(/\s+/);
        forEach(classes, function(klass, i) {
          if (klass && klass.length > 0) {
            className += (i > 0 ? ' ' : '') + klass + suffix;
          }
        });
        return className;
      }
    }]);
  }]);


})(window, window.angular);
