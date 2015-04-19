## v0.15.4 2015-01-29
 * improved Traceur compatibility ("ngInject" prologue => fn.$inject = [..] arrays)

## v0.15.3 2015-01-28
 * bugfix "ngInject" directive prologue (removing and rebuilding)
 * bugfix extra newlines when rebuilding existing fn.$inject = [..] arrays

## v0.15.2 2015-01-26
 * bugfix crash on ES6 input (but ng-annotate does not yet understand ES6)

## v0.15.1 2015-01-15
 * bugfix release for compatibility with io.js

## v0.15.0 2015-01-15
 * "ngInject" directive prologue (usage like "use strict")
 * /* @ngNoInject */, ngNoInject(..) and "ngNoInject" for suppressing false positives
 * Acorn is now the default and only parser
 * removed the experimental --es6 option and made it the default

## v0.14.1 2014-12-04
 * bugfix /* @ngInject */ not working as expected in case of other matches

## v0.14.0 2014-11-27
 * support sourcemap combination and better map granularity

## v0.13.0 2014-11-18
 * match $mdDialog.show, $mdToast.show and $mdBottomSheet.show
 * improved $provide matching (.decorator, .service, .factory and .provider)

## v0.12.1 2014-11-13
 * bugfix crash when reference-following to an empty variable declarator

## v0.12.0 2014-11-10
 * improved TypeScript compatibility due to improved matching through IIFE's
 * match $injector.invoke
 * $modal.open is no longer experimental
 * reference-following is no longer experimental

## v0.11.0 2014-11-03
 * bugfix reference-following such as var Ctrl = function(dep1, dep2) {}

## v0.10.3 2014-11-03
 * match properties {name: ..}, {"name": ..} and {'name': ..} alike

## v0.10.2 2014-10-09
 * --es6 option for ES6 support via the Acorn parser (experimental)

## v0.10.1 2014-09-19
 * support stateHelperProvider.setNestedState nested children

## v0.10.0 2014-09-15
 * support stateHelperProvider.setNestedState
 * optional renaming of declarations and references (experimental)
 * further improved detection of existing fn.$inject = [..] arrays
 * improved insertion of $inject arrays in case of early return
 * improved angular module detection (reference-following)
 * restrict matching based on method context (directive, provider)

## v0.9.11 2014-08-09
 * improved detection of existing fn.$inject = [..] arrays

## v0.9.10 2014-08-07
 * reference-following (experimental)
 * ngInject(..) as an alternative to /* @ngInject */ ..
 * more flexible /* @ngInject */ placement (object literals)

## v0.9.9 2014-08-02
 * --sourcemap option for generating inline source maps

## v0.9.8 2014-07-28
 * match implicit config function: angular.module("MyMod", function(dep) {})
 * match through IIFE's

## v0.9.7 2014-07-11
 * more capable /* @ngInject */ (support function expression assignment)

## v0.9.6 2014-06-12
 * match myMod.invoke
 * more capable --regexp option (match any method callee, identifier or not)

## v0.9.5 2014-05-23
 * added ability to read from stdin and write to file
 * bugfix name of generated fn.$inject = [..] arrays (was: fn.$injects)

## v0.9.4 2014-05-19
 * stricter match: only match code inside of angular modules (except explicit)
 * ui-router declarations improvements
 * bugfix duplicated annotations arrays in case of redundant /* @ngInject */
 * indent generated fn.$inject = [..] arrays nicely

## v0.9.3 2014-05-16
 * /* @ngInject */ object literal support
 * bugfix ES5 strict mode oops
 * added more tools that support ng-annotate to README

## v0.9.2 2014-05-15
 * match $modal.open from angular-ui/bootstrap (experimental)
 * --stats option for runtime statistics (experimental)

## v0.9.1 2014-05-14
 * revert match .controller(name, ..) that was added in 0.9.0 because it
   triggered false positives

## v0.9.0 2014-05-13
 * explicit annotations using /* @ngInject */
 * --plugin option to load user plugins (experimental, 0.9.x may change API)
 * match $httpProvider.interceptors.push(function($scope) {})
 * match $httpProvider.responseInterceptors.push(function($scope) {})
 * match self and that as aliases to this for this.$get = function($scope){}
 * match .controller(name, ..) in addition to .controller("name", ..)
 * bugfix ui-router declarations
 * bugfix angular.module("MyMod").bootstrap(e, [], {}) disrupting chaining
 * even faster (~6% faster annotating angular.js)
 * add error array to API return object

## v0.8.0 2014-05-09
 * ngRoute support: $routeProvider.when("path", { .. })
 * even faster (~11% faster annotating angular.js)

## v0.7.3 2014-05-07
 * support obj.myMod.controller(..) in addition to myMod.controller(..)

## v0.7.2 2014-05-01
 * ui-router declarations improvements

## v0.7.1 2014-04-30
 * ui-router declarations improvements

## v0.7.0 2014-04-30
 * ui-router declarations support

## v0.6.0 2014-04-20
 * --single_quotes option to output '$scope' instead of "$scope"

## v0.5.0 2014-04-11
 * tweaked output: ["foo", "bar", ..] instead of ["foo","bar", ..]

## v0.4.0 2013-10-31
 * match angular.module("MyMod").animation(".class", function ..)

## v0.3.3 2013-10-03
 * bugfix .provider("foo", function($scope) ..) annotation. fixes #2

## v0.3.2 2013-09-30
 * bugfix angular.module("MyMod").constant("foo", "bar") disrupting chaining
 * match $provide.decorator (in addition to other $provide methods)

## v0.3.1 2013-09-30
 * bugfix angular.module("MyMod").value("foo", "bar") disrupting chaining

## v0.3.0 2013-09-30
 * ES5 build via defs
 * Grunt-support via grunt-ng-annotate

## v0.2.0 2013-09-06
 * better matching

## v0.1.2 2013-09-03
 * better README

## v0.1.1 2013-09-03
 * cross-platform shell script wrapper
