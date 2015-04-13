/**
 * Angular directive to convert JSON into human readable table. Inspired by https://github.com/marianoguerra/json.human.js.
 * @version v1.2.1 - 2014-12-22
 * @link https://github.com/yaru22/angular-json-human
 * @author Brian Park <yaru22@gmail.com>
 * @license MIT License, http://www.opensource.org/licenses/MIT
 */
/* global _, angular */
'use strict';
angular.module('yaru22.jsonHuman', ['yaru22.jsonHuman.tmpls']).factory('RecursionHelper', [
  '$compile',
  function ($compile) {
    var RecursionHelper = {
        compile: function (element) {
          var contents = element.contents().remove();
          var compiledContents;
          return function (scope, element) {
            if (!compiledContents) {
              compiledContents = $compile(contents);
            }
            compiledContents(scope, function (clone) {
              element.append(clone);
            });
            var json = scope.json;
            scope.isBoolean = _.isBoolean(json);
            scope.isNumber = _.isNumber(json);
            scope.isString = _.isString(json);
            scope.isPrimitive = scope.isBoolean || scope.isNumber || scope.isString;
            scope.isObject = _.isPlainObject(json);
            scope.isArray = _.isArray(json);
            scope.isEmpty = _.isEmpty(json);
          };
        }
      };
    return RecursionHelper;
  }
]).directive('jsonHuman', function () {
  return {
    restrict: 'A',
    scope: { data: '=jsonHuman' },
    templateUrl: 'template/angular-json-human-root.tmpl',
    link: function (scope) {
      scope.$watch('data', function (json) {
        if (typeof json === 'string') {
          try {
            json = JSON.parse(json);
          } catch (e) {
          }
        }
        scope.json = json;
        scope.isObject = _.isPlainObject(json);
        scope.isArray = _.isArray(json);
      });
    }
  };
}).directive('jsonHumanHelper', [
  'RecursionHelper',
  function (RecursionHelper) {
    return {
      restrict: 'A',
      scope: { json: '=jsonHumanHelper' },
      templateUrl: 'template/angular-json-human.tmpl',
      compile: function (tElem) {
        return RecursionHelper.compile(tElem);
      }
    };
  }
]);
angular.module('yaru22.jsonHuman.tmpls', []).run([
  '$templateCache',
  function ($templateCache) {
    'use strict';
    $templateCache.put('template/angular-json-human-root.tmpl', '<table class=jh-root ng-class="{ \'jh-type-array\': isArray, \'jh-type-object\': isObject }" json-human-helper=json></table>');
    $templateCache.put('template/angular-json-human.tmpl', '<span ng-if=isPrimitive ng-class="{ \'jh-type-bool\': isBoolean, \'jh-type-number\': isNumber, \'jh-type-string\': isString, \'jh-type-array\': isArray, \'jh-type-object\': isObject }">{{ json }} <span ng-if="isEmpty && isString" class=jh-empty>(Empty String)</span></span> <span ng-if="isEmpty && isArray" class=jh-empty>(Empty List)</span> <span ng-if="isEmpty && isObject" class=jh-empty>(Empty Object)</span><table ng-if="!isEmpty && !isPrimitive" ng-class="{ \'jh-type-array\': isArray, \'jh-type-object\': isObject }"><tbody><tr ng-repeat="(key, val) in json track by $index"><th class=jh-key ng-class="{ \'jh-array-key\': isArray, \'jh-object-key\': isObject }">{{ key }}</th><td class=jh-value ng-class="{ \'jh-array-value\': isArray, \'jh-object-value\': isObject }" json-human-helper=val></td></tr></tbody></table>');
  }
]);