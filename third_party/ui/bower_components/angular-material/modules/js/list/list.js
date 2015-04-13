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
 * @name material.components.list
 * @description
 * List module
 */
angular.module('material.components.list', [
  'material.core'
])
  .directive('mdList', mdListDirective)
  .directive('mdItem', mdItemDirective);

/**
 * @ngdoc directive
 * @name mdList
 * @module material.components.list
 *
 * @restrict E
 *
 * @description
 * The `<md-list>` directive is a list container for 1..n `<md-item>` tags.
 *
 * @usage
 * <hljs lang="html">
 * <md-list>
 *   <md-item ng-repeat="item in todos">
 *     <md-item-content>
 *       <div class="md-tile-left">
 *         <img ng-src="{{item.face}}" class="face" alt="{{item.who}}">
 *       </div>
 *       <div class="md-tile-content">
 *         <h3>{{item.what}}</h3>
 *         <h4>{{item.who}}</h4>
 *         <p>
 *           {{item.notes}}
 *         </p>
 *       </div>
 *     </md-item-content>
 *   </md-item>
 * </md-list>
 * </hljs>
 *
 */
function mdListDirective() {
  return {
    restrict: 'E',
    link: function($scope, $element, $attr) {
      $element.attr({
        'role' : 'list'
      });
    }
  };
}

/**
 * @ngdoc directive
 * @name mdItem
 * @module material.components.list
 *
 * @restrict E
 *
 * @description
 * The `<md-item>` directive is a container intended for row items in a `<md-list>` container.
 *
 * @usage
 * <hljs lang="html">
 *  <md-list>
 *    <md-item>
 *            Item content in list
 *    </md-item>
 *  </md-list>
 * </hljs>
 *
 */
function mdItemDirective() {
  return {
    restrict: 'E',
    link: function($scope, $element, $attr) {
      $element.attr({
        'role' : 'listitem'
      });
    }
  };
}
})();
