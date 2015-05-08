/*!
 * Angular Material Design
 * https://github.com/angular/material
 * @license MIT
 * v0.7.0-rc3
 */
!function(){"use strict";function e(e,n,t){return{restrict:"E",replace:!0,transclude:!0,template:'<h2 class="md-subheader"><span class="md-subheader-content"></span></h2>',compile:function(r,a,c){var u=r[0].outerHTML;return function(r,a){function i(e){return angular.element(e[0].querySelector(".md-subheader-content"))}t(a),c(r,function(e){i(a).append(e)}),c(r,function(c){var o=n(angular.element(u))(r);t(o),i(o).append(c),e(r,a,o)})}}}}angular.module("material.components.subheader",["material.core","material.components.sticky"]).directive("mdSubheader",e),e.$inject=["$mdSticky","$compile","$mdTheming"]}();