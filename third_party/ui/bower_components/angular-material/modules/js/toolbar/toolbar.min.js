/*!
 * Angular Material Design
 * https://github.com/angular/material
 * @license MIT
 * v0.7.0-rc3
 */
!function(){"use strict";function n(n,t,o,r){return{restrict:"E",controller:angular.noop,link:function(e,a,c){function i(){function r(t,o){a.parent()[0]===o.parent()[0]&&(d&&d.off("scroll",f),o.on("scroll",f),o.attr("scroll-shrink","true"),d=o,n(i))}function i(){s=a.prop("offsetHeight"),d.css("margin-top",-s*p+"px"),l()}function l(n){var o=n?n.target.scrollTop:u;S(),m=Math.min(s/p,Math.max(0,m+o-u)),a.css(t.CSS.TRANSFORM,"translate3d(0,"+-m*p+"px,0)"),d.css(t.CSS.TRANSFORM,"translate3d(0,"+(s-m)*p+"px,0)"),u=o}var s,d,m=0,u=0,p=c.mdShrinkSpeedFactor||.5,f=n.debounce(l),S=o.debounce(i,5e3);e.$on("$mdContentLoaded",r)}r(a),angular.isDefined(c.mdScrollShrink)&&i()}}}angular.module("material.components.toolbar",["material.core","material.components.content"]).directive("mdToolbar",n),n.$inject=["$$rAF","$mdConstant","$mdUtil","$mdTheming"]}();