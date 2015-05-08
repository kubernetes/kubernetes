/*!
 * Angular Material Design
 * https://github.com/angular/material
 * @license MIT
 * v0.7.0-rc3
 */
!function(){"use strict";function n(){return function(n,e){return e||(e="swipeleft swiperight"),function(i,t,r){function o(e){e.srcEvent.stopPropagation(),angular.isFunction(t)&&n.$apply(function(){t(e)})}function u(){return a.on(e,o),function(){a.off(e)}}function c(n,e){var i=e.indexOf("pan")>-1,t=e.indexOf("swipe")>-1;return i&&n.push([Hammer.Pan,{direction:Hammer.DIRECTION_HORIZONTAL}]),t&&n.push([Hammer.Swipe,{direction:Hammer.DIRECTION_HORIZONTAL}]),n}var a=new Hammer(i[0],{recognizers:c([],e)});return r||u(),n.$on("$destroy",function(){a.destroy()}),u}}}function e(n,e){return{restrict:"A",link:t(n,e,"SwipeLeft")}}function i(n,e){return{restrict:"A",link:t(n,e,"SwipeRight")}}function t(n,e,i){return function(t,r,o){var u=i.toLowerCase(),c="md"+i,a=n(o[c])||angular.noop,f=e(t,u),p=function(n){a(t,n)};f(r,function(n){n.type==u&&p()})}}angular.module("material.components.swipe",[]).factory("$mdSwipe",n).directive("mdSwipeLeft",e).directive("mdSwipeRight",i),e.$inject=["$parse","$mdSwipe"],i.$inject=["$parse","$mdSwipe"]}();