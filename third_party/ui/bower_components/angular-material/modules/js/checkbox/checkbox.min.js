/*!
 * Angular Material Design
 * https://github.com/angular/material
 * @license MIT
 * v0.7.0-rc3
 */
!function(){"use strict";function e(e,i,n,t,c,r){function a(i,a){return a.type="checkbox",a.tabIndex=0,i.attr("role",a.type),function(a,l,o,s){function u(e){e.which===t.KEY_CODE.SPACE&&(e.preventDefault(),m(e))}function m(e){l[0].hasAttribute("disabled")||a.$apply(function(){k=!k,s.$setViewValue(k,e&&e.type),s.$render()})}function p(){k=s.$viewValue,k?l.addClass(d):l.removeClass(d)}s=s||r.fakeNgModel();var k=!1;c(l),n.expectWithText(i,"aria-label"),e.link.pre(a,{on:angular.noop,0:{}},o,[s]),o.mdNoClick||l.on("click",m),l.on("keypress",u),s.$render=p}}e=e[0];var d="md-checked";return{restrict:"E",transclude:!0,require:"?ngModel",template:'<div class="md-container" md-ink-ripple md-ink-ripple-checkbox><div class="md-icon"></div></div><div ng-transclude class="md-label"></div>',compile:a}}angular.module("material.components.checkbox",["material.core"]).directive("mdCheckbox",e),e.$inject=["inputDirective","$mdInkRipple","$mdAria","$mdConstant","$mdTheming","$mdUtil"]}();