# AngularCSS

##### CSS on-demand for AngularJS
Optimize the presentation layer of your single-page apps by dynamically injecting stylesheets as needed.

AngularCSS listens for [route](https://github.com/angular/bower-angular-route) (or [states](https://github.com/angular-ui/ui-router)) change events, adds the CSS defined on the current route and removes the CSS from the previous route. It also works with directives in the same fashion with the compile and scope destroy events. See the code samples below for more details.

##### Read the Article

[Introducing AngularCSS: CSS On-Demand for AngularJS](http://door3.com/insights/introducing-angularcss-css-demand-angularjs)

### Demos

[Angular's ngRoute Demo](http://door3.github.io/angular-css) ([source](../gh-pages/app.routes.js))

[UI Router Demo](http://door3.github.io/angular-css/states.html) ([source](../gh-pages/app.states.js))


### Quick Start

Install and manage with [Bower](http://bower.io). A [CDN](http://cdnjs.com/libraries/angular-css) is also provided by cdnjs.com

``` bash
$ bower install angular-css
```


1) Include the required JavaScript libraries in your `index.html` (ngRoute and UI Router are optional). 

``` html
<script src="/libs/angularjs/1.3.7/angular.min.js"></script>
<script src="/libs/angularjs/1.3.7/angular-routes.min.js"></script>
<script src="/libs/angular-css/angular-css.min.js"></script>
```

2) Add `door3.css` as a dependency for your app.

``` js
var myApp = angular.module('myApp', ['ngRoute','door3.css']);
```

### Examples

This module can be used by adding a css property in your routes values, directives or by calling the `$css` service methods from controllers and services.

The css property supports a string, an array of strings, object notation or an array of objects.

See examples below for more information.


#### In Directives

``` js
myApp.directive('myDirective', function () {
  return {
    restrict: 'E',
    templateUrl: 'my-directive/my-directive.html',
    /* Binding css to directives */
    css: 'my-directive/my-directive.css'
  }
});
```


#### In Controllers

``` js
myApp.controller('pageCtrl', function ($scope, $css) {

  // Binds stylesheet(s) to scope create/destroy events (recommended over add/remove)
  $css.bind({ 
    href: 'my-page/my-page.css'
  }, $scope);

  // Simply add stylesheet(s)
  $css.add('my-page/my-page.css');

  // Simply remove stylesheet(s)
  $css.remove(['my-page/my-page.css','my-page/my-page2.css']);

  // Remove all stylesheets
  $css.removeAll();

});
```


#### For Routes (Angular's ngRoute)

Requires [ngRoute](https://github.com/angular/bower-angular-route) as a dependency


``` js
myApp.config(function($routeProvider) {

  $routeProvider
    .when('/page1', {
      templateUrl: 'page1/page1.html',
      controller: 'page1Ctrl',
      /* Now you can bind css to routes */
      css: 'page1/page1.css'
    })
    .when('/page2', {
      templateUrl: 'page2/page2.html',
      controller: 'page2Ctrl',
      /* You can also enable features like bust cache, persist and preload */
      css: {
        href: 'page2/page2.css',
        bustCache: true
      }
    })
    .when('/page3', {
      templateUrl: 'page3/page3.html',
      controller: 'page3Ctrl',
      /* This is how you can include multiple stylesheets */
      css: ['page3/page3.css','page3/page3-2.css']
    })
    .when('/page4', {
      templateUrl: 'page4/page4.html',
      controller: 'page4Ctrl',
      css: [
        {
          href: 'page4/page4.css',
          persist: true
        }, {
          href: 'page4/page4.mobile.css',
          /* Media Query support via window.matchMedia API
           * This will only add the stylesheet if the breakpoint matches */
          media: 'screen and (max-width : 768px)'
        }, {
          href: 'page4/page4.print.css',
          media: 'print'
        }
      ]
    });

});
```

#### For States (UI Router)

Requires [ui.router](https://github.com/angular-ui/ui-router) as a dependency


``` js
myApp.config(function($stateProvider) {

  $stateProvider
    .state('page1', {
      url: '/page1',
      templateUrl: 'page1/page1.html',
      css: 'page1/page1.css'
    })
    .state('page2', {
      url: '/page2',
      templateUrl: 'page2/page2.html',
      css: {
        href: 'page2/page2.css',
        preload: true,
        persist: true
      }
    })
    .state('page3', {
      url: '/page3',
      templateUrl: 'page3/page3.html',
      css: ['page3/page3.css','page3/page3-2.css'],
      views: {
        'state1': {
          templateUrl: 'page3/states/page3-state1.html',
          css: 'page3/states/page3-state1.css'
        },
        'state2': {
          templateUrl: 'page3/states/page3-state2.html',
          css: ['page3/states/page3-state2.css']
        },
        'state3': {
          templateUrl: 'page3/states/page3-state3.html',
          css: {
            href: 'page3/states/page3-state3.css'
          }
        }
      }
    })
    .state('page4', {
      url: '/page4',
      templateUrl: 'page4/page4.html',
      views: {
        'state1': {
          templateUrl: 'states/page4/page4-state1.html',
          css: 'states/page4/page4-state1.css'
        },
        'state2': {
          templateUrl: 'states/page4/page4-state2.html',
          css: ['states/page4/page4-state2.css']
        },
        'state3': {
          templateUrl: 'states/page4/page4-state3.html',
          css: {
            href: 'states/page4/page4-state3.css'
          }
        }
      },
      css: [
        {
          href: 'page4/page4.css',
        }, {
          href: 'page4/page4.mobile.css',
          media: 'screen and (max-width : 768px)'
        }, {
          href: 'page4/page4.print.css',
          media: 'print'
        }
      ]
    });

});
```

### Responsive Design

AngularCSS supports "smart media queries". This means that stylesheets with media queries will be only added when the breakpoint matches.
This will significantly optimize the load time of your apps.

```js
$routeProvider
  .when('/my-page', {
    templateUrl: 'my-page/my-page.html',
    css: [
      {
        href: 'my-page/my-page.mobile.css',
        media: '(max-width: 480px)'
      }, {
        href: 'my-page/my-page.tablet.css',
        media: '(min-width: 768px) and (max-width: 1024px)'
      }, {
        href: 'my-page/my-page.desktop.css',
        media: '(min-width: 1224px)'
      }
    ]
  });
```

Even though you can use the `media` property to specify media queries, the best way to manage your breakpoins is by settings them in the provider's defaults. For example:

```js
myApp.config(function($routeProvider, $cssProvider) {

  angular.extend($cssProvider.defaults, {
    breakpoints: {
      mobile: '(max-width: 480px)',
      tablet: '(min-width: 768px) and (max-width: 1024px)',
      desktop: '(min-width: 1224px)'
    }
  });

  $routeProvider
    .when('/my-page', {
      templateUrl: 'my-page/my-page.html',
      css: [
        {
          href: 'my-page/my-page.mobile.css',
          breakpoint: 'mobile'
        }, {
          href: 'my-page/my-page.tablet.css',
          breakpoint: 'tablet'
        }, {
          href: 'my-page/my-page.desktop.css',
          breakpoint: 'desktop'
        }
      ]
    });

});
```


### Config

You can configure AngularCSS at the global level or at the stylesheet level.

#### Configuring global options

These options are applied during the `config` phase of your app via `$cssProvider`.

``` js
myApp.config(function($cssProvider) {

  angular.extend($cssProvider.defaults, {
    container: 'head',
    method: 'append',
    persist: false,
    preload: false,
    bustCache: false
  });

});
```

#### Configuring CSS options

These options are applied at the stylesheet level.

``` js
css: {
  href: 'file-path.css',
  rel: 'stylesheet',
  type: 'text/css',
  media: false,
  persist: false,
  preload: false,
  bustCache: false,
  weight: 0
}
```

### Support

AngularCSS is fully supported by AngularJS 1.3+

There is partial support for AngularJS 1.2. It does not support `css` property via DDO (Directive Definition Object).
The workarond is to bind (or add) the CSS in the directive's controller or link function via `$css` service.

``` js
myApp.directive('myDirective', function () {
  return {
    restrict: 'E',
    templateUrl: 'my-directive/my-directive.html',
    controller: function ($scope, $css) {
      $css.bind('my-directive/my-directive.css', $scope);
    }
  }
});
```


#### Browsers

Chrome, Firefox, Safari, iOS Safari, Android and IE9+

IE9 Does not support [matchMedia](http://caniuse.com/#feat=matchmedia) API. This means that in IE9, stylesheets with media queries will be added without checking if the breakpoint matches.


### Contributing

Please submit all pull requests the against master branch. If your pull request contains JavaScript patches or features, you should include relevant unit tests.

### Copyright and license

```
The MIT License

Copyright (c) 2014 DOOR3, Alex Castillo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
