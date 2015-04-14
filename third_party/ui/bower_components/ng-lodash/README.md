[![Built with Grunt](https://cdn.gruntjs.com/builtwith.png)](http://gruntjs.com/)
[![Build Status](https://travis-ci.org/rockabox/ng-lodash.svg?branch=master)](https://travis-ci.org/rockabox/ng-lodash)
[![devDependency Status](https://david-dm.org/rockabox/ng-lodash/dev-status.svg)](https://david-dm.org/rockabox/ng-lodash#info=devDependencies)

ng-lodash
=========

This is a wrapper for the utility library [Lo-Dash](http://lodash.com/) for
Angular JS. One aim for this project is to ensure Lo-Dash doesn't have to be
left on the window, and we use Lo-Dash with Angular, in the normal depenedency
 injection manner.

## Installing
Install via bower

```bower install ng-lodash```

Require it into your application (after Angular)

```<script src="ng-lodash.min.js"></script>```

Add the module as a dependency to your app

```js
var app = angular.module('yourAwesomeApp', ['ngLodash']);
```

And inject it into your controller like so!

```js
var YourCtrl = app.controller('yourController', function($scope, lodash) {
  lodash.assign({ 'a': 1 }, { 'b': 2 }, { 'c': 3 });
});
```

## Developing

To help us develop this module, we are using Grunt some tasks that may be
helpful for you to know about are:

### Testing

This command will run JSHint and JSCS testing JS Files (note files within build)
are not tested, it will also run your local build of the module with all of the
Karma tests:

```grunt test``` it can also be run by using ```npm test```

### Build

This command will build the module, run it through ngMin and then create a
minified version of the module, ready for distribution:

```grunt build```

### Dist

This command will build the module initially and then run the test suite.
Testing with JSHint, JSCS and Karma:

```grunt dist```
