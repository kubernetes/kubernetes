# ng-annotate
ng-annotate adds and removes AngularJS dependency injection annotations.
It is non-intrusive so your source code stays exactly the same otherwise.
No lost comments or moved lines. Annotations are useful because with them
you're able to minify your source code using your favorite JS minifier.

You write your code without annotations, like this:

```js
angular.module("MyMod").controller("MyCtrl", function($scope, $timeout) {
});
```

You then run ng-annotate as a build-step to produce this intermediary,
annotated, result (later sent to the minifier):

```js
angular.module("MyMod").controller("MyCtrl", ["$scope", "$timeout", function($scope, $timeout) {
}]);
```

You can also use ng-annotate to rebuild or remove existing annotations.
Rebuilding is useful if you like to check-in the annotated version of your
source code. When refactoring, just change parameter names once and let
ng-annotate rebuild the annotations. Removing is useful if you want to
de-annotate an existing codebase that came with checked-in annotations

*ng-annotate works by using static analysis to identify common code patterns.
There are patterns it does not and never will understand and for those you
can use an explicit `ngInject` annotation instead, see section further down.


## Installation and usage

```bash
npm install -g ng-annotate
```

Then run it as `ng-annotate OPTIONS <file>`. The errors (if any) will go to stderr,
the transpiled output to stdout.

Use the `--add` (`-a`) option to add annotations where non-existing,
use `--remove` (`-r`) to remove all existing annotations,
use `--add --remove` (`-ar`) to rebuild all annotations.

Use the `-o` option to write output to file.

Provide `-` instead of an input `<file>` to read input from stdin.

Use the `--sourcemap` option to generate an inline sourcemap.

Use the `--sourceroot` option to set the sourceRoot property of the generated sourcemap.

Use the `--single_quotes` option to output `'$scope'` instead of `"$scope"`.

Use the `--regexp` option to restrict matching further or to expand matching.
See description further down.

*experimental* Use the `--rename` option to rename providers (services, factories,
controllers, etc.) with a new name when declared and referenced through annotation.
Use it like this: `--rename oldname1 newname1 oldname2 newname2`

*experimental* Use the `--plugin` option to load a user plugin with the provided path,
0.9.x may change API). See [plugin-example.js](plugin-example.js) for more info.

*experimental* Use the `--stats` option to print statistics on stderr.


## Highly recommended: enable ng-strict-di in your minified builds
`<div ng-app="myApp" ng-strict-di>`

Do that in your ng-annotate processed builds and AngularJS will let you know if there are
any missing dependency injection annotations. [ng-strict-di](https://docs.angularjs.org/api/ng/directive/ngApp)
is available in AngularJS 1.3 or later.


## Tools support
* [Grunt](http://gruntjs.com/): [grunt-ng-annotate](https://www.npmjs.org/package/grunt-ng-annotate) by [Michał Gołębiowski](https://github.com/mzgol)
* [Browserify](http://browserify.org/): [browserify-ngannotate](https://www.npmjs.org/package/browserify-ngannotate) by [Owen Smith](https://github.com/omsmith)
* [Brunch](http://brunch.io/): [ng-annotate-uglify-js-brunch](https://www.npmjs.org/package/ng-annotate-uglify-js-brunch) by [Kagami Hiiragi](https://github.com/Kagami)
* [Gulp](http://gulpjs.com/): [gulp-ng-annotate](https://www.npmjs.org/package/gulp-ng-annotate/) by [Kagami Hiiragi](https://github.com/Kagami)
* [Broccoli](https://github.com/broccolijs/broccoli): [broccoli-ng-annotate](https://www.npmjs.org/package/broccoli-ng-annotate) by [Gilad Peleg](https://github.com/pgilad)
* [Rails asset pipeline](http://guides.rubyonrails.org/asset_pipeline.html): [ngannotate-rails](https://rubygems.org/gems/ngannotate-rails) by [Kari Ikonen](https://github.com/kikonen)
* [Grails asset pipeline](https://github.com/bertramdev/asset-pipeline): [angular-annotate-asset-pipeline](https://github.com/craigburke/angular-annotate-asset-pipeline) by [Craig Burke](https://github.com/craigburke)
* [Webpack](http://webpack.github.io/): [ng-annotate-webpack-plugin](https://www.npmjs.org/package/ng-annotate-webpack-plugin) by [Chris Liechty](https://github.com/cliechty)
* [Middleman](http://middlemanapp.com/): [middleman-ngannotate](http://rubygems.org/gems/middleman-ngannotate) by [Michael Siebert](https://github.com/siebertm)
* Something missing? Contributions welcome - create plugin and submit a README pull request!


## Changes
See [CHANGES.md](CHANGES.md).


## Declaration forms
ng-annotate understands the two common declaration forms:

Long form:

```js
angular.module("MyMod").controller("MyCtrl", function($scope, $timeout) {
});
```

Short form:

```js
myMod.controller("MyCtrl", function($scope, $timeout) {
});
```

It's not limited to `.controller` of course. It understands `.config`, `.factory`,
`.directive`, `.filter`, `.run`, `.controller`, `.provider`, `.service`, `.animation` and
`.invoke`.

For short forms it does not need to see the declaration of `myMod` so you can run it
on your individual source files without concatenating. If ng-annotate detects a short form
false positive then you can use the `--regexp` option to limit the module identifier.
Examples: `--regexp "^myMod$"` (match only `myMod`) or `--regexp "^$"` (ignore short forms).
You can also use `--regexp` to opt-in for more advanced method callee matching, for
example `--regexp "^require(.*)$"` to detect and transform
`require('app-module').controller(..)`. Not using the option is the same as passing
`--regexp "^[a-zA-Z0-9_\$\.\s]+$"`, which means that the callee can be a (non-unicode)
identifier (`foo`), possibly with dot notation (`foo.bar`).

ng-annotate understands `angular.module("MyMod", function(dep) ..)` as an alternative to
`angular.module("MyMod").config(function(dep) ..)`.

ng-annotate understands `this.$get = function($scope) ..` and
`{.., $get: function($scope) ..}` inside a `provider`. `self` and `that` can be used as
aliases for `this`.

ng-annotate understands `return {.., controller: function($scope) ..}` inside a
`directive`.

ng-annotate understands `$provide.decorator("bar", function($scope) ..)`, `$provide.service`,
`$provide.factory` and `$provide.provider`.

ng-annotate understands `$routeProvider.when("path", { .. })`.

ng-annotate understands `$httpProvider.interceptors.push(function($scope) ..)` and
`$httpProvider.responseInterceptors.push(function($scope) ..)`.

ng-annotate understands `$injector.invoke(function ..)`.

ng-annotate understands [ui-router](https://github.com/angular-ui/ui-router) (`$stateProvider` and
`$urlRouterProvider`).

ng-annotate understands `$modal.open` ([angular-ui/bootstrap](http://angular-ui.github.io/bootstrap/)).

ng-annotate understands `$mdDialog.show`, `$mdToast.show` and `$mdBottomSheet.show`
([angular material design](https://material.angularjs.org/#/api/material.components.dialog/service/$mdDialog)).

ng-annotate understands chaining.

ng-annotate understands IIFE's and attempts to match through them, so
`(function() { return function($scope) .. })()` works anywhere
`function($scope) ..` does (for any IIFE args and params).


## Reference-following
ng-annotate follows references. This works iff the referenced declaration is
a) a function declaration or
b) a variable declaration with an initializer.
Modifications to a reference outside of its declaration site are ignored by ng-annotate.

These examples will get annotated:

```js
function MyCtrl($scope, $timeout) {
}
var MyCtrl2 = function($scope) {};

angular.module("MyMod").controller("MyCtrl", MyCtrl);
angular.module("MyMod").controller("MyCtrl", MyCtrl2);
```


## Explicit annotations with ngInject
You can prepend a function with `/*@ngInject*/` to explicitly state that the function
should get annotated. ng-annotate will leave the comment intact and will thus still
be able to also remove or rewrite such annotations.

You can also wrap an expression inside an `ngInject(..)` function call. If you use this
syntax then add `function ngInject(v) { return v }` somewhere in your codebase, or process
away the `ngInject` function call in your build step.

You can also add the `"ngInject"` directive prologue at the beginning of a function,
similar to how `"use strict"` is used, to state that the surrounding function should get
annotated.

Use `ngInject` to support your code style when it's not in a form ng-annotate understands
natively. Remember that the intention of ng-annotate is to reduce stuttering for you,
and `ngInject` does this just as well. You don't need to keep two lists in sync. Use it!

`ngInject` may be particularly useful if you use a compile-to-JS language that doesn't
preserve comments.


### Suppressing false positives with ngNoInject
The `/*@ngInject*/`, `ngInject(..)` and `"ngInject"` siblings have three cousins that
are used for the opposite purpose, suppressing an annotation that ng-annotate added
incorrectly (a "false positive"). They are called `/*@ngNoInject*/`, `ngNoInject(..)`
and `"ngNoInject"` and do exactly what you think they do.


### ngInject examples
Here follows some ngInject examples using the `/*@ngInject*/` syntax. Most examples
works fine using the `ngInject(..)` or `"ngInject"` syntax as well.

```js
x = /*@ngInject*/ function($scope) {};
obj = {controller: /*@ngInject*/ function($scope) {}};
obj.bar = /*@ngInject*/ function($scope) {};

=>

x = /*@ngInject*/ ["$scope", function($scope) {}];
obj = {controller: /*@ngInject*/ ["$scope", function($scope) {}]};
obj.bar = /*@ngInject*/ ["$scope", function($scope) {}];
```

Prepended to an object literal, `/*@ngInject*/` will annotate all of its contained
function expressions, recursively:

```js
obj = /*@ngInject*/ {
    controller: function($scope) {},
    resolve: { data: function(Service) {} },
};

=>

obj = /*@ngInject*/ {
    controller: ["$scope", function($scope) {}],
    resolve: { data: ["Service", function(Service) {}] },
};
```

Prepended to a function statement, to a single variable declaration initialized with a
function expression or to an assignment where the rvalue is a function expression,
 `/*@ngInject*/` will attach an `$inject` array to the function:

```js
// @ngInject
function Foo($scope) {}

// @ngInject
var foo = function($scope) {}

// @ngInject
module.exports = function($scope) {}

=>

// @ngInject
function Foo($scope) {}
Foo.$inject = ["$scope"];

// @ngInject
var foo = function($scope) {}
foo.$inject = ["$scope"];

// @ngInject
module.exports = function($scope) {}
module.exports.$inject = ["$scope"];
```


## Build and test
ng-annotate is written in ES6 constlet style and uses [defs.js](https://github.com/olov/defs)
to transpile to ES5. See [BUILD.md](BUILD.md) for build and test instructions.


## License
`MIT`, see [LICENSE](LICENSE) file.

ng-annotate is written by [Olov Lassus](https://github.com/olov) with the kind help by
[contributors](https://github.com/olov/ng-annotate/graphs/contributors).
[Follow @olov](https://twitter.com/olov) on Twitter for updates about ng-annotate.


## How does ng-annotate compare to ngmin?
ngmin has been deprecated in favor of ng-annotate. In short:
ng-annotate is much faster, finds more declarations to annotate (including ui-router),
treats your source code better, is actively maintained and has a bunch of extra features
on top of that. A much more elaborated answer can be found in
["The future of ngmin and ng-annotate"](https://github.com/btford/ngmin/issues/93).

*Migrating from ngmin*:
`ng-annotate -a -` is similar to `ngmin` (use stdin and
stdout). `ng-annotate -a in.js -o out.js` is similar to `ngmin in.js out.js`. Grunt users
can migrate easily by installing
[grunt-ng-annotate](https://www.npmjs.org/package/grunt-ng-annotate) and replacing `ngmin`
with `ngAnnotate` in their Gruntfile. Scroll down for information about other tools.


## Library (API)
ng-annotate can be used as a library. See [ng-annotate.js](ng-annotate.js) for further info about
options and return value.

```js
var ngAnnotate = require("ng-annotate");
var somePlugin = require("./some/path/some-plugin");
var res = ngAnnotate(src, {
    add: true,
    plugin: [somePlugin],
    rename: [{from: "generalname", to: "uniquename"}, {from: "alpha", to: "beta"}],
    sourcemap: { inline: false, inFile: "source.js", sourceRoot: "/path/to/source/root" }
});
var errorstringArray = res.errors;
var transformedSource = res.src;
var transformedSourceMap = res.map;
```
