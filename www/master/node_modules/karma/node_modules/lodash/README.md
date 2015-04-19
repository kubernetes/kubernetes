# Lo-Dash v1.1.1

A utility library delivering consistency, [customization](http://lodash.com/custom-builds), [performance](http://lodash.com/benchmarks), & [extras](http://lodash.com/#features).

## Download

* Lo-Dash builds (for modern environments):<br>
[Development](https://raw.github.com/lodash/lodash/1.1.1/dist/lodash.js) and
[Production](https://raw.github.com/lodash/lodash/1.1.1/dist/lodash.min.js)

* Lo-Dash compatibility builds (for legacy and modern environments):<br>
[Development](https://raw.github.com/lodash/lodash/1.1.1/dist/lodash.compat.js) and
[Production](https://raw.github.com/lodash/lodash/1.1.1/dist/lodash.compat.min.js)

* Underscore compatibility builds:<br>
[Development](https://raw.github.com/lodash/lodash/1.1.1/dist/lodash.underscore.js) and
[Production](https://raw.github.com/lodash/lodash/1.1.1/dist/lodash.underscore.min.js)

* CDN copies of ≤ v1.1.1’s builds are available on [cdnjs](http://cdnjs.com/) thanks to [CloudFlare](http://www.cloudflare.com/):<br>
[Lo-Dash dev](http://cdnjs.cloudflare.com/ajax/libs/lodash.js/1.1.1/lodash.js),
[Lo-Dash prod](http://cdnjs.cloudflare.com/ajax/libs/lodash.js/1.1.1/lodash.min.js),<br>
[Lo-Dash compat-dev](http://cdnjs.cloudflare.com/ajax/libs/lodash.js/1.1.1/lodash.compat.js),
[Lo-Dash compat-prod](http://cdnjs.cloudflare.com/ajax/libs/lodash.js/1.1.1/lodash.compat.min.js),<br>
[Underscore compat-dev](http://cdnjs.cloudflare.com/ajax/libs/lodash.js/1.1.1/lodash.underscore.js), and
[Underscore compat-prod](http://cdnjs.cloudflare.com/ajax/libs/lodash.js/1.1.1/lodash.underscore.min.js)

* For optimal file size, [create a custom build](http://lodash.com/custom-builds) with only the features you need

## Dive in

We’ve got [API docs](http://lodash.com/docs), [benchmarks](http://lodash.com/benchmarks), and [unit tests](http://lodash.com/tests).

For a list of upcoming features, check out our [roadmap](https://github.com/lodash/lodash/wiki/Roadmap).

## Resources

For more information check out these articles, screencasts, and other videos over Lo-Dash:

 * Posts
  - [Say “Hello” to Lo-Dash](http://kitcambridge.be/blog/say-hello-to-lo-dash/)

 * Videos
  - [Introducing Lo-Dash](https://vimeo.com/44154599)
  - [Lo-Dash optimizations and custom builds](https://vimeo.com/44154601)
  - [Lo-Dash’s origin and why it’s a better utility belt](https://vimeo.com/44154600)
  - [Unit testing in Lo-Dash](https://vimeo.com/45865290)
  - [Lo-Dash’s approach to native method use](https://vimeo.com/48576012)
  - [CascadiaJS: Lo-Dash for a better utility belt](http://www.youtube.com/watch?v=dpPy4f_SeEk)

## Features

 * AMD loader support ([RequireJS](http://requirejs.org/), [curl.js](https://github.com/cujojs/curl), etc.)
 * [_(…)](http://lodash.com/docs#_) supports intuitive chaining
 * [_.at](http://lodash.com/docs#at) for cherry-picking collection values
 * [_.bindKey](http://lodash.com/docs#bindKey) for binding [*“lazy”* defined](http://michaux.ca/articles/lazy-function-definition-pattern) methods
 * [_.cloneDeep](http://lodash.com/docs#cloneDeep) for deep cloning arrays and objects
 * [_.contains](http://lodash.com/docs#contains) accepts a `fromIndex` argument
 * [_.createCallback](http://lodash.com/docs#createCallback) to customize how callback arguments are handled and support callback shorthands in mixins
 * [_.findIndex](http://lodash.com/docs#findIndex) and [_.findKey](http://lodash.com/docs#findKey) for finding indexes and keys of collections
 * [_.forEach](http://lodash.com/docs#forEach) is chainable and supports exiting iteration early
 * [_.forIn](http://lodash.com/docs#forIn) for iterating over an object’s own and inherited properties
 * [_.forOwn](http://lodash.com/docs#forOwn) for iterating over an object’s own properties
 * [_.isPlainObject](http://lodash.com/docs#isPlainObject) checks if values are created by the `Object` constructor
 * [_.merge](http://lodash.com/docs#merge) for a deep [_.extend](http://lodash.com/docs#extend)
 * [_.parseInt](http://lodash.com/docs#parseInt) for consistent cross-environment behavior
 * [_.partial](http://lodash.com/docs#partial) and [_.partialRight](http://lodash.com/docs#partialRight) for partial application without `this` binding
 * [_.runInContext](http://lodash.com/docs#runInContext) for easier mocking and extended environment support
 * [_.support](http://lodash.com/docs#support) to flag environment features
 * [_.template](http://lodash.com/docs#template) supports [*“imports”* options](http://lodash.com/docs#templateSettings_imports), [ES6 template delimiters](http://people.mozilla.org/~jorendorff/es6-draft.html#sec-7.8.6), and [sourceURLs](http://www.html5rocks.com/en/tutorials/developertools/sourcemaps/#toc-sourceurl)
 * [_.where](http://lodash.com/docs#where) supports deep object comparisons
 * [_.clone](http://lodash.com/docs#clone), [_.omit](http://lodash.com/docs#omit), [_.pick](http://lodash.com/docs#pick),
   [and more…](http://lodash.com/docs "_.assign, _.cloneDeep, _.first, _.initial, _.isEqual, _.last, _.merge, _.rest") accept `callback` and `thisArg` arguments
 * [_.contains](http://lodash.com/docs#contains), [_.size](http://lodash.com/docs#size), [_.toArray](http://lodash.com/docs#toArray),
   [and more…](http://lodash.com/docs "_.at, _.countBy, _.every, _.filter, _.find, _.forEach, _.groupBy, _.invoke, _.map, _.max, _.min, _.pluck, _.reduce, _.reduceRight, _.reject, _.shuffle, _.some, _.sortBy, _.where") accept strings
 * [_.filter](http://lodash.com/docs#filter), [_.find](http://lodash.com/docs#find), [_.map](http://lodash.com/docs#map),
   [and more…](http://lodash.com/docs "_.countBy, _.every, _.first, _.groupBy, _.initial, _.last, _.max, _.min, _.reject, _.rest, _.some, _.sortBy, _.sortedIndex, _.uniq") support *“_.pluck”* and *“_.where”* `callback` shorthands

## Support

Lo-Dash has been tested in at least Chrome 5~25, Firefox 2~19, IE 6-10, Opera 9.25-12, Safari 3-6, Node.js 0.4.8-0.10.1, Narwhal 0.3.2, PhantomJS 1.8.1, RingoJS 0.9, and Rhino 1.7RC5.

## Installation and usage

In browsers:

```html
<script src="lodash.js"></script>
```

Using [`npm`](http://npmjs.org/):

```bash
npm install lodash

npm install -g lodash
npm link lodash
```

To avoid potential issues, update `npm` before installing Lo-Dash:

```bash
npm install npm -g
```

In [Node.js](http://nodejs.org/) and [RingoJS ≥ v0.8.0](http://ringojs.org/):

```js
var _ = require('lodash');

// or as a drop-in replacement for Underscore
var _ = require('lodash/dist/lodash.underscore');
```

**Note:** If Lo-Dash is installed globally, run [`npm link lodash`](http://blog.nodejs.org/2011/03/23/npm-1-0-global-vs-local-installation/) in your project’s root directory before requiring it.

In [RingoJS ≤ v0.7.0](http://ringojs.org/):

```js
var _ = require('lodash')._;
```

In [Rhino](http://www.mozilla.org/rhino/):

```js
load('lodash.js');
```

In an AMD loader like [RequireJS](http://requirejs.org/):

```js
require({
  'paths': {
    'underscore': 'path/to/lodash'
  }
},
['underscore'], function(_) {
  console.log(_.VERSION);
});
```

## Release Notes

### <sup>v1.1.1</sup>

 * Ensured the `underscore` build version of `_.forEach` accepts a `thisArg` argument
 * Updated vendor/tar to work with Node v0.10.x

### <sup>v1.1.0</sup>

 * Added `rhino -require` support
 * Added `_.createCallback`, `_findIndex`, `_.findKey`, `_.parseInt`, `_.runInContext`, and `_.support`
 * Added support for `callback` and `thisArg` arguments to `_.flatten`
 * Added CommonJS/Node support to precompiled templates
 * Ensured the `exports` object is not a DOM element
 * Ensured `_.isPlainObject` returns `false` for objects without a `[[Class]]` of “Object”
 * Made `_.cloneDeep`’s `callback` support more closely follow its documentation
 * Made the template precompiler create nonexistent directories of `--output` paths
 * Made `_.object` an alias of `_.zipObject`
 * Optimized method chaining, object iteration, `_.find`, and `_.pluck` (an average of 18% overall better performance)
 * Updated `backbone` build Lo-Dash method dependencies

The full changelog is available [here](https://github.com/lodash/lodash/wiki/Changelog).

## BestieJS

Lo-Dash is part of the [BestieJS](https://github.com/bestiejs)  *“Best in Class”* module collection. This means we promote solid browser/environment support, ES5+ precedents, unit testing, and plenty of documentation.

## Author

| [![twitter/jdalton](http://gravatar.com/avatar/299a3d891ff1920b69c364d061007043?s=70)](http://twitter.com/jdalton "Follow @jdalton on Twitter") |
|---|
| [John-David Dalton](http://allyoucanleet.com/) |

## Contributors

| [![twitter/blainebublitz](http://gravatar.com/avatar/ac1c67fd906c9fecd823ce302283b4c1?s=70)](http://twitter.com/blainebublitz "Follow @BlaineBublitz on Twitter") | [![twitter/kitcambridge](http://gravatar.com/avatar/6662a1d02f351b5ef2f8b4d815804661?s=70)](https://twitter.com/kitcambridge "Follow @kitcambridge on Twitter") | [![twitter/mathias](http://gravatar.com/avatar/24e08a9ea84deb17ae121074d0f17125?s=70)](http://twitter.com/mathias "Follow @mathias on Twitter") |
|---|---|---|
| [Blaine Bublitz](http://iceddev.com/) | [Kit Cambridge](http://kitcambridge.github.io/) | [Mathias Bynens](http://mathiasbynens.be/) |
