# redeyed [![build status](https://secure.travis-ci.org/thlorenz/redeyed.png?branch=master)](http://travis-ci.org/thlorenz/redeyed)

*Add color to your JavaScript!*

![frog](http://allaboutfrogs.org/gallery/photos/redeyes/red1.gif)

[Red Eyed Tree Frog](http://allaboutfrogs.org/info/species/redeye.html) *(Agalychnis callidryas)*

## What?

Takes JavaScript code, along with a config and returns the original code with tokens wrapped and/or replaced as configured.

## Where?

- server side using nodejs
- in the [browser](#browser-support)

## What for?

One usecase is adding metadata to your code that can then be used to apply syntax highlighting.

## How?

- copy the [config.js](https://github.com/thlorenz/redeyed/blob/master/config.js) and edit it in order to specify how
  certain tokens are to be surrounded/replaced
- replace the `undefined` of each token you want to configure with one of the following

### {String} config

`'before:after'`

wraps the token inside before/after 

### {Object} config

`{ _before: 'before', _after: 'after' }`

wraps token inside before/after

#### Missing before and after resolution for {String} and {Object} config

For the `{String}` and `{Object}` configurations, 'before' or 'after' may be omitted:

- `{String}`: 
  - `'before:'` (omitting 'after')
  - `':after'` (omitting 'before')
- `{Object}`: 
  - `{ _before: 'before' }` (omitting '_after')
  - `{ _after: 'after' }` (omitting '_before')

In these cases the missing half is resolved as follows:

- from the `parent._default` (i.e., `Keyword._default`) if found
- otherwise from the `config._default` if found
- otherwise `''` (empty string)

### {Function} config

`function (tokenString, info) { return {String}|{Object}; }`

#### Inputs

- tokenString: the content of the token that is currently being processed
- info: an object with the following structure

```js
{
    // {Int}
    // the index of the token being processed inside tokens
    tokenIndex

    // {Array}
    // all tokens that are being processed including comments 
    // (i.e. the result of merging esprima tokens and comments)
  , tokens  

    // {Object} 
    // the abstract syntax tree of the parsed code
  , ast  

    // {String}
    // the code that was parsed (same string as the one passed to redeyed(code ..)
  , code
}
```

In most cases the `tokenString` is all you need. The extra info object is passed in case you need to gather more
information about the `token`'s surroundings in order to decide how to transform it. 
See: [replace-log-example](https://github.com/thlorenz/redeyed/blob/master/examples/replace-log.js)

#### Output

You can return a {String} or an {Object} from a {Function} config.

- when returning a {String}, the token value will be replaced with it
- when returning an {Object}, it should be of the following form:

```js
{
    // {String}
    // the string that should be substituted for the value of the current and all skipped tokens
    replacement

    // {Object} (Token)
    // the token after which processing should continue
    // all tokens in between the current one and this one inclusive will be ignored
  , skipPastToken
}
```

### Transforming JavaScript code

***redeyed(code, config[, opts])***

Invoke redeyed with your **config**uration, a **code** snippet and maybe **opts** as in the below example:

```javascript
var redeyed = require('redeyed')
  , config = require('./path/to/config')
  , code = 'var a = 3;'
  , result;

// redeyed will throw an error (caused by the esprima parser) if the code has invalid javascript
try {
  result = redeyed(code, config);
  console.log(result.code);
} catch(err) {
  console.error(err);
}
```

***opts***:
```js
{ // {Boolean}
  // if true `result.code` is not assigned and therefore `undefined`
  // if false (default) `result.code` property contains the result of `split.join`
  nojoin: true|false
}
```

***return value***:

```js
{   ast      
  , tokens   
  , comments 
  , splits   
  , code     
}
```

- ast `{Array}`: [abstract syntax tree](http://en.wikipedia.org/wiki/Abstract_syntax_tree) as returned by [esprima
  parse](http://en.wikipedia.org/wiki/Abstract_syntax_tree)
- tokens `{Array}`: [tokens](http://en.wikipedia.org/wiki/Token_(parser)) provided by esprima (excluding
  comments)
- comments `{Array}`: block and line comments as provided by esprima
- splits `{Array}`: code pieces split up, some of which where transformed as configured
- code `{String}`: transformed code, same as `splits.join('')` unless this step has been skipped (see opts)

## Browser Support

### AMD

Ensure to include [esprima](https://github.com/ariya/esprima) as one of your dependencies

```js
define(['redeyed'], function (redeyed) {
 [ .. ]
});
```

### Attached to global window object

The `redeyed {Function}` will be exposed globally as `window.redeyed` - big surprise!

```html
<script type="text/javascript" src="https://raw.github.com/ariya/esprima/master/esprima.js"></script>
<script type="text/javascript" src="path/to/redeyed.js"></script>
```

## redeyed in the wild

- [cardinal](https://github.com/thlorenz/cardinal): Syntax highlights JavaScript code with ANSI colors to be printed to
  the terminal
- [peacock](http://thlorenz.github.com/peacock/): JavaScript syntax highlighter that generates html that is compatible
  with pygments styles.

## Examples

- `npm explore redeyed; npm demo` will let you try the [browser example](https://github.com/thlorenz/redeyed/tree/master/examples/browser)
- `npm explore redeyed; npm demo-log` will let you try the [replace log example](https://github.com/thlorenz/redeyed/blob/master/examples/replace-log.js)

