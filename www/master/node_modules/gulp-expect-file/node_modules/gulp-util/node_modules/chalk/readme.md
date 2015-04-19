# <img width="300" src="https://cdn.rawgit.com/sindresorhus/chalk/77ae94f63ab1ac61389b190e5a59866569d1a376/logo.svg" alt="chalk">

> Terminal string styling done right

[![Build Status](https://travis-ci.org/sindresorhus/chalk.svg?branch=master)](https://travis-ci.org/sindresorhus/chalk)
![](http://img.shields.io/badge/unicorn-approved-ff69b4.svg)

[colors.js](https://github.com/Marak/colors.js) is currently the most popular string styling module, but it has serious deficiencies like extending String.prototype which causes all kinds of [problems](https://github.com/yeoman/yo/issues/68). Although there are other ones, they either do too much or not enough.

**Chalk is a clean and focused alternative.**

![screenshot](https://github.com/sindresorhus/ansi-styles/raw/master/screenshot.png)


## Why

- Highly performant
- Doesn't extend String.prototype
- Expressive API
- Ability to nest styles
- Clean and focused
- Auto-detects color support
- Actively maintained
- [Used by 1000+ modules](https://npmjs.org/browse/depended/chalk)


## Install

```sh
$ npm install --save chalk
```


## Usage

Chalk comes with an easy to use composable API where you just chain and nest the styles you want.

```js
var chalk = require('chalk');

// style a string
console.log(  chalk.blue('Hello world!')  );

// combine styled and normal strings
console.log(  chalk.blue('Hello'), 'World' + chalk.red('!')  );

// compose multiple styles using the chainable API
console.log(  chalk.blue.bgRed.bold('Hello world!')  );

// pass in multiple arguments
console.log(  chalk.blue('Hello', 'World!', 'Foo', 'bar', 'biz', 'baz')  );

// nest styles
console.log(  chalk.red('Hello', chalk.underline.bgBlue('world') + '!')  );

// nest styles of the same type even (color, underline, background)
console.log(  chalk.green('I am a green line ' + chalk.blue('with a blue substring') + ' that becomes green again!')  );
```

Easily define your own themes.

```js
var chalk = require('chalk');
var error = chalk.bold.red;
console.log(error('Error!'));
```

Take advantage of console.log [string substitution](http://nodejs.org/docs/latest/api/console.html#console_console_log_data).

```js
var name = 'Sindre';
console.log(chalk.green('Hello %s'), name);
//=> Hello Sindre
```


## API

### chalk.`<style>[.<style>...](string, [string...])`

Example: `chalk.red.bold.underline('Hello', 'world');`

Chain [styles](#styles) and call the last one as a method with a string argument. Order doesn't matter.

Multiple arguments will be separated by space.

### chalk.enabled

Color support is automatically detected, but you can override it.

### chalk.supportsColor

Detect whether the terminal [supports color](https://github.com/sindresorhus/supports-color).

Can be overridden by the user with the flags `--color` and `--no-color`.

Used internally and handled for you, but exposed for convenience.

### chalk.styles

Exposes the styles as [ANSI escape codes](https://github.com/sindresorhus/ansi-styles).

Generally not useful, but you might need just the `.open` or `.close` escape code if you're mixing externally styled strings with yours.

```js
var chalk = require('chalk');

console.log(chalk.styles.red);
//=> {open: '\u001b[31m', close: '\u001b[39m'}

console.log(chalk.styles.red.open + 'Hello' + chalk.styles.red.close);
```

### chalk.hasColor(string)

Check whether a string [has color](https://github.com/sindresorhus/has-ansi).

### chalk.stripColor(string)

[Strip color](https://github.com/sindresorhus/strip-ansi) from a string.

Can be useful in combination with `.supportsColor` to strip color on externally styled text when it's not supported.

Example:

```js
var chalk = require('chalk');
var styledString = getText();

if (!chalk.supportsColor) {
	styledString = chalk.stripColor(styledString);
}
```


## Styles

### General

- `reset`
- `bold`
- `dim`
- `italic` *(not widely supported)*
- `underline`
- `inverse`
- `hidden`
- `strikethrough` *(not widely supported)*

### Text colors

- `black`
- `red`
- `green`
- `yellow`
- `blue`
- `magenta`
- `cyan`
- `white`
- `gray`

### Background colors

- `bgBlack`
- `bgRed`
- `bgGreen`
- `bgYellow`
- `bgBlue`
- `bgMagenta`
- `bgCyan`
- `bgWhite`


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
