# ansi-styles [![Build Status](https://travis-ci.org/sindresorhus/ansi-styles.svg?branch=master)](https://travis-ci.org/sindresorhus/ansi-styles)

> [ANSI escape codes](http://en.wikipedia.org/wiki/ANSI_escape_code#Colors_and_Styles) for styling strings in the terminal

You probably want the higher-level [chalk](https://github.com/sindresorhus/chalk) module for styling your strings.

![screenshot](screenshot.png)


## Install

```sh
$ npm install --save ansi-styles
```


## Usage

```js
var ansi = require('ansi-styles');

console.log(ansi.green.open + 'Hello world!' + ansi.green.close);
```


## API

Each style has an `open` and `close` property.


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
