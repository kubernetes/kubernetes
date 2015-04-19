# meow [![Build Status](https://travis-ci.org/sindresorhus/meow.svg?branch=master)](https://travis-ci.org/sindresorhus/meow)

> CLI app helper

![](meow.gif)


## Features

- Parses arguments using [minimist](https://github.com/substack/minimist)
- Converts flags to [camelCase](https://github.com/sindresorhus/camelcase)
- Outputs version when `--version`
- Outputs description and supplied help text when `--help`


## Install

```
$ npm install --save meow
```


## Usage

```sh
$ ./foo-app.js unicorns --rainbow-cake
```

```js
#!/usr/bin/env node
'use strict';
var meow = require('meow');
var fooApp = require('./');

var cli = meow({
	help: [
		'Usage',
		'  foo-app <input>'
	].join('\n')
});
/*
{
	input: ['unicorns'],
	flags: {rainbowCake: true},
	...
}
*/

fooApp(cli.input[0], cli.flags);
```


## API

### meow(options, minimistOptions)

Returns an object with:

- `input` *(array)* - Non-flag arguments
- `flags` *(object)* - Flags converted to camelCase
- `pkg` *(object)* - The `package.json` object
- `help` *(object)* - The help text used with `--help`
- `showHelp()` *(function)* - Show the help text and exit

#### options

##### help

Type: `string`, `boolean`

The help text you want shown.

If you don't specify anything, it will still show the package.json `"description"`.

Set it to `false` to disable it all together.

##### version

Type: `string`, `boolean`  
Default: the package.json `"version"` property

Set a custom version output.

Set it to `false` to disable it all together.

##### pkg

Type: `string`, `object`  
Default: `package.json`

Relative path to `package.json` or it as an object.

##### argv

Type: `array`  
Default: `process.argv.slice(2)`

Custom arguments object.

#### minimistOptions

Type: `object`  
Default: `{}`

Minimist [options](https://github.com/substack/minimist#var-argv--parseargsargs-opts).


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
