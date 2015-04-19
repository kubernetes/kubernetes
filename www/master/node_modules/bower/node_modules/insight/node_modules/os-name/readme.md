# os-name [![Build Status](https://travis-ci.org/sindresorhus/os-name.svg?branch=master)](https://travis-ci.org/sindresorhus/os-name)

> Get the name of the current operating system. Example: `OS X Mavericks`

Useful for analytics and debugging.


## Install

```sh
$ npm install --save os-name
```


## Usage

```js
var os = require('os');
var osName = require('os-name');

// on an OS X Mavericks system

osName();
//=> OS X Mavericks

osName(os.platform(), os.release());
//=> OS X Mavericks

osName(os.platform());
//=> OS X

osName('linux', '3.13.0-24-generic');
//=> Linux 3.13

osName('win32', '6.3.9600');
//=> Windows 8.1

osName('win32');
// Windows
```


## API

### osName([platform, release])

By default the name of the current operating system is returned.

You can optionally supply a custom [`os.platform()`](http://nodejs.org/api/os.html#os_os_platform) and [`os.release()`](http://nodejs.org/api/os.html#os_os_release).

Check out [getos](https://github.com/wblankenship/getos) if you need the Linux distribution name.


## CLI

```sh
$ npm install --global os-name
```

```sh
$ os-name --help

  Example
    os-name
    OS X Mavericks
```


## Contributing

Production systems depend on this package for logging / tracking. Please be careful when introducing new output, and adhere to existing output format (whitespace, capitalization, etc.).


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
