# osx-release [![Build Status](https://travis-ci.org/sindresorhus/osx-release.svg?branch=master)](https://travis-ci.org/sindresorhus/osx-release)

> Get the name and version of a OS X release from the Darwin version.  
> Example: `13.2.0` → `{name: 'Mavericks', version: '10.9'}`


## Install

```sh
$ npm install --save osx-release
```


## Usage

```js
var os = require('os');
var osxRelease = require('osx-release');

// on an OS X Mavericks system

osxRelease();
//=> {name: 'Mavericks', version: '10.9'}

os.release();
//=> 13.2.0
// this is the Darwin kernel version

osxRelease(os.release());
//=> {name: 'Mavericks', version: '10.9'}

osxRelease('14.0.0');
//=> {name: 'Yosemite', version: '10.10'}
```


## API

### osRelease([release])

#### release

Type: `string`

By default the current OS is used, but you can supply a custom [Darwin kernel version](http://en.wikipedia.org/wiki/Darwin_%28operating_system%29#Release_history), which is the output of [`os.release()`](http://nodejs.org/api/os.html#os_os_release).


## CLI

```sh
$ npm install --global osx-release
```

```sh
$ osx-release --help

  Usage
    osx-release [release]

  Example
    osx-release
    Mavericks 10.9

    osx-release 14.0.0
    Yosemite 10.10
```


## Related

- [os-name](https://github.com/sindresorhus/os-name) - Get the name of the current operating system. Example: `OS X Mavericks`
- [osx-version](https://github.com/sindresorhus/osx-version) - Get the OS X version of the current system. Example: `10.9.3`


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
