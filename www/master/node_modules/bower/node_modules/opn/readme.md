# opn

> A better [node-open](https://github.com/pwnall/node-open). Opens stuff like websites, files, executables. Cross-platform.


#### Why?

- Actively maintained
- Includes the latest [xdg-open script](http://portland.freedesktop.org/download/)
- Fixes most of the `node-open` issues


## Install

```sh
$ npm install --save opn
```


## Usage

```js
var opn = require('opn');

opn('http://sindresorhus.com');
// opens that url in the default browser

opn('http://sindresorhus.com', 'firefox');
// you can also specify the app to open in

opn('unicorn.png');
// opens the image in the default image viewer
```


## API

Uses the command `open` on OS X, `start` on Windows and `xdg-open` on other platforms.

### opn(target, [app, callback])

#### target

*Required*
Type: `string`

The thing you want to open. Can be a url, file, or executable.

Opens in the default app for the file type. Eg. urls opens in your default browser.

#### app

Type: `string`

Specify the app to open the `target` with.

The app name is platform dependent. Don't hard code it in reusable modules.

#### callback(err)

Type: `function`

Executes when the opened app exits.

On Windows you have to explicitly specify an app for it to be able to wait.


## CLI

You can also use it as a CLI app by installing it globally:

```sh
$ npm install --global opn
```

```sh
$ opn --help

Usage
  $ opn <file|url> [app]

Example
  $ opn http://sindresorhus.com
  $ opn http://sindresorhus.com firefox
  $ opn unicorn.png
```


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
