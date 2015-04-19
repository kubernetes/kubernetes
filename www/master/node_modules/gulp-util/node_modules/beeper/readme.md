# beeper [![Build Status](https://travis-ci.org/sindresorhus/beeper.svg?branch=master)](https://travis-ci.org/sindresorhus/beeper)

> Make your terminal beep

Useful as an attention grabber e.g. when an error happens.


## Install

```sh
$ npm install --save beeper
```


## Usage

```js
var beeper = require('beeper');

beeper();
// beep one time

beeper(3);
// beep three times

beeper('****-*-*');
// beep, beep, beep, beep, pause, beep, pause, beep
```


## API

It will not beep if stdout is not TTY or if the user supplies the `--no-beep` flag.

### beeper([count|melody])

#### count

Type: `number`  
Default: `1`

How many times you want it to beep.

#### melody

Type: `string`

Construct your own melody by supplying a string of `*` for beep `-` for pause.


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
