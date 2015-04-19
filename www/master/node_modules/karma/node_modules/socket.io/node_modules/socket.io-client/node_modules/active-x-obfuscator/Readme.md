# active-x-obfuscator

A module to (safely) obfuscate all occurrences of the string 'ActiveX' inside
any JavaScript code.

## Why?

Some corporate firewalls /proxies such as Blue Coat block JavaScript files to be
downloaded if they contain the word `'ActiveX'`. That of course is very annoying
for libraries such as [socket.io][] that need to use `ActiveXObject` for
supporting IE8 and older.

## Install

```
npm install active-x-obfuscator
```

## Usage

```js
var activeXObfuscator = require('active-x-obfuscator');
var code = 'foo(new ActiveXObject());';

var obfuscated = activeXObfuscator(code);
// -> foo(new window[(['Active'].concat('Object').join('X'))])
```

## License

Licensed under the MIT license.

[socket.io]: http://socket.io/
