shell-escape
============

Escape and stringify an array of arguments to be executed on the shell

Install
-------

    npm install any-shell-escape

Example
-------

### simple

``` js
var shellescape = require('any-shell-escape');

var args = ['curl', '-v', '-H', 'Location;', '-H', "User-Agent: FooBar's so-called \"Browser\"", 'http://www.daveeddy.com/?name=dave&age=24'];

var escaped = shellescape(args);
console.log(escaped);
```

yields (on POSIX shells):

```
curl -v -H 'Location;' -H 'User-Agent: FoorBar'"'"'s so-called "Browser"' 'http://www.daveeddy.com/?name=dave&age=24'
```

or (on Windows):

```
curl -v -H "Location;" -H "User-Agent: FooBar's so-called ""Browser""" "http://www.daveeddy.com/?name=dave&age=24"
```

Which is suitable for being executed by the shell.

### Advanced Usage:

``` js
var shellescape = require('shell-escape');

var args = ['hello!', 'how are you doing $USER', '"double"', "'single'"];

var escaped = 'echo ' + shellescape.msg(args);
console.log(escaped);
```

yields (on POSIX shells):

```
echo 'hello!' 'how are you doing $USER' '"double"' "'"'single'"'"
```

or (on Windows, which doesn't support escaping echoed messages):

```
echo hello! how are you doing $USER "double" 'single'
```

and when run on the shell:

```
$ echo 'hello!' 'how are you doing $USER' '"double"' "'"'single'"'"
hello! how are you doing $USER "double" 'single'
```

License
-------

MIT
