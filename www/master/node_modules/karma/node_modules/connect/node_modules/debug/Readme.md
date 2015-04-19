# debug

  tiny node.js debugging utility modelled after node core's debugging technique.

## Installation

```bash
$ npm install debug
```

## Usage

 With `debug` you simply invoke the exported function to generate your debug function, passing it a name which will determine if a noop function is returned, or a decorated `console.error`, so all of the `console` format string goodies you're used to work fine. A unique color is selected per-function for visibility.

Example _app.js_:

```js
var debug = require('debug')('http')
  , http = require('http')
  , name = 'My App';

// fake app

debug('booting %s', name);

http.createServer(function(req, res){
  debug(req.method + ' ' + req.url);
  res.end('hello\n');
}).listen(3000, function(){
  debug('listening');
});

// fake worker of some kind

require('./worker');
```

Example _worker.js_:

```js
var debug = require('debug')('worker');

setInterval(function(){
  debug('doing some work');
}, 1000);
```

 The __DEBUG__ environment variable is then used to enable these based on space or comma-delimited names. Here are some examples:

  ![debug http and worker](http://f.cl.ly/items/18471z1H402O24072r1J/Screenshot.png)

  ![debug worker](http://f.cl.ly/items/1X413v1a3M0d3C2c1E0i/Screenshot.png)

#### Windows note

 On Windows the environment variable is set using the `set` command. 
 
 ```cmd
 set DEBUG=*,-not_this
 ```

Then, run the program to be debugged as usual.

## Millisecond diff

  When actively developing an application it can be useful to see when the time spent between one `debug()` call and the next. Suppose for example you invoke `debug()` before requesting a resource, and after as well, the "+NNNms" will show you how much time was spent between calls.

  ![](http://f.cl.ly/items/2i3h1d3t121M2Z1A3Q0N/Screenshot.png)

  When stdout is not a TTY, `Date#toUTCString()` is used, making it more useful for logging the debug information as shown below:

  ![](http://f.cl.ly/items/112H3i0e0o0P0a2Q2r11/Screenshot.png)

## Conventions

 If you're using this in one or more of your libraries, you _should_ use the name of your library so that developers may toggle debugging as desired without guessing names. If you have more than one debuggers you _should_ prefix them with your library name and use ":" to separate features. For example "bodyParser" from Connect would then be "connect:bodyParser".

## Wildcards

  The `*` character may be used as a wildcard. Suppose for example your library has debuggers named "connect:bodyParser", "connect:compress", "connect:session", instead of listing all three with `DEBUG=connect:bodyParser,connect.compress,connect:session`, you may simply do `DEBUG=connect:*`, or to run everything using this module simply use `DEBUG=*`.

  You can also exclude specific debuggers by prefixing them with a "-" character.  For example, `DEBUG=*,-connect:*` would include all debuggers except those starting with "connect:".

## Browser support

  Debug works in the browser as well, currently persisted by `localStorage`. Consider the situation shown below where you have `worker:a` and `worker:b`, and wish to debug both. Somewhere in the code on your page, include:

```js
window.myDebug = require("debug");
```

  ("debug" is a global object in the browser so we give this object a different name.) When your page is open in the browser, type the following in the console:

```js
myDebug.enable("worker:*")
```

  Refresh the page. Debug output will continue to be sent to the console until it is disabled by typing `myDebug.disable()` in the console.

```js
a = debug('worker:a');
b = debug('worker:b');

setInterval(function(){
  a('doing some work');
}, 1000);

setInterval(function(){
  b('doing some work');
}, 1200);
```

#### Web Inspector Colors

  Colors are also enabled on "Web Inspectors" that understand the `%c` formatting
  option. These are WebKit web inspectors, Firefox ([since version
  31](https://hacks.mozilla.org/2014/05/editable-box-model-multiple-selection-sublime-text-keys-much-more-firefox-developer-tools-episode-31/))
  and the Firebug plugin for Firefox (any version).

  Colored output looks something like:

  ![](https://cloud.githubusercontent.com/assets/71256/3139768/b98c5fd8-e8ef-11e3-862a-f7253b6f47c6.png)

### stderr vs stdout

You can set an alternative logging method per-namespace by overriding the `log` method on a per-namespace or globally:

Example _stdout.js_:

```js
var debug = require('debug');
var error = debug('app:error');

// by default stderr is used
error('goes to stderr!');

var log = debug('app:log');
// set this namespace to log via console.log
log.log = console.log.bind(console); // don't forget to bind to console!
log('goes to stdout');
error('still goes to stderr!');

// set all output to go via console.info
// overrides all per-namespace log settings
debug.log = console.info.bind(console);
error('now goes to stdout via console.info');
log('still goes to stdout, but via console.info now');
```

## Authors

 - TJ Holowaychuk
 - Nathan Rajlich

## License

(The MIT License)

Copyright (c) 2014 TJ Holowaychuk &lt;tj@vision-media.ca&gt;

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
