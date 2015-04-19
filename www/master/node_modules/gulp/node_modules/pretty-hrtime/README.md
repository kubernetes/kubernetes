[![Build Status](https://secure.travis-ci.org/robrich/pretty-hrtime.png?branch=master)](https://travis-ci.org/robrich/pretty-hrtime)
[![Dependency Status](https://david-dm.org/robrich/pretty-hrtime.png)](https://david-dm.org/robrich/pretty-hrtime)

pretty-hrtime
============

[process.hrtime()](http://nodejs.org/api/process.html#process_process_hrtime) to words

Usage
-----

```javascript
var prettyHrtime = require('pretty-hrtime');

var start = process.hrtime();
// do stuff
var end = process.hrtime(start);

var words = prettyHrtime(end);
console.log(words); // '1.2 ms'

words = prettyHrtime(end, {verbose:true});
console.log(words); // '1 millisecond 209 microseconds'

words = prettyHrtime(end, {precise:true});
console.log(words); // '1.20958 ms'
```

Note: process.hrtime() has been available since 0.7.6.
See [http://nodejs.org/changelog.html](http://nodejs.org/changelog.html)
and [https://github.com/joyent/node/commit/f06abd](https://github.com/joyent/node/commit/f06abd).

LICENSE
-------

(MIT License)

Copyright (c) 2013 [Richardson & Sons, LLC](http://richardsonandsons.com/)

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
