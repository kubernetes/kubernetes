# dateformat

A node.js package for Steven Levithan's excellent [dateFormat()][dateformat] function.

[![Build Status](https://travis-ci.org/felixge/node-dateformat.svg)](https://travis-ci.org/felixge/node-dateformat)

## Modifications

* Removed the `Date.prototype.format` method. Sorry folks, but extending native prototypes is for suckers.
* Added a `module.exports = dateFormat;` statement at the bottom
* Added the placeholder `N` to get the ISO 8601 numeric representation of the day of the week

## Installation

```bash
$ npm install dateformat
$ dateformat --help
```

## Usage

As taken from Steven's post, modified to match the Modifications listed above:
```js
    var dateFormat = require('dateformat');
    var now = new Date();

    // Basic usage
    dateFormat(now, "dddd, mmmm dS, yyyy, h:MM:ss TT");
    // Saturday, June 9th, 2007, 5:46:21 PM

    // You can use one of several named masks
    dateFormat(now, "isoDateTime");
    // 2007-06-09T17:46:21

    // ...Or add your own
    dateFormat.masks.hammerTime = 'HH:MM! "Can\'t touch this!"';
    dateFormat(now, "hammerTime");
    // 17:46! Can't touch this!

    // When using the standalone dateFormat function,
    // you can also provide the date as a string
    dateFormat("Jun 9 2007", "fullDate");
    // Saturday, June 9, 2007

    // Note that if you don't include the mask argument,
    // dateFormat.masks.default is used
    dateFormat(now);
    // Sat Jun 09 2007 17:46:21

    // And if you don't include the date argument,
    // the current date and time is used
    dateFormat();
    // Sat Jun 09 2007 17:46:22

    // You can also skip the date argument (as long as your mask doesn't
    // contain any numbers), in which case the current date/time is used
    dateFormat("longTime");
    // 5:46:22 PM EST

    // And finally, you can convert local time to UTC time. Simply pass in
    // true as an additional argument (no argument skipping allowed in this case):
    dateFormat(now, "longTime", true);
    // 10:46:21 PM UTC

    // ...Or add the prefix "UTC:" or "GMT:" to your mask.
    dateFormat(now, "UTC:h:MM:ss TT Z");
    // 10:46:21 PM UTC

    // You can also get the ISO 8601 week of the year:
    dateFormat(now, "W");
    // 42

    // and also get the ISO 8601 numeric representation of the day of the week:
    dateFormat(now,"N");
    // 6
```
## License

(c) 2007-2009 Steven Levithan [stevenlevithan.com][stevenlevithan], MIT license.

[dateformat]: http://blog.stevenlevithan.com/archives/date-time-format
[stevenlevithan]: http://stevenlevithan.com/
