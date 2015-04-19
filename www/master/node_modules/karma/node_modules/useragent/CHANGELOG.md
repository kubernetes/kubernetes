## Version 2.0
* __v2.0.0__ *breaking*
  - Added support for Operating System version parsing
  - Added support for Device parsing
  - Introduced deferred OnDemand parsing for Operating and Devices
  - The `Agent#toJSON` method now returns an object instread of JSON string. Use
    `JSON.stringify(agent)` instead.
  - Removed the fromAgent method
  - semver is removed from the dependencies, if you use the useragent/features
    you should add it to your own dependencies.

* __v2.0.1__
  - Fixed broken reference to the update module.
  - Updated with some new parsers.

* __v2.0.2__
  - Use LRU-cache for the lookups so it doesn't create a memory "leak" #22
  - Updated with some new parsers.

* __v2.0.3__
  - Updated regexp library with new parsers as Opera's latest browser which runs
    WebKit was detected as Chrome Mobile.

* __v2.0.4__
  - Added support for IE11 and PhantomJS. In addition to that when you run the
    updater without the correct dependencies it will just output an error
    instead of throwing an error.

* __v2.0.5__
  - Upgraded the regular expressions to support Opera Next

* __v2.0.6__
  - Only write the parse file when there isn't an error. #30
  - Output an error in the console when we fail to compile new parsers #30

## Version 1.0
* __v1.1.0__
  - Removed the postupdate hook, it was causing to much issues #9

* __v1.0.6__
  - Updated the agent parser, JHint issues and leaking globals.

* __v1.0.5__
  - Potential fix for #11 where it doesn't install the stuff in windows this also
    brings a fresh update of the agents.js.

* __v1.0.3__
  - Rewritten the `is` method so it doesn't display IE as true for firefox, chrome
    etc fixes #10 and #7.

* __v1.0.3__
  - A fix for bug #6, updated the semver dependency for browserify support.

* __v1.0.2__
  - Don't throw errors when .parse is called without a useragent string. It now
    defaults to a empty Agent instance.

* __v1.0.1__
  - Added support for cURL, Wget and thunderbird using a custom useragent
    definition file.

* __v1.0.0__ *breaking*
  - Complete rewrite of the API and major performance improvements.
