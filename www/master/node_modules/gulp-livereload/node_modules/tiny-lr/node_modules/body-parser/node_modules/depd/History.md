0.4.5 / 2014-09-09
==================

  * Improve call speed to functions using the function wrapper
  * Support Node.js 0.6

0.4.4 / 2014-07-27
==================

  * Work-around v8 generating empty stack traces

0.4.3 / 2014-07-26
==================

  * Fix exception when global `Error.stackTraceLimit` is too low

0.4.2 / 2014-07-19
==================

  * Correct call site for wrapped functions and properties

0.4.1 / 2014-07-19
==================

  * Improve automatic message generation for function properties

0.4.0 / 2014-07-19
==================

  * Add `TRACE_DEPRECATION` environment variable
  * Remove non-standard grey color from color output
  * Support `--no-deprecation` argument
  * Support `--trace-deprecation` argument
  * Support `deprecate.property(fn, prop, message)`

0.3.0 / 2014-06-16
==================

  * Add `NO_DEPRECATION` environment variable

0.2.0 / 2014-06-15
==================

  * Add `deprecate.property(obj, prop, message)`
  * Remove `supports-color` dependency for node.js 0.8

0.1.0 / 2014-06-15
==================

  * Add `deprecate.function(fn, message)`
  * Add `process.on('deprecation', fn)` emitter
  * Automatically generate message when omitted from `deprecate()`

0.0.1 / 2014-06-15
==================

  * Fix warning for dynamic calls at singe call site

0.0.0 / 2014-06-15
==================

  * Initial implementation
