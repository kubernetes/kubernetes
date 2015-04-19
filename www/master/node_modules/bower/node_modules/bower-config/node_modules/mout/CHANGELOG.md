mout changelog
==============

v0.9.1 (2014/04/08)
-------------------

 - fix `array/slice` behavior when start and end are higher than length.


v0.9.0 (2014/02/04)
-------------------

 - add `date/quarter`;
 - add `function/constant`;
 - add `random/randBool`;
 - add un-padded 12-hour (`%l`) to `date/strftime`;
 - fix `array/slice` on IE < 9 by using a custom implementation.
 - fix `object/forIn` iteration for IE < 9 constructor property;
 - improve `lang/inheritPrototype` by returning the `prototype`;
 - improve `string/removeNonWord` to cover more chars;
 - improve `string/repeat` performance;
 - improve `string/unescapeHtml` by accepting leading zeros for `&#39`;


v0.8.0 (2013/11/22)
-------------------

 - add `array/findLast`.
 - add `array/findLastIndex`.
 - add `array/slice` and use it internally.
 - add `array/sortBy`
 - add `function/awaitDelay`.
 - add `function/identity`
 - add `number/isNaN`.
 - add `number/nth`.
 - add `number/ordinal`.
 - allows nested replacement patterns in `string/interpolate`.
 - change `function/makeIterator_` behavior (uses `identity` by default).
 - simplify `string/escapeRegExp`.
 - support custom equality on `array/compare`.


v0.7.1 (2013/09/18)
-------------------

 - fix `null` value handling in object/get.


v0.7.0 (2013/09/05)
-------------------

 - add bower ignores.
 - add german translation for date localization.
 - alias `function` package as `fn` since "function" is a reserved keyword.
 - allow second argument on `array/pick`.
 - improve `string/makePath` to not remove "//" after protocol.
 - make sure methods inside `number` package works with mixed types.
 - support arrays on `queryString/encode`.
 - support multiple values for same property on `queryString/decode`.
 - add `cancel()` method to `throttled/debounced` functions.
 - add `function/times`.
 - add `lang/toNumber`.
 - add `string/insert`.
 - add `super_` to constructor on `lang/inheritPrototype`.


v0.6.0 (2013/05/22)
-------------------

 - add optional delimeter to `string/unCamelCase`
 - allow custom char on `number/pad`
 - allow underscore characters in `string/removeNonWord`
 - accept `level` on `array/flatten` instead of a flag
 - convert underscores to camelCase in `string/camelCase`
 - remove `create()` from number/currencyFormat
 - add `date/dayOfTheYear`
 - add `date/diff`
 - add `date/isSame`
 - add `date/startOf`
 - add `date/strftime`
 - add `date/timezoneAbbr`
 - add `date/timezoneOffset`
 - add `date/totalDaysInYear`
 - add `date/weekOfTheYear`
 - add `function/timeout`
 - add `object/bindAll`
 - add `object/functions`
 - add `time/convert`


v0.5.0 (2013/04/04)
-------------------

 - add `array/collect`
 - add `callback` parameter to `object/equals` and `object/deepEquals` to allow
   custom compare operations.
 - normalize behavior in `array/*` methods to treat `null` values as empty
   arrays when reading from array
 - add `date/parseIso`
 - add `date/isLeapYear`
 - add `date/totalDaysInMonth`
 - add `object/deepMatches`
 - change `function/makeIterator_` to use `deepMatches` (affects nearly all
   iteration methods)
 - add `thisObj` parameter to `array/min` and `array/max`


v0.4.0 (2013/02/26)
-------------------

 - add `object/equals`
 - add `object/deepEquals`
 - add `object/matches`.
 - add `lang/is` and `lang/isnt`.
 - add `lang/isInteger`.
 - add `array/findIndex`.
 - add shorthand syntax to `array/*`, `object/*` and `collection/*` methods.
 - improve `number/sign` behavior when value is NaN or +0 or -0.
 - improve `lang/isNaN` to actually check if value *is not a number* without
   coercing value; so `[]`, `""`, `null` and `"12"` are considered NaN (#39).
 - improve `string/contains` to match ES6 behavior (add fromIndex argument).


v0.3.0 (2013/02/01)
-------------------

 - add `lang/clone`.
 - add `lang/toString`.
 - add `string/replace`.
 - add `string/WHITE_SPACES`
 - rename `function/curry` to `function/partial`.
 - allow custom chars in `string/trim`, `ltrim`, and `rtrim`.
 - convert values to strings in the `string/*` functions.


v0.2.0 (2013/01/13)
-------------------

 - fix bug in `math/ceil` for negative radixes.
 - change `object/deepFillIn` and `object/deepMixIn` to recurse only if both
   existing and new values are plain objects. Will not recurse into arrays
   or objects not created by the Object constructor.
 - add `lang/isPlainObject` to check if a file is a valid object and is created
   by the Object constructor
 - change `lang/clone` behavior when dealing with custom types (avoid cloning
   it by default) and add second argument to allow custom behavior if needed.
 - rename `lang/clone` to `lang/deepClone`.
 - add VERSION property to index.js
 - simplify `math/floor`, `math/round`, `math/ceil` and `math/countSteps`.


v0.1.0 (2013/01/09)
-------------------

- Rename project from "amd-utils" to "mout"

