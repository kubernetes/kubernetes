# time #

Utilities for time manipulation.


## convert(value, sourceUnit, [destinationUnit]):Number

Converts time between units.

Available units: `millisecond`, `second`, `minute`, `hour`, `day`, `week`.
Abbreviations: `ms`, `s`, `m`, `h`, `d`, `w`.

We do **not** support year and month as a time unit since their values are not
fixed.

The default `destinationUnit` is `ms`.

```js
convert(1, 'minute');    // 60000
convert(2.5, 's', 'ms'); // 2500
convert(2, 'm', 's');    // 120
convert(500, 'ms', 's'); // 0.5
```



## now():Number

Returns the number of milliseconds elapsed since 1 January 1970 00:00:00 UTC.
Uses `Date.now()` if available.

### Example

```js
now(); // 1335449614650
```



## parseMs(ms):Object

Parse timestamp (milliseconds) into an object `{milliseconds:number,
seconds:number, minutes:number, hours:number, days:number}`.

### Example

```js
// {days:27, hours:4, minutes:26, seconds:5, milliseconds:454}
parseMs(2348765454);
```



## toTimeString(ms):String

Convert timestamp (milliseconds) into a time string in the format "[H:]MM:SS".

### Example

```js
toTimeString(12513);   // "00:12"
toTimeString(951233);  // "15:51"
toTimeString(8765235); // "2:26:05"
```
