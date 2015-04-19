# number #

Number utilities.


## abbreviate(val[, nDecimalDigits, dictionary]):String

Abbreviate number to thousands (K), millions (M) or billions (B).

The default value for `nDecimalDigits` is `1`.

### Example

    abbreviate(123456);     // "123.5K"
    abbreviate(12345678);   // "12.3M"
    abbreviate(1234567890); // "1.2B"

You can set the amount of decimal digits (default is `1`):

    abbreviate(543);    // "0.5K"
    abbreviate(543, 1); // "0.5K"
    abbreviate(543, 2); // "0.54K"
    abbreviate(543, 3); // "0.543K"

You can customize the abbreviation by passing a custom "dictionary":

    var _ptbrDict = {
        thousands : ' mil',
        millions : ' Mi',
        billions : ' Bi'
    };
    function customAbbr(val) {
        return abbreviate(val, 1, _ptbrDict);
    }

    customAbbr(123456); // "123.5 mil"
    customAbbr(12345678); // "12.3 Mi"
    customAbbr(1234567890); // "1.2 Bi"



## currencyFormat(val[, nDecimalDigits, decimalSeparator, thousandsSeparator]):String

Format a number as currency.

### Example:

    currencyFormat(1000);              // "1,000.00"
    currencyFormat(1000, 1);           // "1,000.0"
    currencyFormat(1000, 2, ',', '.'); // "1.000,00"



## enforcePrecision(val, nDecimalDigits):Number

Enforce a specific amount of decimal digits and also fix floating point
rounding issues.

### Example:

```js
enforcePrecision(0.615, 2); // 0.62
enforcePrecision(0.625, 2); // 0.63
//floating point rounding "error" (rounds to odd number)
+(0.615).toFixed(2);        // 0.61
+(0.625).toFixed(2);        // 0.63
```


## isNaN(val):Boolean

ES6 `Number.isNaN()`, checks if supplied value is `NaN`.

```js
// only returns `true` for `NaN`
isNaN(NaN);    // true
isNaN(0 / 0);  // true

// everything else is `false`
isNaN(true);   // false
isNaN(123);    // false
isNaN('asd');  // false
isNaN('NaN');  // false
```


## MAX_INT:Number

Maximum 32-bit signed integer value. `Math.pow(2, 31) - 1`

### Example:

```js
console.log( MAX_INT ); // 2147483647
```


## MAX_UINT:Number

Maximum 32-bit unsigned integer value. `Math.pow(2, 32) - 1`

### Example:

```js
console.log( MAX_UINT ); // 4294967295
```


## MIN_INT:Number

Minimum 32-bit signed integer value. `Math.pow(2, 31) * -1`.

### Example:

```js
console.log( MIN_INT ); // -2147483648
```


## nth(n):String

Returns the "nth" of number. (`"st"`, `"nd"`, `"rd"`, `"th"`)

```js
nth(1); // "st"
nth(2); // "nd"
nth(12); // "th"
nth(22); // "nd"
nth(23); // "rd"
nth(34); // "th"
```

See: [`ordinal()`](#ordinal)



## ordinal(n):String

Converts number into ordinal form (1st, 2nd, 3rd, 4th, ...)

```js
ordinal(1); // "1st"
ordinal(2); // "2nd"
ordinal(3); // "3rd"
ordinal(14); // "14th"
ordinal(21); // "21st"
```

See: [`nth()`](#nth)



## pad(n, minLength[, char]):String

Add padding zeros if `n.length` < `minLength`.

### Example:

```js
pad(1, 5);      // "00001"
pad(12, 5);     // "00012"
pad(123, 5);    // "00123"
pad(1234, 5);   // "01234"
pad(12345, 5);  // "12345"
pad(123456, 5); // "123456"

// you can also specify the "char" used for padding
pad(12, 5, '_'); // "___12"
```

see: [string/lpad](./string.html#lpad)



## rol(val, shift):Number

Bitwise circular shift left.

More info at [Wikipedia#Circular_shift](http://en.wikipedia.org/wiki/Circular_shift)



## ror(val, shift):Number

Bitwise circular shift right.

More info at [Wikipedia#Circular_shift](http://en.wikipedia.org/wiki/Circular_shift)



## sign(val):Number

Returns `-1` if value is negative, `0` if the value is `0` and `1` if value is positive. Useful for
multiplications.

```js
sign(-123); // -1
sign(123);  // 1
sign(0);    // 0
```



## toInt(val):Number

"Convert" value into an 32-bit integer.  Works like `Math.floor` if `val > 0` and
`Math.ceil` if `val < 0`.

**IMPORTANT:** val will wrap at [number/MIN_INT](#MIN_INT) and
[number/MAX_INT](#MAX_INT).

Created because most people don't know bitwise operations and also because this
feature is commonly needed.

[Perf tests](http://jsperf.com/vs-vs-parseint-bitwise-operators/7)

### Example:

```js
toInt(1.25);   // 1
toInt(0.75);   // 0
toInt(-0.55);  // 0
toInt(-5.0001) // -5
```



## toUInt(val):Number

"Convert" value into an 32-bit unsigned integer.

Works like AS3#uint().

**IMPORTANT:** val will wrap at 2^32.

### Example:

```js
toUInt(1.25);                 // 1
toUInt(0.75);                 // 0
toUInt(-0.55);                // 0
toUInt(-5.0001);              // 4294967291
toUInt(Math.pow(2,32) - 0.5); // 4294967295
toUInt(Math.pow(2,32) + 0.5); // 0
```


## toUInt31(val):Number

"Convert" value into an 31-bit unsigned integer (since 1 bit is used for sign).

Useful since all bitwise operators besides `>>>` treat numbers as signed
integers.

**IMPORTANT:** val will wrap at 2^31 and negative numbers will be treated as
`zero`.

### Example:

```js
toUInt31(1.25);                 // 1
toUInt31(0.75);                 // 0
toUInt31(-0.55);                // 0
toUInt31(-5.0001);              // 0
toUInt31(Math.pow(2,31) - 0.5); // 21474836470
toUInt31(Math.pow(2,31) + 0.5); // 0
```


-------------------------------------------------------------------------------

For more usage examples check specs inside `/tests` folder. Unit tests are the
best documentation you can get...

