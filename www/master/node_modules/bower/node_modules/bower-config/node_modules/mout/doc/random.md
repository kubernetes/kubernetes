# random #

Pseudo-random generators.

mout uses `Math.random` by default on all the pseudo-random generators, if
you need a seeded random or a better algorithm see the [`random()`](#random)
documentation for instructions.



## choice(...items):*

Returns a random element from the supplied arguments or from an array if single
argument is an array.

### Example:

```js
choice(1, 2, 3, 4, 5); // 3

var arr = ['lorem', 'ipsum', 'dolor'];
choice(arr); // 'dolor'
```



## guid():String

Generates a pseudo-random [Globally Unique Identifier](http://en.wikipedia.org/wiki/Globally_unique_identifier) (v4).

Since the total number of GUIDs is 2<sup>122</sup> the chance of generating the
same value twice is negligible.

**Important:** this method uses `Math.random` by default so the UUID isn't
*safe* (sequence of outputs can be predicted in some cases), check the
[`random()`](#random) documentation for more info on how to replace the default
PRNG if you need extra safety or need *seeded* results.

See: [`randHex()`](#randHex), [`random()`](#random)

### Example:

```js
guid();      // 830e9f50-ac7f-4369-a14f-ed0e62b2fa0b
guid();      // 5de3d09b-e79c-4727-932b-48c49228d508
```



## rand([min], [max]):Number

Gets a random number inside range or snap to min/max values.

### Arguments:

 1. `[min]` (Number)         : Minimum value. Defaults to `number/MIN_INT`.
 2. `[max]` (Number)         : Maximum value. Defaults to `number/MAX_INT`.


### Example:

```js
rand();      // 448740433.55274725
rand();      // -31797596.097682
rand(0, 10); // 7.369723
rand(0, 10); // 5.987042
```

See: [`random()`](#random)



## randBit():Number

Returns a random "bit" (0 or 1). Useful for addition/subtraction.

It's slightly faster than `choice(0, 1)` since implementation is simpler (not
that it will make a huge difference in most cases).

See: [`choice()`](#choice)

### Example:

```js
randBit(); // 1
randBit(); // 0

//same effect as
choice(0, 1);
```


## randBool():Boolean

Returns a random Boolean (`true` or `false`).

Since this is very common it makes sense to abstract it into a discrete method.

### Example:

```js
randBool(); // true
randBool(); // false
```



## randHex([size]):String

Returns a random hexadecimal string.

The default `size` is `6`.

### Example:

```js
randHex();   // "dd8575"
randHex();   // "e6baeb"
randHex(2);  // "a2"
randHex(30); // "effd7e2ad9a4a3067e30525fab983a"
```



## randInt([min], [max]):Number

Gets a random integer inside range or snap to min/max values.

### Arguments:

 1. `[min]` (Number)         : Minimum value. Defaults to `number/MIN_INT`.
 2. `[max]` (Number)         : Maximum value. Defaults to `number/MAX_INT`.


### Example:

```js
randInt();      // 448740433
randInt();      // -31797596
randInt(0, 10); // 7
randInt(0, 10); // 5
```



## randSign():Number

Returns a random "sign" (-1 or 1). Useful for multiplications.

It's slightly faster than `choice(-1, 1)` since implementation is simpler (not
that it will make a huge difference in most cases).

See: [`choice()`](#choice)

### Example:

```js
randSign(); // -1
randSign(); // 1

//same effect as
choice(-1, 1);
```



## random():Number

Returns a random number between `0` and `1`. Same as `Math.random()`.

```js
random(); // 0.35435103671625257
random(); // 0.8768321881070733
```

**Important:** No methods inside mout should call `Math.random()`
directly, they all use `random/random` as a proxy, that way we can
inject/replace the pseudo-random number generator if needed (ie. in case we
need a seeded random or a better algorithm than the native one).

### Replacing the PRNG

In some cases we might need better/different algorithms than the one provided
by `Math.random` (ie. safer, seeded).

Because of licensing issues, file size limitations and different needs we
decided to **not** implement a custom PRNG and instead provide a easy way to
override the default behavior. - [issue #99](https://github.com/millermedeiros/amd-utils/issues/99)

If you are using mout with a loader that supports the [AMD map
config](https://github.com/amdjs/amdjs-api/wiki/Common-Config), such as
[RequireJS](http://requirejs.org/), you can use it to replace the PRNG
(recommended approach):

```js
requirejs.config({
    map : {
        // all modules will load "my_custom_prng" instead of
        // "mout/random/random"
        '*' : {
            'mout/random/random' : 'my_custom_prng'
        }
    }
});
```

You also have the option to override `random.get` in case you are using
mout on node.js or with a loader which doesn't support the map config:

```js
// replace the PRNG
var n = 0;
random.get = function(){
    return ++n % 2? 0 : 1; // not so random :P
};
random(); // 0
random(); // 1
random(); // 0
random(); // 1
```

See this [detailed explanation about PRNG in
JavaScript](http://baagoe.org/en/w/index.php/Better_random_numbers_for_javascript)
to understand the issues with the native `Math.random` and also for a list of
algorithms that could be used instead.



-------------------------------------------------------------------------------

For more usage examples check specs inside `/tests` folder. Unit tests are the
best documentation you can get...
