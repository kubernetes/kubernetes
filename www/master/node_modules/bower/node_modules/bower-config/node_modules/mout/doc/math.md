# math #

Math utilities.


## ceil(val[, step]):Number

Round value up to full steps. Similar to `Math.ceil()` but can round value
to an arbitrary *radix*.

    ceil(7.2);   // 8
    ceil(7.8);   // 8
    ceil(7, 5);  // 10
    ceil(11, 5); // 15
    ceil(15, 5); // 15

### Common use cases

Round values by increments of 5/10/1000/etc.

See: [`round()`](#round), [`floor()`](#floor), [`countSteps()`](#countSteps)



## clamp(val, min, max):Number

Clamps value inside range.

`clamp()` is extremely useful in cases where you need to limit a value inside
a certain range. So instead of doing a complex `if/else` to filter/process the
value you can restrict it to always be inside the desired range:

    clamp(-5, 0, 10); // 0
    clamp(7, 1, 10);  // 7
    clamp(8, 1, 10);  // 8
    clamp(10, 1, 10); // 10
    clamp(11, 1, 10); // 10

If the value is smaller than `min` it returns the `min`, if `val` is higher
than `max` it returns `max`.

### Common use cases

Any situation where you need to limit a number inside a range like: slider
position, image galleries <small>(so user can't skip to an image that doesn't
exist)</small>, drag and drop, scroll boundaries, etc.

See: [`loop()`](#loop)




## countSteps(val, step[, overflow]):Number

Count number of full steps.

### Arguments:

 1. `val` (Number)        : Value.
 2. `step` (Number)       : Step size.
 3. `[overflow]` (Number) : Maximum number of steps, nSteps will loop if
`>=` than overflow.


Count steps is very useful for cases where you need to know how many "full
steps" the number *completed*. Think of it as a division that only returns
integers and ignore remainders.

    countSteps(3,  5);    // 0
    countSteps(6,  5);    // 1
    countSteps(12, 5);    // 2
    countSteps(18, 5);    // 3
    countSteps(21, 5);    // 4

You can also set an `overflow` which will reset the *counter* before reaching
this number.

    countSteps(3, 5, 3);  // 0
    countSteps(6, 5, 3);  // 1
    countSteps(12, 5, 3); // 2
    countSteps(18, 5, 3); // 0
    countSteps(21, 5, 3); // 1

### Common use cases

#### How many items fit inside an area:

    var containerWidth = 100;
    var itemWidth = 8;
    var howManyFit = countSteps(containerWidth, itemWidth); // 12

#### Split value into different scales or convert value from one scale to another

From [mout/time/parseMs](time.html#parseMs):

    function parseMs(ms){
        return {
            milliseconds : countSteps(ms, 1, 1000),
            seconds      : countSteps(ms, 1000, 60),
            minutes      : countSteps(ms, 60000, 60),
            hours        : countSteps(ms, 3600000, 24),
            days         : countSteps(ms, 86400000)
        };
    }

    // {days:27, hours:4, minutes:26, seconds:5, milliseconds:454}
    parseMs(2348765454);



## floor(val[, step]):Number

Round value down to full steps. Similar to `Math.floor()` but can round value
to an arbitrary *radix*. (formerly `snap`)

    floor(7.2);   // 7
    floor(7.8);   // 7
    floor(7, 5);  // 5
    floor(11, 5); // 10
    floor(15, 5); // 15

### Common use cases

Round values by increments of 5/10/1000/etc.

See: [`round()`](#round), [`ceil()`](#ceil), [`countSteps()`](#countSteps)



## inRange(val, min, max[, threshold]):Boolean

Checks if value is inside the range.

    inRange(-6, 1, 10);    // false
    inRange( 5, 1, 10);    // true
    inRange(12, 1, 10);    // false

The threshold can be useful when you want granular control of what should match
and/or the precision could change at runtime or by some configuration option,
avoids the clutter of adding/subtracting the `threshold` from `mix` and `max`.

    inRange(12, 1, 10, 2); // true
    inRange(13, 1, 10, 2); // false

### Common use cases

Anything that needs to check if value is inside a range, be it collision
detection, limiting interaction by mouse position, ignoring parts of the logic
that shouldn't happen if value isn't valid, simplify `if/else` conditions,
making code more readable, etc...




## isNear(val, target, threshold):Boolean

Check if value is close to target.

Similar to `inRange()` but used to check if value is close to a certain value
or match the desired value. Basically to simplify `if/else` conditions and to
make code clearer.

    isNear( 10.2, 10, 0.5); // true
    isNear( 10.5, 10, 0.5); // true
    isNear(10.51, 10, 0.5); // false

### Common use cases

Games where a certain action should happen if an *actor* is close to a target,
gravity fields, any numeric check that has some tolerance.




## lerp(ratio, start, end):Number

Linear interpolation.

    lerp(0.5, 0, 10);  // 5
    lerp(0.75, 0, 10); // 7.5

### Common use cases

Linear interpolation is commonly used to create animations of elements moving
from one point to another, where you simply update the current ratio (which in
this case represents time) and get back the position of the element at that
"frame".

The core idea of `lerp` is that you are using a number that goes from `0` to
`1` to specify a ratio inside that scale. This concept can be applied to
convert numbers from different scales easily.

See: [`map()`](#map), [`norm()`](#norm)




## loop(val, min, max):Number

Loops value inside range. Will return `min` if `val > max` and `max` if `val
< min`, otherwise it returns `val`.

    loop(-1, 0, 10); // 10
    loop( 1, 0, 10); // 1
    loop( 5, 0, 10); // 5
    loop( 9, 0, 10); // 9
    loop(10, 0, 10); // 10
    loop(11, 0, 10); // 0

Similar to [`clamp()`](#clamp) but *loops* the value inside the range when an
overflow occurs.

### Common use cases

Image galleries, infinite scroll, any kind of logic that requires that the
first item should be displayed after the last one or the last one should be
displayed after first if going on the opposite direction.

See: [`clamp()`](#clamp)




## map(val, min1, max1, min2, max2):Number

Maps a number from one scale to another.

    map(3, 0, 4, -1, 1)   // 0.5
    map(3, 0, 10, 0, 100) // 30

### Common use cases

Very useful to convert values from/to multiple scales.

Let's suppose we have a slider that needs to go from `2000` to `5000` and that slider
has `300px` of width, here is how we would translate the knob position into the
current value:

    var knobX = 123;
    var sliderWid = 300;
    var minVal = 2000;
    var maxVal = 5000;

    var curVal = map(knobX, 0, sliderWid, minVal, maxVal); // 3230

See: [`lerp()`](#lerp), [`norm()`](#norm)




## norm(val, min, max):Number

Gets normalized ratio of value inside range.

    norm(50, 0, 100); // 0.5
    norm(75, 0, 100); // 0.75

### Common use cases

Convert values between scales, used by [`map()`](#map) internally. Direct
opposite of [`lerp()`](#lerp).

See: [`lerp()`](#lerp), [`map()`](#map)



## round(val[, step]):Number

Round value to full steps. Similar to `Math.round()` but allow setting an
arbitrary *radix*.

    // default
    round(0.22);      // 0
    round(0.49);      // 0
    round(0.51);      // 1

    // custom radix
    round(0.22, 0.5); // 0
    round(0.49, 0.5); // 0.5
    round(0.51, 0.5); // 0.5
    round(0.74, 0.5); // 0.5
    round(0.75, 0.5); // 1
    round(1.24, 0.5); // 1
    round(1.25, 0.5); // 1.5
    round(1.74, 0.5); // 1.5

### Common use cases

Round values by increments of 0.5/5/10/1000/etc.

See: [`floor()`](#floor), [`ceil()`](#ceil), [`countSteps()`](#countSteps)



-------------------------------------------------------------------------------

For more usage more info check the specs and source code.

