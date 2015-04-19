# function #

Function*(al)* utilities.


## awaitDelay(fn, delay):Function

Returns a function that ensures that `fn` is only called *after* `delay`
milliseconds have elapsed. When the returned function is called before the
delay has elapsed, it will wait until the delay has elapsed and then call `fn`.
When the returned function is called after the delay has elapsed, it will call
`fn` after the next "tick" (it will always be called asynchronously). The
context and arguments that the returned function is called in are applied to
`fn`.

In the below example `onLoaded` will not be executed before a 1000 millisecond
delay. Even if `loadImages` loads and calls `callback` earlier.  However, say
the images take 1500 milliseconds to load, it will trigger `onLoaded`
immediately.

```js
var callback = after(onLoaded, 1000);
loadImages(callback);
function onLoaded(){
    console.log('loaded');
}
```

You can also cancel de delayed call by simply using the native `clearTimeout`
method (like a regular `setTimeout` call).

```js
var timeoutId = callback();
// onLoaded won't be called since it was canceled before the 1000ms delay
clearTimeout(timeoutId);
```

### Arguments:

 1. `fn` (Function)    : Target Function
 2. `delay` (Number)   : Delay of execution in milliseconds

See: [`debounce()`](#debounce)



## bind(fn, context, [...args]):Function

Return a function that will execute in the given context, optionally adding any additional supplied parameters to the beginning of the arguments collection.

### Arguments

 1. `fn` (Function)    : Target Function
 2. `context` (Object) : Execution context (object used as `this`)
 3. `[...args]` (*)    : Arguments (0...n arguments)

See: [`partial()`](#partial), [object/bindAll](./object.html#bindAll)



## compose(...fn):Function

Returns the composition of a list of functions, where each function consumes
the return value of the function that follows. In math terms, composing the
functions `f()`, `g()`, and `h()` produces `f(g(h()))`.

```js
function add2(x) { return x + 2 }
function multi2(x) { return x * 2 }
map([1, 2, 3], compose(add2, multi2)); // [4, 6, 8]

//same as
map([1, 2, 3], function(x){
    return add2( multi2(x) );
});
```



## constant(value):Function

Returns a new function that will always return `value` when called.

```js
var f = constant('foo');
f(); // 'foo'

// Provided arguments are ignored; value is always returned
f(1); // 'foo'

f = constant({ foo: 'bar' });
f(); // { foo: 'bar' }
```



## debounce(fn, delay[, isAsap]):Function

Creates a function that will delay the execution of `fn` until after `delay`
milliseconds have elapsed since the last time it was invoked.

Subsequent calls to the debounced function will return the result of the last
`fn` call.

```js
// sometimes less is more
var lazyRedraw = debounce(redraw, 300);
foo.on.resize.add(lazyRedraw);
```

In this visualization, `|` is a debounced-function call and `X` is the actual
callback execution:

    Default
    ||||||||||||||||||||||||| (pause) |||||||||||||||||||||||||
                             X                                 X

    Debounced with `isAsap == true`:
    ||||||||||||||||||||||||| (pause) |||||||||||||||||||||||||
    X                                 X

You also have the option to cancel the debounced call if it didn't happen yet:

```js
lazyRedraw();
// lazyRedraw won't be called since `cancel` was called before the `delay`
lazyRedraw.cancel();
```

See: [`throttle()`](#throttle)


## func(name):Function

Returns a function that calls a method with given `name` on supplied object.
Useful for iteration methods like `array/map` and `array/forEach`.

See: [`prop()`](#prop)

```js
// will call the method `getName()` for each `user`
var names = map(users, func('getName'));
```



## identity(val):*

Returns the first argument provided to it.

```js
identity(3);     // 3
identity(1,2,3); // 1
identity('foo'); // "foo"
```



## partial(fn, [...args]):Function

Return a partially applied function supplying default arguments.

This method is similar to [`bind`](#bind), except it does not alter the this
binding.

### Arguments

 1. `fn` (Function)    : Target Function
 2. `[...args]` (*)    : Arguments (0...n arguments)

See: [`bind()`](#bind)

```js
function add(a, b){ return a + b }
var add10 = partial(add, 10);
console.log( add10(2) ); // 12
```



## prop(name):Function

Returns a function that gets a property with given `name` from supplied object.
Useful for using in conjunction with `array/map` and/or for creating getters.

See: [`array/pluck()`](array.html#pluck)

```js
var users = [{name:"John", age:21}, {name:"Jane", age:25}];
// ["John", "Jane"]
var names = map(users, prop('name'));
```



## series(...fn):Function

Returns a function that will execute all the supplied functions in order and
passing the same parameters to all of them. Useful for combining multiple
`array/forEach` into a single one and/or for debugging.

```js
// call `console.log()` and `doStuff()` for each item item in the array
forEach(arr, series(console.log, doStuff));
```



## throttle(fn, interval):Function

Creates a function that, when executed, will only call the `fn` function at
most once per every `interval` milliseconds.

If the throttled function is invoked more than once during the wait timeout,
`fn` will also be called on the trailing edge of the timeout.

Subsequent calls to the throttled function will return the result of the last
`fn` call.

```js
// sometimes less is more
var lazyRedraw = throttle(redraw, 300);
foo.on.resize.add(lazyRedraw);
```

In this visualization, `|` is a throttled-function call and `X` is the actual
`fn` execution:

    ||||||||||||||||||||||||| (pause) |||||||||||||||||||||||||
    X    X    X    X    X    X        X    X    X    X    X    X

You also have the option to cancel the throttled call if it didn't happen yet:

```js
lazyRedraw();
setTimeout(function(){
    lazyRedraw();
    // lazyRedraw will be called only once since `cancel` was called before
    // the `interval` for 2nd call completed
    lazyRedraw.cancel();
}, 250);
```

See: [`debounce()`](#debounce)


## timeout(fn, millis, context, [...args]):Number

Functions as a wrapper for `setTimeout`. Calls a the function `fn` after a given delay `millis` in milliseconds.
The function is called within the specified context. The return value can be used to clear the timeout using `clearTimeout`.

```js
var id = timeout(doStuff, 300, this);

clearTimeout(id);
```

## times(n, callback, [context]):void

Iterates over a callback `n` times.

### Arguments

 1. `n` (Number)           : Number of iterations
 2. `callback` (Function)  : Closure executed for every iteration
 3. `context` (Object)     : Execution context (object used as `this`)

```js
var output = '';
times(5, function(i) {
    output += i.toString();
});
// output: 01234
```

## wrap(fn, wrapper):Function

Wraps the first `fn` inside of the `wrapper` function, passing it as the first argument. This allows the `wrapper` to execute code before and after the `fn` runs, adjust the arguments, and execute it conditionally.

```js
var hello = function(name) { return "hello: " + name; };
hello = wrap(hello, function(func) {
  return "before, " + func("moe") + ", after";
});
hello();
// output: 'before, hello: moe, after'
```

See: [`partial()`](#partial)
-------------------------------------------------------------------------------

For more usage examples check specs inside `/tests` folder. Unit tests are the
best documentation you can get...
