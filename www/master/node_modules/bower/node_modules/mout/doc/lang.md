# lang #

Language Utilities. Easier inheritance, scope handling, type checks.



## clone(val):*

Clone native types like Object, Array, RegExp, Date and primitives.

This method will not clone values that are referenced within `val`. It will
only copy the value reference to the new value. If the value is not a plain
object but is an object, it will return the value unchanged.

### Example

```js
var a = { foo: 'bar' };
var b = clone(a);
console.log(a === b); // false
console.log(a.foo === b.foo); // true

var c = [1, 2, 3];
var d = clone(b);
console.log(c === d); // false
console.log(c); // [1, 2, 3]
```

See: [`deepClone()`](#deepClone)



## createObject(parent, [props]):Object

Create Object using prototypal inheritance and setting custom properties.

Mix between [Douglas Crockford Prototypal Inheritance](http://javascript.crockford.com/prototypal.html) and [`object/mixIn`](./object.html#mixIn).

### Arguments

 1. `parent` (Object)  : Parent Object
 2. `[props]` (Object) : Object properties

### Example

```js
var base = {
    trace : function(){
        console.log(this.name);
    }
};

var myObject = createObject(base, {
    name : 'Lorem Ipsum'
});

myObject.trace(); // "Lorem Ipsum"
```



## ctorApply(constructor, args):Object

Do `Function.prototype.apply()` on a constructor while maintaining prototype
chain.

```js
function Person(name, surname) {
    this.name = name;
    this.surname = surname;
}

Person.prototype.walk = function(){
    console.log(this.name +' is walking');
};

var args = ['John', 'Doe'];

// "similar" effect as calling `new Person("John", "Doe")`
var john = ctorApply(Person, args);
john.walk(); // "John is walking"
```



## deepClone(val, [instanceClone]):*

Deep clone native types like Object, Array, RegExp, Date and primitives.

The `instanceClone` function will be invoked to clone objects that are not
"plain" objects (as defined by [`isPlainObject`](#isPlainObject)) if it is
provided. If `instanceClone` is not specified, it will not attempt to clone
non-plain objects, and will copy the object reference.

### Example

```js
var a = {foo:'bar', obj: {a:1, b:2}};
var b = deepClone(a); // {foo:'bar', obj: {a:1, b:2}}
console.log( a === b ); // false
console.log( a.obj === b.obj ); // false

var c = [1, 2, [3, 4]];
var d = deepClone(c); // [1, 2, [3, 4]]
var e = c.concat(); // [1, 2, [3, 4]]

console.log( c[2] === d[2] ); // false
// concat doesn't do a deep clone, arrays are passed by reference
console.log( e[2] === d[2] ); // true

function Custom() { }
function cloneCustom(x) { return new Custom(); }
var f = { test: new Custom() };
var g = deepClone(f, cloneCustom);
g.test === f.test // false, since new Custom instance will be created
```

See: [`clone()`](#clone)



## deepEquals(a, b, [callback]):Boolean

Recursively tests whether two values contains the same keys and values.

`callback` specifies the equality comparison function used to compare
non-object values. It defaults to using the [`is()`](#is) function.

If the values are both an object or array, it will recurse into both values,
checking if their keys/values are equal. It will only check the keys and values
contained by the objects; it will not check the objects' prototypes.  If either
of the values are not objects, they will be checked using the `callback`
function.

Example:

```js
deepEquals({ a: 1 }, { a: 1 }); // true
deepEquals({ value: { a: 1 } }, { value: { a: 1 } }); // true
deepEquals({ value: { a: 1 } }, { value: { a: 2 } }); // false
deepEquals({ value: { a: 1 } }, { value: { a: 1, b: 2 } }); // false
deepEquals({}, null); // false
deepEquals(null, null); // true
deepEquals(
    { a: { b: 1 } },
    { a: { b: '1' } },
    function(a, b) { return a == b; }); // true
```

See: [object/equals](object.html#equals), [array/equals](array.html#equals)



## defaults(val, ...defaults):void

Return first value that isn't `null` or `undefined`.

    function doSomethingAwesome(foo, bar) {
        // default arguments
        foo = defaults(foo, 'lorem');
        bar = defaults(bar, 123);
        // ...
    }



## GLOBAL:Object

Reference to the global context (`window` inside a browser, `global` on
node.js). Works on ES3 and ES5 strict-mode.



## inheritPrototype(childCtor, parentCtor):Object

Inherit the prototype methods from one constructor into another.

Similar to [node.js util/inherits](http://nodejs.org/docs/latest/api/util.html#util_util_inherits_constructor_superconstructor).

It returns the the `childCtor.prototype` for convenience.

```js
function Foo(name){
    this.name = name;
}
Foo.prototype = {
    getName : function(){
        return this.name;
    }
};

function Bar(name){
    Foo.call(this, name);
}
//should be called before calling constructor
var proto = inheritPrototype(Bar, Foo);

// for convenience we return the new prototype object
console.log(proto === Bar.prototype); // true

var myObj = new Bar('lorem ipsum');
myObj.getName(); // "lorem ipsum"

console.log(myObj instanceof Foo); // true

// you also have access to the "super" constructor
console.log(Bar.super_ === Foo); // true
```


## is(x, y):Boolean

Check if both values are identical/egal.

```js
// wtfjs
NaN === NaN; // false
-0 === +0;   // true

is(NaN, NaN); // true
is(-0, +0);   // false
is('a', 'b'); // false
```

See: [`isnt()`](#isnt)



## isnt(x, y):Boolean

Check if both values are not identical/egal.

```js
// wtfjs
NaN === NaN; // false
-0 === +0;   // true

isnt(NaN, NaN); // false
isnt(-0, +0);   // true
isnt('a', 'b'); // true
```

See: [`is()`](#is)




## isArguments(val):Boolean

If value is an "Arguments" object.



## isArray(val):Boolean

If value is an Array. Uses native ES5 `Array.isArray()` if available.



## isBoolean(val):Boolean

If value is a Boolean.



## isDate(val):Boolean

If value is a Date.



## isEmpty(val):Boolean

Checks if Array/Object/String is empty.

Will return `true` for any object that doesn't contain enumerable properties
and also to any type of value that isn't considered a collection (boolean,
null, undefined, function, etc).

```js
isEmpty('');         // true
isEmpty('bar');      // false
isEmpty([]);         // true
isEmpty([1, 2]);     // false
isEmpty({});         // true
isEmpty({a:1, b:2}); // false
// null, undefined, booleans, numbers are considered as "empty" values
isEmpty(null);       // true
isEmpty(undefined);  // true
isEmpty(123);        // true
isEmpty(true);       // true
```


## isFinite(val):Boolean

Checks if value is Finite.

**IMPORTANT:** This is not the same as native `isFinite`, which will return
`true` for values that can be coerced into finite numbers. See
http://es5.github.com/#x15.1.2.5.

```js
isFinite(123);      // true
isFinite(Infinity); // false

// this is different than native behavior
isFinite('');   // false
isFinite(true); // false
isFinite([]);   // false
isFinite(null); // false
```


## isFunction(val):Boolean

If value is a Function.



## isKind(val, kind):Boolean

If value is of "kind". (used internally by some of the *isSomething* checks).

Favor the other methods since strings are commonly mistyped and also because
some "kinds" can only be accurately checked by using other methods (e.g.
`Arguments`), some of the other checks are also faster.

```js
isKind([1,2], 'Array'); // true
isKind(3, 'Array');     // false
isKind(3, 'Number');    // true
```

See: [`kindOf()`](#kindOf)



## isInteger(val):Boolean

Check if value is an integer.

```js
isInteger(123);    // true
isInteger(123.45); // false
isInteger({});     // false
isInteger('foo');  // false
isInteger('123');  // false
```



## isNaN(val):Boolean

Check if value is not a number.

It doesn't coerce value into number before doing the check, giving better
results than native `isNaN`. Returns `true` for everything besides numeric
values.

**IMPORTANT:** behavior is very different than the native `isNaN` and way more
useful!!! See: http://es5.github.io/#x15.1.2.4

```js
isNaN(123);       // false

isNaN(NaN);       // true
isNaN({});        // true
isNaN(undefined); // true
isNaN([4,5]);     // true

// these are all "false" on native isNaN and main reason why this module exists
isNaN('');    // true
isNaN(null);  // true
isNaN(true);  // true
isNaN(false); // true
isNaN("123"); // true
isNaN([]);    // true
isNaN([5]);   // true
```



## isNull(val):Boolean

If value is `null`.



## isNumber(val):Boolean

If value is a Number.



## isObject(val):Boolean

If value is an Object.



## isPlainObject(val):Boolean

If the value is an Object created by the Object constructor.



## isRegExp(val):Boolean

If value is a RegExp.



## isString(val):Boolean

If value is a String.



## isUndefined(val):Boolean

If value is `undefined`.



## kindOf(val):String

Gets kind of value (e.g. "String", "Number", "RegExp", "Null", "Date").
Used internally by `isKind()` and most of the other *isSomething* checks.

```js
kindOf([1,2]); // "Array"
kindOf('foo'); // "String"
kindOf(3);     // "Number"
```

See: [`isKind()`](#isKind)


## toArray(val):Array

Convert array-like object into Array or wrap value into Array.

```js
toArray({
    "0" : "foo",
    "1" : "bar",
    "length" : 2
});                              // ["foo", "bar"]

function foo(){
    return toArray(arguments);
}
foo("lorem", 123);               // ["lorem", 123]

toArray("lorem ipsum");          // ["lorem ipsum"]
toArray(window);                 // [window]
toArray({foo:"bar", lorem:123}); // [{foo:"bar", lorem:123}]
```

See: object/values()



## toNumber(val):Number

Convert value into number.

```js
// numeric values are typecasted as Number
toNumber('123');     // 123
toNumber(-567);      // -567

// falsy values returns zero
toNumber('');        // 0
toNumber(null);      // 0
toNumber(undefined); // 0
toNumber(false);     // 0

// non-numeric values returns NaN
toNumber('asd');     // NaN
toNumber({});        // NaN
toNumber([]);        // NaN

// Date objects return milliseconds since epoch
toNumber(new Date(1985, 6, 23)); // 490935600000
```



## toString(val):String

Convert any value to its string representation.

Will return an empty string for `undefined` or `null`, otherwise will convert
the value to its string representation.

```js
// null and undefined are converted into empty strings
toString(null);      // ""
toString(undefined); // ""

toString(1);       // "1"
toString([1,2,3]); // "1,2,3"
toString(false);   // "false"

// uses `val.toString()` to convert value
toString({toString:funtion(){ return 'foo'; }}); // "foo"
```



-------------------------------------------------------------------------------

For more usage examples check specs inside `/tests` folder. Unit tests are the
best documentation you can get...
