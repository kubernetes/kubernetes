# object #

Object utilities.



## bindAll(obj, [...methodNames]):void

Bind methods of the target object to always execute on its own context
(ovewritting the original function).

See: [function/bind](./function.html#bind)

```js
var view = {
    name: 'Lorem Ipsum',
    logNameOnClick: function() {
        console.log(this.name);
    }
};

// binds all methods by default
bindAll(view);
jQuery('#docs').on('click', view.logNameOnClick);
```

You can also specify the list of methods that you want to bind (in case you
just want to bind a few of them).

```js
// only the listed methods will be bound to `obj` context
bindAll(obj, 'logNameOnClick', 'doAwesomeStuffOnDrag');
```



## contains(obj, value):Boolean

Similar to [Array/contains](array.html#contains). Checks if Object contains
value.

```js
var obj = {
    a: 1,
    b: 2,
    c: 'bar'
};
contains(obj, 2);      // true
contains(obj, 'foo');  // false
```



## deepFillIn(target, ...objects):Object

Fill missing properties recursively.

It's different from `deepMixIn` since it won't override any existing property.
It's also different from `merge` since it won't clone child objects during the
process.

It returns the target object and mutates it in place.

See: [`fillIn()`](#fillIn), [`deepMixIn()`](#deepMixIn), [`merge()`](#merge)

```js
var base = {
    foo : {
        bar : 123
    },
    lorem : 'ipsum'
};
var options = deepFillIn({foo : { baz : 45 }, lorem : 'amet'}, base);
// > {foo: {bar:123, baz : 45}, lorem : 'amet'}
```



## deepMatches(target, pattern):Boolean

Recursively checks if object contains all properties/value pairs. When both
the target and pattern values are arrays, it checks that the target value
contain matches for all the items in the pattern array (independent of order).

```js
var john = {
    name: 'John',
    age: 22,
    pets: [
        { type: 'cat', name: 'Grumpy Cat' },
        { type: 'dog', name: 'Hawk' }
    ]
};

deepMatches(john, { name: 'John' }); // true
deepMatches(john, { age: 21 }); // false
deepMatches(john, { pets: [ { type: 'cat' } ] }); // true
deepMatches(john, { pets: [ { name: 'Hawk' } ] }); // true
deepMatches(john, { pets: [ { name: 'Hairball' } ] }); // false
```

See [`matches()`](#matches)



## deepMixIn(target, ...objects):Object

Mixes objects into the target object, recursively mixing existing child objects
as well.

It will only recursively mix objects if both (existing and new) values are
plain objects.

Returns the target object. Like [`merge()`](#merge), but mutates the target
object, and does not clone child objects.

```js
var target = {
    foo: {
        name: "foo",
        id: 1
    }
};

deepMixIn(target, { foo: { id: 2 } });
console.log(target); // { foo: { name: "foo", id: 2 } }
```

See: [`mixIn()`](#mixIn), [`merge()`](#merge), [`deepFillIn()`](#deepFillIn)



## equals(a, b, [callback]):Boolean

Tests whether two objects contain the same keys and values.

`callback` specifies the equality comparison function used to compare the
values. It defaults to using [lang/is](lang.html#is).

It will only check the keys and values contained by the objects; it will not
check the objects' prototypes. If either of the values are not objects, they
will be compared using the `callback` function.

```js
equals({}, {}); // true
equals({ a: 1 }, { a: 1 }); // true
equals({ a: 1 }, { a: 2 }); // false
equals({ a: 1, b: 2 }, { a: 1 }); // false
equals({ a: 1 }, { a: 1, b: 2 }); // false
equals(null, null); // true
equals(null, {}); // false
equals({ a: 1 }, { a: '1' }, function(a, b) { return a == b; }); // true
```

See: [array/equals](array.html#equals), [lang/deepEquals](lang.html#deepEquals)


## every(obj, callback, [thisObj]):Boolean

Similar to [Array/every](array.html#every). Tests whether all properties in the
object pass the test implemented by the provided callback.

```js
var obj = {
    a: 1,
    b: 2,
    c: 3,
    d: 'string'
};

every(obj, isNumber); // false
```



## fillIn(obj, ...default):Object

Fill in missing properties in object with values from the *defaults* objects.

    var base = {
        foo : 'bar',
        num : 123
    };

    fillIn({foo:'ipsum'}, base); // {foo:'ipsum', num:123}

PS: it allows merging multiple objects at once, the first ones will take
precedence.

See: [`mixIn()`](#mixIn), [`merge()`](#merge), [`deepFillIn()`](#deepFillIn)



## filter(obj, callback, [thisObj])

Returns a new object containing all properties where `callback` returns true,
similar to Array/filter. It does not use properties from the object's
prototype.

Callback receives the same arguments as `forOwn()`.

See: [`forOwn()`](#forOwn), [`forIn()`](#forIn), [`pick()`](#pick)

```js
var obj = {
    foo: 'value',
    bar: 'bar value'
};

// returns { bar: 'bar value' }
filter(obj, function(v) { return value.length > 5; });

// returns { foo: 'value' }
filter(obj, function(v, k) { return k === 'foo'; });
```



## find(obj, callback, [thisObj])

Loops through all the properties in the Object and returns the first one that
passes a truth test (callback), similar to [Array/find](array.html#find).
Unlike Array/find, order of iteration is not guaranteed.

```js
var obj = {
    a: 'foo',
    b: 12
};

find(obj, isString); // 'foo'
find(obj, isNumber); // 12
```



## forIn(obj, callback[, thisObj])

Iterate over all properties of an Object, similar to
[Array/forEach](array.html#forEach).

It [avoids don't enum bug on IE](https://developer.mozilla.org/en/ECMAScript_DontEnum_attribute#JScript_DontEnum_Bug).
It **will** iterate over inherited (enumerable) properties from the prototype.

It allows exiting the iteration early by returning `false` on the callback.

See: [`forOwn()`](#forOwn), [`keys()`](#keys), [`values()`](#values)

### Callback arguments

Callback will receive the following arguments:

 1. Property Value (*)
 2. Key name (String)
 3. Target object (Object)

### Example

```js
function Foo(){
    this.foo = 1;
    this.bar = 2;
}

Foo.prototype.lorem = 4;

var obj = new Foo();

var result = 0;
var keys = [];

forIn(obj, function(val, key, o){
    result += val;
    keys.push(key);
});

console.log(result); // 7
console.log(keys);   // ['foo', 'bar', 'lorem']
```



## forOwn(obj, callback[, thisObj])

Iterate over all own properties from an Object, similar to
[Array/forEach](array.html#forEach).

It [avoids don't enum bug on IE](https://developer.mozilla.org/en/ECMAScript_DontEnum_attribute#JScript_DontEnum_Bug).
Notice that it **won't** iterate over properties from the prototype.

It allows exiting the iteration early by returning `false` on the callback.

See: [`forIn()`](#forIn), [`keys()`](#keys), [`values()`](#values)

### Callback arguments

Callback will receive the following arguments:

 1. Property Value (*)
 2. Key name (String)
 3. Target object (Object)

### Example

```js
function Foo(){
    this.foo = 1;
    this.bar = 2;
}

// will be ignored
Foo.prototype.lorem = 4;

var obj = new Foo();

var result = 0;
var keys = [];

forOwn(obj, function(val, key, o){
    result += val;
    keys.push(key);
});

console.log(result); // 3
console.log(keys);   // ['foo', 'bar']
```



## functions(obj):Array

Returns a sorted list of all enumerable properties that have function values
(including inherited properties).

```js
var obj = {
    foo : function(){},
    bar : 'baz'
};
functions(obj); // ['foo']
```



## get(obj, propName):*

Returns nested property value. Will return `undefined` if property doesn't
exist.

See: [`set()`](#set), [`namespace()`](#namespace), [`has()`](#has)

```js
var lorem = {
        ipsum : {
            dolor : {
                sit : 'amet'
            }
        }
    };

get(lorem, 'ipsum.dolor.sit'); // "amet"
get(lorem, 'foo.bar');         // undefined
```



## has(obj, propName):Boolean

Checks if object contains a child property. Useful for cases where you need to
check if an object contain a *nested* property. It will get properties
inherited by the prototype.

see: [`hasOwn()`](#hasOwn), [`get()`](#get)

```js
var a = {
        b : {
            c : 123
        }
    };

has(a, 'b.c');   // true
has(a, 'foo.c'); // false
```

### Common use case

```js
if( has(a, 'foo.c') ){ // false
    // ...
}

if( a.foo.c ){ // ReferenceError: `foo` is not defined
    // ...
}
```



## hasOwn(obj, propName):Boolean

Safer `Object.hasOwnProperty`. Returns a boolean indicating whether the object
has the specified property.

see: [`has()`](#has)

```js
var obj = {
    foo: 1,
    hasOwnProperty : 'bar'
};

obj.hasOwnProperty('foo'); // ERROR! hasOwnProperty is not a function

hasOwn(obj, 'foo');            // true
hasOwn(obj, 'hasOwnProperty'); // true
hasOwn(obj, 'toString');       // false
```



## keys(obj):Array

Returns an array of all own enumerable properties found upon a given object.
It will use the native `Object.keys` if present.

PS: it won't return properties from the prototype.

See: [`forOwn()`](#forOwn), [`values()`](#values)

```js
var obj = {
    foo : 1,
    bar : 2,
    lorem : 3
};
keys(obj); // ['foo', 'bar', 'lorem']
```



## map(obj, callback, [thisObj]):Object

Returns a new object where the property values are the result of calling the
callback for each property in the original object, similar to Array/map.

The callback function receives the same arguments as in `forOwn()`.

See: [`forOwn()`](#forOwn)

```js
var obj = { foo: 1, bar: 2 },
    data = { foo: 0, bar: 1 };

map(obj, function(v) { return v + 1; }); // { foo: 2, bar: 3 }
map(obj, function(v, k) { return k; }); // { foo: "foo", bar: "bar" }
map(obj, function(v, k) { return this[k]; }, data); // { foo: 0, bar: 1 }
```



## matches(obj, props):Boolean

Checks if object contains all properties/values pairs. Useful for validation
and filtering.

```js
var john = {age:25, hair:'long', beard:true};
var mark = {age:27, hair:'short', beard:false};
var hippie = {hair:'long', beard:true};
matches(john, hippie); // true
matches(mark, hippie); // false
```

See [`deepMatches()`](#deepMatches)



## merge(...objects):Object

Deep merges objects. Note that objects and properties will be cloned during the
process to avoid undesired side effects. It return a new object and won't
affect source objects.

```js
var obj1 = {a: {b: 1, c: 1, d: {e: 1, f: 1}}};
var obj2 = {a: {b: 2, d : {f : 'yeah'} }};

merge(obj1, obj2); // {a: {b : 2, c : 1, d : {e : 1, f : 'yeah'}}}
```

See: [`deepMixIn()`](#deepMixIn), [`deepFillIn()`](#deepFillIn)



## max(obj[, iterator]):*

Returns maximum value inside object or use a custom iterator to define how
items should be compared. Similar to [Array/max](array.html#max).

See: [`min()`](#min)

```js
max({a: 100, b: 2, c: 1, d: 3, e: 200}); // 200
max({a: 'foo', b: 'lorem', c: 'amet'}, function(val){
    return val.length;
}); // 'lorem'
```



## min(obj[, iterator]):*

Returns minimum value inside object or use a custom iterator to define how
items should be compared. Similar to [Array/min](array.html#min).

See: [`max()`](#max)

```js
min({a: 100, b: 2, c: 1, d: 3, e: 200}); // 1
min({a: 'foo', b: 'lorem', c: 'amet'}, function(val){
    return val.length;
}); // 'foo'
```



## mixIn(target, ...objects):Object

Combine properties from all the objects into first one.

This method affects target object in place, if you want to create a new Object
pass an empty object as first parameter.

### Arguments

 1. `target` (Object)        : Target Object.
 2. `...objects` (...Object) : Objects to be combined (0...n objects).

### Example

```js
var a = {foo: "bar"};
var b = {lorem: 123};

mixIn({}, a, b); // {foo: "bar", lorem: 123}
console.log(a);  // {foo: "bar"}

mixIn(a, b);     // {foo: "bar", lorem: 123}
console.log(a);  // {foo: "bar", lorem: 123}
```

See: [`fillIn()`](#fillIn), [`merge()`](#merge)




## namespace(obj, propName):Object

Creates an empty object inside namespace if not existent. Will return created
object or existing object.

See: [`get()`](#get), [`set()`](#set)

```js
var obj = {};
namespace(obj, 'foo.bar'); // {}
console.log(obj);          // {foo:{bar:{}}}
```


## omit(obj, ...keys):Object

Return a copy of the object without the blacklisted keys.

See: [`filter()`](#filter)

```js
var user = {
    firstName : 'John',
    lastName : 'Doe',
    dob : '1985/07/23',
    gender : 'male'
};

// can pass an array of keys as second argument
var keys = ['firstName', 'dob']
omit(user, keys); // {lastName : 'Doe', gender : 'male'}

// or multiple arguments
omit(user, 'firstName', 'lastName'); // {dob : '1985/07/23', gender : 'male'}
```



## pick(obj, ...keys):Object

Return a copy of the object that contains only the whitelisted keys.

See: [`filter()`](#filter)

```js
var user = {
    firstName : 'John',
    lastName : 'Doe',
    dob : '1985/07/23',
    gender : 'male'
};

// can pass an array of keys as second argument
var keys = ['firstName', 'dob']
pick(user, keys); // {firstName:"John", dob: "1985/07/23"}

// or multiple arguments
pick(user, 'firstName', 'lastName'); // {firstName:"John", lastName: "Doe"}
```



## pluck(obj, propName):Object

Extract an object containing property values with keys as they appear in the
passed object.

```js
var users = {
    first: {
        name : 'John',
        age : 21
    },
    second: {
        name : 'Mary',
        age : 25
    }
};

pluck(users, 'name'); // {first: 'John', second: 'Mary'} );
pluck(users, 'age');  // {first: 21, second: 25} );
```



## reduce(obj, callback, initial, [thisObj]):*

Similar to [Array/reduce](array.html#reduce).

Apply a function against an accumulator and each property of the object (order
is undefined) as to reduce it to a single value.

```js
var obj = {a: 1, b: 2, c: 3, d: 4};

function sum(prev, cur, key, list) {
    compare1.push(prev);
    return prev + cur;
}

reduce(obj, sum); // 10
```



## reject(obj, callback, thisObj):Object

Returns a new object containing all properties where `callback` returns true,
similar to [Array/reject](array.html#reject). It does not use properties from
the object's prototype. Opposite of [`filter()`](#filter).

See [`filter()`](#filter)

### Example

```js
var obj = {a: 1, b: 2, c: 3, d: 4, e: 5};
reject(obj, function(x) { return (x % 2) !== 0; }); // {b: 2, d: 4}
```



## values(obj):Array

Returns an array of all own enumerable properties values found upon a given object.

PS: it won't return properties from the prototype.

See: [`forOwn()`](#forOwn), [`keys()`](#keys)

```js
var obj = {
    foo : 1,
    bar : 2,
    lorem : 3
};
values(obj); // [1, 2, 3]
```



## set(obj, propName, value)

Sets a nested property value.

See: [`get()`](#get), [`namespace()`](#namespace)

```js
var obj = {};
set(obj, 'foo.bar', 123);
console.log(obj.foo.bar); // 123
console.log(obj);         // {foo:{bar:123}}
```



## size(obj):Number

Returns the count of own enumerable properties found upon a given object.

PS: it won't return properties from the prototype.

See: [`forOwn()`](#forOwn), [`keys()`](#keys)

```js
var obj = {
    foo : 1,
    bar : 2,
    lorem : 3
};
size(obj); // 3
```



## some(obj, callback, [thisObj]):Boolean

Similar to [Array/some](array.html#some). Tests whether any properties in the
object pass the test implemented by the provided callback.

```js
var obj = {
    a: 1,
    b: 2,
    c: 3,
    d: 'string'
};

some(obj, isNumber); // true
```



## unset(obj, propName):Boolean

Delete object property if existent and returns a boolean indicating succes. It
will also return `true` if property doesn't exist.

Some properties can't be deleted, to understand why [check this
article](http://perfectionkills.com/understanding-delete/).

See: [`set()`](#set)

```js
var lorem = {
        ipsum : {
            dolor : {
                sit : 'amet'
            }
        }
    };

unset(lorem, 'ipsum.dolor.sit'); // true
console.log(lorem.ipsum.dolor);  // {}
unset(lorem, 'foo.bar');         // true
```



## result(object, property):Mixed

Evaluates an objects property and returns result.

```js
var person = {
    name: 'john',

    mood: function() {
        // some dynamic calculated property.
        return 'happy';
    }
};

var name = result(person, 'name'), // john
    mood = result(person, 'mood'); // happy
```
