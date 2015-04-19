# collection #

Methods for dealing with collections (Array or Objects).



## contains(list, value):Boolean

Checks if collection contains value.

```js
contains({a: 1, b: 2, c: 'bar'}, 2); // true
contains([1, 2, 3], 'foo');  // false
```

See: [array/contains](array.html#contains), [object/contains](object.html#contains)



## every(list, callback, [thisObj]):Boolean

Tests whether all values in the collection pass the test implemented by the
provided callback.

```js
var obj = {
    a: 1,
    b: 2,
    c: 3,
    d: 'string'
};

every(obj, isNumber); // false
```

See: [array/every](array.html#every), [object/every](object.html#every)



## filter(list, callback, [thisObj]):Array

Filter collection properties.

See: [array/filter](array.html#filter), [object/filter](object.html#filter)



## find(list, callback, [thisObj]):*

Loops through all the values in the collection and returns the first one that
passes a truth test (callback).

**Important:** loop order over objects properties isn't guaranteed to be the
same on all environments.

```js
find({a: 'foo', b: 12}, isString); // 'foo'
find(['foo', 12], isNumber); // 12
```

See: [array/find](array.html#find), [object/find](object.html#find)



## forEach(list, callback, [thisObj])

Loop through all values of the collection.

See: [array/forEach](array.html#forEach), [object/forOwn](object.html#forOwn)



## map(list, callback, [thisObj]):Array

Returns a new collection where the properties values are the result of calling
the callback for each property in the original collection.

See: [array/map](array.html#map), [object/map](object.html#map)



## max(list, [iterator]):*

Returns maximum value inside collection or use a custom iterator to define how
items should be compared.

See: [`min()`](#min), [array/max](array.html#max), [object/max](object.html#max)

```js
max({a: 100, b: 2, c: 1, d: 3, e: 200}); // 200
max(['foo', 'lorem', 'amet'], function(val){
    return val.length;
}); // 'lorem'
```



## min(list, [iterator]):*

Returns minimum value inside collection or use a custom iterator to define how
items should be compared.

See: [`max()`](#max), [array/min](array.html#min), [object/min](object.html#min)

```js
min([10, 2, 7]); // 2
min({a: 'foo', b: 'lorem', c: 'amet'}, function(val){
    return val.length;
}); // 'foo'
```



## pluck(list, propName):Array

Extract a list of property values.

```js
var users = [
    {
        name : 'John',
        age : 21
    },
    {
        name : 'Jane',
        age : 27
    }
];

pluck(users, 'name'); // ["John", "Jane"]
pluck(users, 'age'); // [21, 27]

users = {
    first: {
        name : 'John',
        age : 21
    },
    second: {
        name : 'Mary',
        age : 25
    }
};

pluck(users, 'name'); // ['John', 'Mary']
```

See: [array/pluck](array.html#pluck), [object/pluck](object.html#pluck)



## reduce(list, callback, initial, [thisObj]):*

Apply a function against an accumulator and each value in the collection as to
reduce it to a single value.

```js
var obj = {a: 1, b: 2, c: 3, d: 4};

function sum(prev, cur, key, list) {
    return prev + cur;
}

reduce(obj, sum); // 10
```

See: [array/reduce](array.html#reduce), [object/reduce](object.html#reduce)



## reject(list, fn, [thisObj]):Array

Creates a new array with all the elements that do **not** pass the truth test.
Opposite of [`filter()`](#filter).

### Example

```js
var numbers = [1, 2, 3, 4, 5];
reject(numbers, function(x) { return (x % 2) !== 0; }); // [2, 4]

var obj = {a: 1, b: 2, c: 3, d: 4, e: 5};
reject(obj, function(x) { return (x % 2) !== 0; }); // [2, 4]
```

See: [array/reject](array.html#reject), [object/reject](object.html#reject)



## size(list):Number

Returns the number of values in the collection.

```js
var obj = {
    foo : 1,
    bar : 2,
    lorem : 3
};
size(obj);     // 3
size([1,2,3]); // 3
size(null);    // 0
```

See: [object/size](object.html#size)



## some(list, callback, [thisObj]):Boolean

Tests whether any values in the collection pass the test implemented by the
provided callback.

```js
var obj = {
    a: 1,
    b: 2,
    c: 3,
    d: 'string'
};

some(obj, isNumber);      // true
some(obj, isString);      // true
some([1, 2, 3], isNumber) // true
some([1, 2, 3], isString) // false
```

See: [array/some](array.html#some), [object/some](object.html#some)


-------------------------------------------------------------------------------

For more usage examples check specs inside `/tests` folder. Unit tests are the
best documentation you can get...
