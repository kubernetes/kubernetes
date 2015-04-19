# array #

Array utilities.




## append(arr1, arr2):Array

Appends an array to the end of the other.
The first array will be modified and will contain the appended items.

See: [`union()`](#union), [`combine()`](#combine)

```js
var foo = ['a', 'b'],
    bar = ['b', 'd'];

append(foo, bar); // ['a', 'b', 'b', 'd']
```



## collect(arr, callback, [thisObj]):Array

Maps the items in `arr` and concatenates the resulting arrays.

See: [`map()`](#map)

```js
collect([1, 2, 3], function(val) {
    return [val, val % 2];
}); // [1, 1, 2, 0, 3, 1];

collect(['a', 'bb', ''], function(val) {
    return val.split('');
}); // ['a', 'b', 'b']
```

It also supports a shorthand syntax:

```js
var items = [{ a: [1] }, { b: 'foo' }, { a: [2, 3] }];
collect(items, 'a'); // [1, 2, 3];
```



## combine(arr1, arr2):Array

Combines an array with all the items of another.
The first array will be modified and will contain the combined items.
Does not allow duplicates and is case and type sensitive.

See: [`union()`](#union), [`append()`](#append)

```js
var foo = ['a', 'b'],
    bar = ['b', 'd'];

combine(foo, bar); // ['a', 'b', 'd']
```



## compact(arr):Array

Returns a new Array without any `null` or `undefined` values. Note that it will
keep empty strings and other falsy values (simillar to Ruby Array#compact).

```js
var arr = [0, 1, null, false, '', 'foo', undefined, 'bar'];
compact(arr); // [0, 1, false, '', 'foo', 'bar'];
```



## contains(arr, value):Boolean

Checks if Array contains value. Alias to `indexOf(arr, val) !== -1`.

```js
var arr = [1, 2, 3];
contains(arr, 2);      // true
contains(arr, 'foo');  // false
```



## difference(...arrs):Array

Return a new Array with elements that aren't present in the other Arrays
besides the first one.

Works like [Python set#difference](http://docs.python.org/library/stdtypes.html#set.difference).

It will remove duplicates.

See: [`xor()`](#xor), [`intersection()`](#intersection)

```js
var a = ['a', 'b', 1];
var b = ['c', 1];
difference(a, b); // ['a', 'b']
```



## every(arr, callback, [thisObj]):Array

Crossbrowser `Array.every()`.

Tests whether all elements in the array pass the test implemented by the provided function.

It differs from ES5 since it will also loop over sparse items in the array to
normalize the behavior across browsers (avoid inconsistencies).

```js
var items = [1, 'foo', 'bar'];
every(items, isString);   // false
every(items, isFunction); // false
every(items, function(val, key, arr){
    return val != null;
}); // true
```

more info at [MDN Array#every](https://developer.mozilla.org/en/JavaScript/Reference/Global_Objects/Array/every)

It also supports a shorthand syntax:

```js
var items = [{id:1, active:true}, {id:3, active:true}, {id:8, active:true}];
// all items with `id === 8`
every(items, {id:8}); // false
// `active` is truthy on all items
every(items, 'active'); // true
```



## filter(arr, callback, [thisObj]):Array

Crossbrowser `Array.filter()`.

Creates a new array with all elements that pass the callback test.

It differs from ES5 since it will also loop over sparse items in the array to
normalize the behavior across browsers (avoid inconsistencies).

```js
var nums = [1, 2, 3, 4, 5, 6];
var oddNumbers = filter(nums, function(val, key, arr){
    return (val % 2) !== 0;
});
// > [1, 3, 5]
```

more info at [MDN Array#filter](https://developer.mozilla.org/en/JavaScript/Reference/Global_Objects/Array/filter)

Filter also supports shorthand notation:

```js
var users = [
    {name:'john', surname:'connor', beard:false},
    {name:'john', surname:'doe', beard:true},
    {name:'jane', surname:'doe', beard:false}
];
// filter item that matches all properties/values pairs
filter(arr, {name:'john', beard:false});
// > [{name:'john', surnname:'connor', beard:false}]
// items where 'beard' is a truthy value
filter(arr, 'beard');
// > [{name:'john', surnname:'doe', beard:true}]
```

See [`reject()`](#reject)



## find(arr, callback, [thisObj]):*

Loops through all the items in the Array and returns the first one that passes
a truth test (callback).

```js
var arr = [123, {a:'b'}, 'foo', 'bar'];
find(arr, isString); // "foo"
find(arr, isNumber); // 123
find(arr, isObject); // {a:'b'}
```

Find also supports shorthand notation:

```js
var users = [
    {name:'john', surname:'connor', beard:false},
    {name:'john', surname:'doe', beard:true}
];
// first item that matches all properties/values pairs
find(arr, {name:'john'}); // {name:'john', surnname:'connor', beard:false}
// first item where 'beard' is a truthy value
find(arr, 'beard'); // {name:'john', surnname:'doe', beard:true}
```

See: [findIndex()](#findIndex), [findLast()](#findLast),
[findLastIndex()](#findLastIndex)



## findLast(arr, callback, [thisObj]):*

Loops through all the items in the Array (starting from last item) and returns
the first one that passes a truth test (callback).

```js
var arr = [123, {a:'b'}, 'foo', 'bar'];
findLast(arr, isString); // "bar"
findLast(arr, isNumber); // 123
findLast(arr, isObject); // {a:'b'}
```

`findLast` also supports shorthand notation:

```js
var users = [
    {name:'john', surname:'connor', beard:false},
    {name:'john', surname:'doe', beard:true}
];
// last item that matches all properties/values pairs
findLast(arr, {name:'john'}); // {name:'john', surnname:'doe', beard:true}
// last item where 'beard' is a truthy value
findLast(arr, 'beard'); // {name:'john', surnname:'doe', beard:true}
```

See: [find()](#find), [findIndex()](#findIndex),
[findLastIndex()](#findLastIndex)



## findIndex(arr, iterator, [thisObj]):Number

Loops through the items in the Array and returns the index of the first one
that passes a truth test (callback).

Returns `-1` if no item was found that passes the truth test.

```js
var arr = [1, { a: 1 }, 'foo', 'bar'];
findIndex(arr, isString); // 2
findIndex(arr, isNumber); // 0
findIndex(arr, isObject); // 1
findIndex(arr, isRegExp); // -1
```

`findIndex` also supports shorthand notation:

```js
var pets = [
    { pet: 'dog', name: 'Sam' },
    { pet: 'dog', name: 'Maggie' }
];

findIndex(pets, { pet: 'dog' }); // 0
findIndex(pets, { name: 'Maggie' }); // 1
```

See: [find()](#find), [findLastIndex()](#findLastIndex)



## findLastIndex(arr, iterator, [thisObj]):Number

Loops through the items in the Array on the reverse order and returns the index
of the first one that passes a truth test (callback).

Returns `-1` if no item was found that passes the truth test.

```js
var arr = [1, { a: 1 }, 'foo', 'bar'];
findLastIndex(arr, isString); // 3
findLastIndex(arr, isNumber); // 0
findLastIndex(arr, isObject); // 1
findLastIndex(arr, isRegExp); // -1
```

`findLastndex` also supports shorthand notation:

```js
var pets = [
    { pet: 'dog', name: 'Sam' },
    { pet: 'dog', name: 'Maggie' }
];

findLastIndex(pets, { pet: 'dog' }); // 1
findLastIndex(pets, { name: 'Sam' }); // 0
```

See: [find()](#find), [findIndex()](#findIndex)



## flatten(arr, [level]):Array

Recursively flattens an array. A new array containing all the elements is
returned. If `level` is specified, it will only flatten up to that level.

### Example

```js
flatten([1, [2], [3, [4, 5]]]);
// > [1, 2, 3, 4, 5]
flatten([1, [2], [3, [4, 5]]], 1);
// > [1, 2, 3, [4, 5]]
```



## forEach(arr, callback, [thisObj]):void

Crossbrowser `Array.forEach()`.

It allows exiting the iteration early by returning `false` on the callback.

It differs from ES5 since it will also loop over sparse items in the array to
normalize the behavior across browsers (avoid inconsistencies).

```js
var items = ['foo', 'bar', 'lorem', 'ipsum'];
forEach(items, function(val, key, arr){
    console.log(key +' : '+ val);
    if (val === 'lorem') {
        // stop iteration (break)
        return false;
    }
});
```

more info at [MDN Array#forEach](https://developer.mozilla.org/en/JavaScript/Reference/Global_Objects/Array/forEach)



## indexOf(arr, item, [fromIndex]):Number

Crossbrowser `Array.indexOf()`.

It differs from ES5 since it will also loop over sparse items in the array to
normalize the behavior across browsers (avoid inconsistencies).

more info at [MDN Array#indexOf](https://developer.mozilla.org/en/JavaScript/Reference/Global_Objects/Array/indexOf)




## insert(arr, ...items):Number

Push items into array only if they aren't contained by it. Returns the new
`length` of the array.

See: [`remove()`](#remove), [`removeAll()`](#removeAll),
[`contains()`](#contains)

```js
var arr = ['a', 'b'];
insert(arr, 'a');       // 2 : ['a', 'b']
insert(arr, 'c');       // 3 : ['a', 'b', 'c']
insert(arr, 1, 2, 'b'); // 5 : ['a', 'b', 'c', 1, 2]
```



## intersection(...arrs):Array

Return a new Array with elements common to all Arrays.

Similar to Python set#intersection and underscore.js intersection.

It will remove duplicates.

See: [`difference()`](#difference), [`xor()`](#xor)

```js
var a = ['a', 'b', 1],
    b = ['c', 1],
    c = [1, 2, 3];
intersection(a, b, c); // [1]
```



## invoke(arr, methodName[, ...args]):Array

Call `methodName` on each item of the array passing custom arguments if needed.

```js
invoke([[3,2,1], [9,5,2]], 'sort'); // [[1,2,3], [2,5,9]]
```



## join(arr, [separator]):String

Joins the strings in `arr`, inserting `separator` between each value.

This ignores null values and empty strings that are in the array. `separator`
defaults to an empty string. This will convert all non-string objects in the
array to a string.

### Example

```js
join(['a', 'b', 'c']); // 'abc'
join(['foo', 'bar'], ', '); // 'foo, bar'
join([null, 'foo', '', 'bar', undefined], ':'); // 'foo:bar'
```



## lastIndexOf(arr, item, [fromIndex]):Number

Crossbrowser `Array.lastIndexOf()`.

It differs from ES5 since it will also loop over sparse items in the array to
normalize the behavior across browsers (avoid inconsistencies).

more info at [MDN Array#lastIndexOf](https://developer.mozilla.org/en/JavaScript/Reference/Global_Objects/Array/lastIndexOf)



## map(arr, callback, [thisObj]]):Array

Crossbrowser `Array.map()`.

Creates a new array with the results of calling a provided function on every
element in this array.

It differs from ES5 since it will also loop over sparse items in the array to
normalize the behavior across browsers (avoid inconsistencies).

See: [`collect()`](#collect)

```js
var nums = [1,2,3,4];
var double = map(nums, function(val, key, arr){
    return val * 2;
});
// > [2, 4, 6, 8]
```

more info at [MDN Array#map](https://developer.mozilla.org/en/JavaScript/Reference/Global_Objects/Array/map)

It also supports a shorthand notation which can be used to achieve same result
as [`array/pluck`](#pluck):

```js
var src = ['lorem', 'ipsum', 'foo', 'amet'];
// grab the "length" property of all items
var lengths = map(src, 'length'); // [5, 5, 3, 4]
```



## max(arr, [iterator], [thisObj]):*

Returns maximum value inside array or use a custom iterator to define how items
should be compared.

See: [`min()`](#min)

```js
max([10, 2, 7]); // 10
max(['foo', 'lorem', 'amet'], function(val){
    return val.length;
}); // 'lorem'
```

It also supports a shorthand notation:

```js
max(['foo', 'lorem', 'amet'], 'length'); // "lorem"
```



## min(arr, [iterator], [thisObj]):*

Returns minimum value inside array or use a custom iterator to define how items
should be compared.

See: [`max()`](#max)

```js
min([10, 2, 7]); // 2
min(['foo', 'lorem', 'amet'], function(val){
    return val.length;
}); // 'foo'
```

It also supports a shorthand notation:

```js
min(['foo', 'lorem', 'amet'], 'length'); // "foo"
```



## pick(arr, [nItems]):*

Gets random item(s) and removes it from the original array.

If `nItems` is specified it will return a new Array contained the *picked*
items otherwise it returns a single item.

See: [`random/choice()`](./random.html#choice)

### Example:

```js
var arr = [1, 2, 3, 4, 5, 6];
var item1 = pick(arr); // 4
var item2 = pick(arr); // 1
var items = pick(arr, 2); // [5, 2]
console.log(arr); // [3, 6]
```



## pluck(arr, propName):Array

Extract a list of property values.

See: [`function/prop()`](function.html#prop)

```js
var users = [{name : 'John', age: 21}, {name: 'Jane', age : 27}];
var names = pluck(users, 'name'); // ["John", "Jane"]
var ages = pluck(users, 'age'); // [21, 27]
```



## range([start], stop[, step]):Array

Creates a new Array with all the values inside the range. Works similarly to
Python#range or PHP#range.

### Arguments

 1. `[start]` (Number) : Range start. Default is `0`.
 2. `stop` (Number) : Range limit.
 3. `[step]` (Number) : Step size. Default is `1`.

### Example

```js
range(5);         // [0, 1, 2, 3, 4, 5]
range(0, 5);      // [0, 1, 2, 3, 4, 5]
range(0, 5, 2);   // [0, 2, 4]
range(20, 40, 5); // [20, 25, 30, 35, 40]
```



## reduce(arr, fn):*

Crossbrowser `Array.reduce()`.

Apply a function against an accumulator and each value of the array (from
left-to-right) as to reduce it to a single value.

It differs from ES5 since it will also loop over sparse items in the array to
normalize the behavior across browsers (avoid inconsistencies).

more info at [MDN Array#reduce](https://developer.mozilla.org/en/JavaScript/Reference/Global_Objects/Array/reduce)



## reduceRight(arr, fn):*

Crossbrowser `Array.reduceRight()`.

Apply a function simultaneously against two values of the array (from
right-to-left) as to reduce it to a single value.

It differs from ES5 since it will also loop over sparse items in the array to
normalize the behavior across browsers (avoid inconsistencies).

more info at [MDN Array#reduceRight](https://developer.mozilla.org/en/JavaScript/Reference/Global_Objects/Array/reduceRight)



## reject(arr, fn, thisObj):Array

Creates a new array with all the elements that do **not** pass the truth test.
Opposite of [`filter()`](#filter).

See [`filter()`](#filter)

### Example

```js
var numbers = [1, 2, 3, 4, 5, 6];
reject(numbers, function(x) { return (x % 2) !== 0; }); // [2, 4, 6]
```

It also supports a shorthand syntax:

```js
var users = [
    {name:'john', surname:'connor', beard:false},
    {name:'john', surname:'doe', beard:true},
    {name:'jane', surname:'doe', beard:false}
];
// reject items that matches all properties/values pairs
reject(arr, {name:'john'});
// > [{name:'jane', surnname:'doe', beard:false}]
// reject items where 'beard' is a truthy value
filter(arr, 'beard');
// > [{name:'john', surnname:'connor', beard:false},
//    {name:'jane', surname:'doe', beard:false}]
```



## remove(arr, item):void

Remove a single item from the array.

IMPORTANT: it won't remove duplicates, just a single item.

### Example

```js
var foo = [1, 2, 3, 4];
remove(foo, 2);
console.log(foo); // [1, 3, 4]
```



## removeAll(arr, item):void

Remove all instances of an item from the array.

### Example

```js
var foo = [1, 2, 3, 4, 2, 2];
removeAll(foo, 2);
console.log(foo); // [1, 3, 4];
```



## shuffle(arr):Array

Returns a new Array with items randomly sorted (shuffled). Similar to Ruby Array#shuffle.

### Example

```js
var arr = ['a', 'b', 'c', 'd', 'e'];
shuffle(arr); // ['b', 'd', 'e', 'c', 'a']
```



## slice(arr, [start], [end]):Array

Returns a new array containing the items from `arr` from the start index to the
end index.

If `start` is omitted, it will start at `0`. If `end` is omitted, it will used
the last index of the array.  If `start` or `end` is negative, it is used as an
offset from the end of the array.

It will also convert array-like objects to arrays.

### Example

```js
slice([1, 2, 3, 4], 1, 2); // [2, 3]
slice([1, 2, 3], 1); // [2, 3]
slice([1, 2, 3]); // [1, 2, 3]
slice({ length: 2, 0: 'a', 1: 'b' }); // ['a', 'b']
slice([1, 2, 3], 0, -1); // [1, 2]
slice([1, 2, 3], -2); // [2, 3]
```



## some(arr, callback, [thisObj]):Array

Crossbrowser `Array.some()`.

Tests whether some element in the array passes the test implemented by the provided function.

It differs from ES5 since it will also loop over sparse items in the array to
normalize the behavior across browsers (avoid inconsistencies).

```js
var items = [1, 'foo', 'bar'];
some(items, isString);   // true
some(items, isFunction); // false
```

more info at [MDN Array#some](https://developer.mozilla.org/en/JavaScript/Reference/Global_Objects/Array/some)

It also supports a shorthand syntax:

```js
var items = [{id:1, active:true}, {id:3, active:false}, {id:8, active:false}];
// at least one item with `id === 8`
some(items, {id:8}); // true
// `active` is truthy on at least one item
some(items, 'active'); // true
```

see also: [`object/matches`](object.html#matches)



## sort(arr, [compareFn]):Array

Returns a sorted Array using the [Merge Sort](http://en.wikipedia.org/wiki/Merge_sort) algorithm (stable sort).

The `Array.prototype.sort` browser implementations differ since the sorting algorithm isn't described in the ES spec - [in V8 it isn't stable](http://code.google.com/p/v8/issues/detail?id=90) and [on Firefox it is stable](https://bugzilla.mozilla.org/show_bug.cgi?id=224128) - so this function doesn't use the browser native implementation and is recommended in cases where a stable sort is required (items should preserve same order if already sorted).

**Important:** It does logical comparisson by default (greater/less than) and
not a string comparisson like the native `Array#sort`.

### compareFn

If `compareFn` is supplied elements are sorted based on the value returned by
the `compareFn`.

 - If `compareFn(a, b)` is less than `0`, sort `a` to a lower index than `b`.
 - If `compareFn(a, b)` returns `0`, leave `a` and `b` unchanged with respect
   to each other, but sorted with respect to all different elements.
 - If `compareFn(a, b)` is greater than `0`, sort `b` to a lower index than
   `a`.

See: [`sortBy`](#sortBy)

### Example

```js
sort([187, 23, 47, 987, 12, 59, 0]); // [0, 12, 23, 47, 59, 187, 987]
sort(['a', 'z', 'c', 'beta', 'b']); // ['a', 'b', 'beta', 'c', 'z']

// ['sit', 'amet', 'lorem', 'ipsum']
sort(['lorem', 'ipsum', 'sit', 'amet'], function(a, b){
    // sort by length, items with same length
    // will keep the relative order (stable)
    return a.length - b.length;
});

// [4, 3, 2, 1]
sort([2, 3, 1, 4], function(a, b){
    // reverse sort
    return b - a;
});
```



## sortBy(arr, callback, [context]):Array

Returns an array sorted by the result of the callback.

The callback is called for each item that is to be sorted, and the
results of the callback are used to sort the array. The callback
is called with the item as the first parameter, optionally with
the provided context.

It also supports a shorthand notation which can be used to sort by a property
name.

See: [`sort`](#sort)

```js
// Returns [{ a: 1 }, { a: 2 }, { a: 3 }]
sortBy([{ a: 1 }, { a: 3 }, { a: 2 }],
    function(item) { return item.a; });

// Same as above, using shorthand notation
sortBy([{ a: 1 }, { a: 3 }, { a: 2 }], 'a');
```



## split(arr, [segments]):Array

Splits an array into a fixed number of segments.

The number of segments is specified by `segments` and defaults to 2. If the
array cannot be evenly split, the first segments will contain the extra items.
If `arr` is empty, an empty array is returned. If `arr.length` is less than
`segments`, then the resulting array will have `arr.length` number of
single-element arrays.

### Example
```js
split([1, 2, 3, 4, 5], 3) // [ [1, 2], [3, 4], [5] ]
split([1, 2, 3, 4, 5]) // [ [1, 2, 3], [4, 5] ]
split([]) // []
split([1, 2], 3) // [ [1], [2] ]
```



## toLookup(arr, key):Object

Create an object that indexes the items in the array by a key. If `key` is a function, the key for each value in the resulting object will be the result of calling the function with the value as an argument. Otherwise `key` specifies the property on each value to use as the key.

### Example

```js
var foo = [{ name: 'a', thing: 1 }, { name: 'b', thing: 2 }];
// { a: { name: 'a', thing: 1 }, b: { name: 'b', thing: 2 } }
toLookup(foo, 'name');
// same as above
toLookup(foo, function (value) { return value.name; });
```



## union(...arrs):Array

Concat multiple arrays removing duplicates.

```js
var a = ['a', 'b'],
    b = ['c', 'a'],
    c = [1, 'b', 2, 3, 'a'];

//note that unique remove from begin to end
union(a, b, c); // ['c', 1, 'b', 2, 3, 'a']
```



## unique(arr, [compare]):Array

Return a new Array of unique items.

**IMPORTANT:** duplicates are removed starting from begining of array.

```js
var arr = [1, 2, 3, 4, 2, 2, 4];
var foo = unique(arr);
console.log(foo);
// > [1, 3, 2, 4];

// you also have the option to set a custom compare function
var users = [{name: 'john'}, {name: 'paul'}, {name: 'john'}];
var uniqueNames = unique(arr, function(a, b){
    return a.name === b.name;
});
console.log(uniqueNames);
// > [{name: 'paul'}, {name: 'john'}]
```



## xor(arr1, arr2):Array

Exclusive OR. Returns items that are present in a single array.

Works like [Python set#symmetric_difference](http://docs.python.org/library/stdtypes.html#set.symmetric_difference) renamed for brevity.

It will remove duplicates.

See: [`difference()`](#difference), [`intersection()`](#intersection)

```js
var a = ['a', 'b', 1];
var b = ['c', 1];
xor(a, b); // ['a', 'b', 'c']
```



## zip(...arrs):Array

Groups the elements of each array at their corresponding indexes.

Useful for separate data sources that are coordinated through matching array
indexes. For a matrix of nested arrays, `zip.apply(...)` can transpose the
matrix in a similar fashion.

```js
// [['moe', 30, true], ['larry', 40, false], ['curly', 50, false]]
zip(['moe', 'larry', 'curly'], [30, 40, 50], [true, false, false]);
```




-------------------------------------------------------------------------------

For more usage examples check specs inside `/tests` folder. Unit tests are the
best documentation you can get...

