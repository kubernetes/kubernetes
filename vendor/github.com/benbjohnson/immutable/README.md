Immutable ![release](https://img.shields.io/github/release/benbjohnson/immutable.svg) ![test](https://github.com/benbjohnson/immutable/workflows/test/badge.svg) ![coverage](https://img.shields.io/codecov/c/github/benbjohnson/immutable/master.svg) ![license](https://img.shields.io/github/license/benbjohnson/immutable.svg)
=========

This repository contains *generic* immutable collection types for Go. It includes
`List`, `Map`, `SortedMap`, `Set` and `SortedSet` implementations. Immutable collections can
provide efficient, lock free sharing of data by requiring that edits to the
collections return new collections.

The collection types in this library are meant to mimic Go built-in collections
such as`slice` and `map`. The primary usage difference between Go collections
and `immutable` collections is that `immutable` collections always return a new
collection on mutation so you will need to save the new reference.

Immutable collections are not for every situation, however, as they can incur
additional CPU and memory overhead. Please evaluate the cost/benefit for your
particular project.

Special thanks to the [Immutable.js](https://immutable-js.github.io/immutable-js/)
team as the `List` & `Map` implementations are loose ports from that project.


## List

The `List` type represents a sorted, indexed collection of values and operates
similarly to a Go slice. It supports efficient append, prepend, update, and
slice operations.


### Adding list elements

Elements can be added to the end of the list with the `Append()` method or added
to the beginning of the list with the `Prepend()` method. Unlike Go slices,
prepending is as efficient as appending.

```go
// Create a list with 3 elements.
l := immutable.NewList[string]()
l = l.Append("foo")
l = l.Append("bar")
l = l.Prepend("baz")

fmt.Println(l.Len())  // 3
fmt.Println(l.Get(0)) // "baz"
fmt.Println(l.Get(1)) // "foo"
fmt.Println(l.Get(2)) // "bar"
```

Note that each change to the list results in a new list being created. These
lists are all snapshots at that point in time and cannot be changed so they
are safe to share between multiple goroutines.

### Updating list elements

You can also overwrite existing elements by using the `Set()` method. In the
following example, we'll update the third element in our list and return the
new list to a new variable. You can see that our old `l` variable retains a
snapshot of the original value.

```go
l := immutable.NewList[string]()
l = l.Append("foo")
l = l.Append("bar")
newList := l.Set(2, "baz")

fmt.Println(l.Get(1))       // "bar"
fmt.Println(newList.Get(1)) // "baz"
```

### Deriving sublists

You can create a sublist by using the `Slice()` method. This method works with
the same rules as subslicing a Go slice:

```go
l = l.Slice(0, 2)

fmt.Println(l.Len())  // 2
fmt.Println(l.Get(0)) // "baz"
fmt.Println(l.Get(1)) // "foo"
```

Please note that since `List` follows the same rules as slices, it will panic if
you try to `Get()`, `Set()`, or `Slice()` with indexes that are outside of
the range of the `List`.



### Iterating lists

Iterators provide a clean, simple way to iterate over the elements of the list
in order. This is more efficient than simply calling `Get()` for each index.

Below is an example of iterating over all elements of our list from above:

```go
itr := l.Iterator()
for !itr.Done() {
	index, value, _ := itr.Next()
	fmt.Printf("Index %d equals %v\n", index, value)
}

// Index 0 equals baz
// Index 1 equals foo
```

By default iterators start from index zero, however, the `Seek()` method can be
used to jump to a given index.


### Efficiently building lists

If you are building large lists, it is significantly more efficient to use the
`ListBuilder`. It uses nearly the same API as `List` except that it updates
a list in-place until you are ready to use it. This can improve bulk list
building by 10x or more.

```go
b := immutable.NewListBuilder[string]()
b.Append("foo")
b.Append("bar")
b.Set(2, "baz")

l := b.List()
fmt.Println(l.Get(0)) // "foo"
fmt.Println(l.Get(1)) // "baz"
```

Builders are invalid after the call to `List()`.


## Map

The `Map` represents an associative array that maps unique keys to values. It
is implemented to act similarly to the built-in Go `map` type. It is implemented
as a [Hash-Array Mapped Trie](https://lampwww.epfl.ch/papers/idealhashtrees.pdf).

Maps require a `Hasher` to hash keys and check for equality. There are built-in
hasher implementations for most primitive types such as `int`, `uint`, and
`string` keys. You may pass in a `nil` hasher to `NewMap()` if you are using
one of these key types.

### Setting map key/value pairs

You can add a key/value pair to the map by using the `Set()` method. It will
add the key if it does not exist or it will overwrite the value for the key if
it does exist.

Values may be fetched for a key using the `Get()` method. This method returns
the value as well as a flag indicating if the key existed. The flag is useful
to check if a `nil` value was set for a key versus a key did not exist.

```go
m := immutable.NewMap[string,int](nil)
m = m.Set("jane", 100)
m = m.Set("susy", 200)
m = m.Set("jane", 300) // overwrite

fmt.Println(m.Len())   // 2

v, ok := m.Get("jane")
fmt.Println(v, ok)     // 300 true

v, ok = m.Get("susy")
fmt.Println(v, ok)     // 200, true

v, ok = m.Get("john")
fmt.Println(v, ok)     // nil, false
```


### Removing map keys

Keys may be removed from the map by using the `Delete()` method. If the key does
not exist then the original map is returned instead of a new one.

```go
m := immutable.NewMap[string,int](nil)
m = m.Set("jane", 100)
m = m.Delete("jane")

fmt.Println(m.Len())   // 0

v, ok := m.Get("jane")
fmt.Println(v, ok)     // nil false
```


### Iterating maps

Maps are unsorted, however, iterators can be used to loop over all key/value
pairs in the collection. Unlike Go maps, iterators are deterministic when
iterating over key/value pairs.

```go
m := immutable.NewMap[string,int](nil)
m = m.Set("jane", 100)
m = m.Set("susy", 200)

itr := m.Iterator()
for !itr.Done() {
	k, v := itr.Next()
	fmt.Println(k, v)
}

// susy 200
// jane 100
```

Note that you should not rely on two maps with the same key/value pairs to
iterate in the same order. Ordering can be insertion order dependent when two
keys generate the same hash.


### Efficiently building maps

If you are executing multiple mutations on a map, it can be much more efficient
to use the `MapBuilder`. It uses nearly the same API as `Map` except that it
updates a map in-place until you are ready to use it.

```go
b := immutable.NewMapBuilder[string,int](nil)
b.Set("foo", 100)
b.Set("bar", 200)
b.Set("foo", 300)

m := b.Map()
fmt.Println(m.Get("foo")) // "300"
fmt.Println(m.Get("bar")) // "200"
```

Builders are invalid after the call to `Map()`.


### Implementing a custom Hasher

If you need to use a key type besides `int`, `uint`, or `string` then you'll
need to create a custom `Hasher` implementation and pass it to `NewMap()` on
creation.

Hashers are fairly simple. They only need to generate hashes for a given key
and check equality given two keys.

```go
type Hasher[K any] interface {
	Hash(key K) uint32
	Equal(a, b K) bool
}
```

Please see the internal `intHasher`, `uintHasher`, `stringHasher`, and
`byteSliceHasher` for examples.


## Sorted Map

The `SortedMap` represents an associative array that maps unique keys to values.
Unlike the `Map`, however, keys can be iterated over in-order. It is implemented
as a B+tree.

Sorted maps require a `Comparer` to sort keys and check for equality. There are
built-in comparer implementations for `int`, `uint`, and `string` keys. You may
pass a `nil` comparer to `NewSortedMap()` if you are using one of these key
types.

The API is identical to the `Map` implementation. The sorted map also has a
companion `SortedMapBuilder` for more efficiently building maps.


### Implementing a custom Comparer

If you need to use a key type besides `int`, `uint`, or `string` or derived types, then you'll
need to create a custom `Comparer` implementation and pass it to
`NewSortedMap()` on creation.

Comparers on have one method—`Compare()`. It works the same as the
`strings.Compare()` function. It returns `-1` if `a` is less than `b`, returns
`1` if a is greater than `b`, and returns `0` if `a` is equal to `b`.

```go
type Comparer[K any] interface {
	Compare(a, b K) int
}
```

Please see the internal `defaultComparer` for an example, bearing in mind that it works for several types.

## Set

The `Set` represents a collection of unique values, and it is implemented as a
wrapper around a `Map[T, struct{}]`.

Like Maps, Sets require a `Hasher` to hash keys and check for equality. There are built-in
hasher implementations for most primitive types such as `int`, `uint`, and
`string` keys. You may pass in a `nil` hasher to `NewMap()` if you are using
one of these key types.


## Sorted Set

The `SortedSet` represents a sorted collection of unique values.
Unlike the `Set`, however, keys can be iterated over in-order. It is implemented
as a B+tree.

Sorted sets require a `Comparer` to sort values and check for equality. There are
built-in comparer implementations for `int`, `uint`, and `string` keys. You may
pass a `nil` comparer to `NewSortedSet()` if you are using one of these key
types.

The API is identical to the `Set` implementation.


## Contributing

The goal of `immutable` is to provide stable, reasonably performant, immutable
collections library for Go that has a simple, idiomatic API. As such, additional
features and minor performance improvements will generally not be accepted. If
you have a suggestion for a clearer API or substantial performance improvement,
_please_ open an issue first to discuss. All pull requests without a related
issue will be closed immediately.

Please submit issues relating to bugs & documentation improvements.

