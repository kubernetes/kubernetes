[![Build Status](https://travis-ci.org/wk8/go-ordered-map.svg?branch=master)](https://travis-ci.org/wk8/go-ordered-map)

# Goland Ordered Maps

Same as regular maps, but also remembers the order in which keys were inserted, akin to [Python's `collections.OrderedDict`s](https://docs.python.org/3.7/library/collections.html#ordereddict-objects).

It offers the following features:
* optimal runtime performance (all operations are constant time)
* optimal memory usage (only one copy of values, no unnecessary memory allocation)
* allows iterating from newest or oldest keys indifferently, without memory copy, allowing to `break` the iteration, and in time linear to the number of keys iterated over rather than the total length of the ordered map
* takes and returns generic `interface{}`s
* idiomatic API, akin to that of [`container/list`](https://golang.org/pkg/container/list)

## Installation
```bash
go get -u github.com/wk8/go-ordered-map
```

Or use your favorite golang vendoring tool!

## Documentation

[The full documentation is available on godoc.org](https://godoc.org/github.com/wk8/go-ordered-map).

## Example / usage

```go
package main

import (
	"fmt"

	"github.com/wk8/go-ordered-map"
)

func main() {
	om := orderedmap.New()

	om.Set("foo", "bar")
	om.Set("bar", "baz")
	om.Set("coucou", "toi")

	fmt.Println(om.Get("foo"))          // => bar, true
	fmt.Println(om.Get("i dont exist")) // => <nil>, false

	// iterating pairs from oldest to newest:
	for pair := om.Oldest(); pair != nil; pair = pair.Next() {
		fmt.Printf("%s => %s\n", pair.Key, pair.Value)
	} // prints:
	// foo => bar
	// bar => baz
	// coucou => toi

	// iterating over the 2 newest pairs:
	i := 0
	for pair := om.Newest(); pair != nil; pair = pair.Prev() {
		fmt.Printf("%s => %s\n", pair.Key, pair.Value)
		i++
		if i >= 2 {
			break
		}
	} // prints:
	// coucou => toi
	// bar => baz
}
```

All of `OrderedMap`'s methods accept and return `interface{}`s, so you can use any type of keys that regular `map`s accept, as well pack/unpack arbitrary values, e.g.:
```go
type myStruct struct {
	payload string
}

func main() {
	om := orderedmap.New()

	om.Set(12, &myStruct{"foo"})
	om.Set(1, &myStruct{"bar"})

	value, present := om.Get(12)
	if !present {
		panic("should be there!")
	}
	fmt.Println(value.(*myStruct).payload) // => foo

	for pair := om.Oldest(); pair != nil; pair = pair.Next() {
		fmt.Printf("%d => %s\n", pair.Key, pair.Value.(*myStruct).payload)
	} // prints:
	// 12 => foo
	// 1 => bar
}
```

## Alternatives

There are several other ordered map golang implementations out there, but I believe that at the time of writing none of them offer the same functionality as this library; more specifically:
* [iancoleman/orderedmap](https://github.com/iancoleman/orderedmap) only accepts `string` keys, its `Delete` operations are linear
* [cevaris/ordered_map](https://github.com/cevaris/ordered_map) uses a channel for iterations, and leaks goroutines if the iteration is interrupted before fully traversing the map
* [mantyr/iterator](https://github.com/mantyr/iterator) also uses a channel for iterations, and its `Delete` operations are linear
* [samdolan/go-ordered-map](https://github.com/samdolan/go-ordered-map) adds unnecessary locking (users should add their own locking instead if they need it), its `Delete` and `Get` operations are linear, iterations trigger a linear memory allocation
