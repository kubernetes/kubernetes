// Copyright (c) 2015, Emir Pasic. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package containers

// EnumerableWithIndex provides functions for ordered containers whose values can be fetched by an index.
type EnumerableWithIndex interface {
	// Each calls the given function once for each element, passing that element's index and value.
	Each(func(index int, value interface{}))

	// Map invokes the given function once for each element and returns a
	// container containing the values returned by the given function.
	// TODO would appreciate help on how to enforce this in containers (don't want to type assert when chaining)
	// Map(func(index int, value interface{}) interface{}) Container

	// Select returns a new container containing all elements for which the given function returns a true value.
	// TODO need help on how to enforce this in containers (don't want to type assert when chaining)
	// Select(func(index int, value interface{}) bool) Container

	// Any passes each element of the container to the given function and
	// returns true if the function ever returns true for any element.
	Any(func(index int, value interface{}) bool) bool

	// All passes each element of the container to the given function and
	// returns true if the function returns true for all elements.
	All(func(index int, value interface{}) bool) bool

	// Find passes each element of the container to the given function and returns
	// the first (index,value) for which the function is true or -1,nil otherwise
	// if no element matches the criteria.
	Find(func(index int, value interface{}) bool) (int, interface{})
}

// EnumerableWithKey provides functions for ordered containers whose values whose elements are key/value pairs.
type EnumerableWithKey interface {
	// Each calls the given function once for each element, passing that element's key and value.
	Each(func(key interface{}, value interface{}))

	// Map invokes the given function once for each element and returns a container
	// containing the values returned by the given function as key/value pairs.
	// TODO need help on how to enforce this in containers (don't want to type assert when chaining)
	// Map(func(key interface{}, value interface{}) (interface{}, interface{})) Container

	// Select returns a new container containing all elements for which the given function returns a true value.
	// TODO need help on how to enforce this in containers (don't want to type assert when chaining)
	// Select(func(key interface{}, value interface{}) bool) Container

	// Any passes each element of the container to the given function and
	// returns true if the function ever returns true for any element.
	Any(func(key interface{}, value interface{}) bool) bool

	// All passes each element of the container to the given function and
	// returns true if the function returns true for all elements.
	All(func(key interface{}, value interface{}) bool) bool

	// Find passes each element of the container to the given function and returns
	// the first (key,value) for which the function is true or nil,nil otherwise if no element
	// matches the criteria.
	Find(func(key interface{}, value interface{}) bool) (interface{}, interface{})
}
