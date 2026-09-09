//go:build go1.18

/*
Copyright 2022 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package transport

// this is just to make the "unused" linter rule happy
var _ = isCacheKeyComparable[tlsCacheKey]

// assert at compile time that tlsCacheKey is comparable in a way that will never panic at runtime.
//
// Golang 1.20 introduced an exception to type constraints that allows comparable, but not
// necessarily strictly comparable type arguments to satisfy the `comparable` type constraint,
// thus allowing interfaces to fulfil the `comparable` constraint.
// However, by definition, "A comparison of two interface values with identical
// dynamic types causes a run-time panic if that type is not comparable".
//
// We want to make sure that comparing two `tlsCacheKey` elements won't cause a
// runtime panic. In order to do that, we'll force the `tlsCacheKey` to be strictly
// comparable, thus making it impossible for it to contain interfaces.
// To assert strict comparability, we'll use another definition: "Type
// parameters are comparable if they are strictly comparable".
// Below, we first construct a type parameter from the `tlsCacheKey` type so that
// we can then push this type parameter to a comparable check, thus checking these
// are strictly comparable.
//
// Original suggestion from https://github.com/golang/go/issues/56548#issuecomment-1317673963
func isCacheKeyComparable[K tlsCacheKey]() {
	_ = isComparable[K]
}

func isComparable[T comparable]() {}
