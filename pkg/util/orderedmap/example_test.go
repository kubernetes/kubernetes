/*
Copyright 2019 The Kubernetes Authors.

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

package orderedmap_test

import (
	"fmt"

	"k8s.io/kubernetes/pkg/util/orderedmap"
)

func Example() {
	om := orderedmap.New()

	om.Set("foo", "bar")
	om.Set("bar", "baz")
	om.Set("coucou", "toi")

	fmt.Println("## Get operations: ##")
	fmt.Println(om.Get("foo"))
	fmt.Println(om.Get("i dont exist"))

	fmt.Println("## Iterating over pairs from oldest to newest: ##")
	for pair := om.Oldest(); pair != nil; pair = pair.Next() {
		fmt.Printf("%s => %s\n", pair.Key, pair.Value)
	}

	fmt.Println("## Iterating over the 2 newest pairs: ##")
	i := 0
	for pair := om.Newest(); pair != nil; pair = pair.Prev() {
		fmt.Printf("%s => %s\n", pair.Key, pair.Value)
		i++
		if i >= 2 {
			break
		}
	}

	// Output:
	// ## Get operations: ##
	// bar true
	// <nil> false
	// ## Iterating over pairs from oldest to newest: ##
	// foo => bar
	// bar => baz
	// coucou => toi
	// ## Iterating over the 2 newest pairs: ##
	// coucou => toi
	// bar => baz
}
