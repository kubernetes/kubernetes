/*
Copyright 2016 The Kubernetes Authors.

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

package maps

import "fmt"

// ExampleNewCounter is a https://blog.golang.org/examples styled unit test.
func ExampleNewCounter() {

	c := NewCounter()
	c.Incr("kubernetes")
	c.Incr("kubernetes")
	c.Incr("counters")
	c.Incr("are great!")
	fmt.Println(c.Get("kubernetes"))
	fmt.Println(c.Get("counters"))
	fmt.Println(c.Get("are great!"))
	fmt.Println(c.Get("THIS KEY DOESNT EXIST!"))

	// Output:
	// 2
	// 1
	// 1
	// 0
}
