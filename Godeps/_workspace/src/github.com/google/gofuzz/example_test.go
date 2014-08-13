/*
Copyright 2014 Google Inc. All rights reserved.

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

package fuzz_test

import (
	"encoding/json"
	"fmt"
	"math/rand"

	"github.com/google/gofuzz"
)

func ExampleSimple() {
	type MyType struct {
		A string
		B string
		C int
		D struct {
			E float64
		}
	}

	f := fuzz.New()
	object := MyType{}

	uniqueObjects := map[MyType]int{}

	for i := 0; i < 1000; i++ {
		f.Fuzz(&object)
		uniqueObjects[object]++
	}
	fmt.Printf("Got %v unique objects.\n", len(uniqueObjects))
	// Output:
	// Got 1000 unique objects.
}

func ExampleCustom() {
	type MyType struct {
		A int
		B string
	}

	counter := 0
	f := fuzz.New().Funcs(
		func(i *int, c fuzz.Continue) {
			*i = counter
			counter++
		},
	)
	object := MyType{}

	uniqueObjects := map[MyType]int{}

	for i := 0; i < 100; i++ {
		f.Fuzz(&object)
		if object.A != i {
			fmt.Printf("Unexpected value: %#v\n", object)
		}
		uniqueObjects[object]++
	}
	fmt.Printf("Got %v unique objects.\n", len(uniqueObjects))
	// Output:
	// Got 100 unique objects.
}

func ExampleComplex() {
	type OtherType struct {
		A string
		B string
	}
	type MyType struct {
		Pointer             *OtherType
		Map                 map[string]OtherType
		PointerMap          *map[string]OtherType
		Slice               []OtherType
		SlicePointer        []*OtherType
		PointerSlicePointer *[]*OtherType
	}

	f := fuzz.New().RandSource(rand.NewSource(0)).NilChance(0).NumElements(1, 1).Funcs(
		func(o *OtherType, c fuzz.Continue) {
			o.A = "Foo"
			o.B = "Bar"
		},
		func(op **OtherType, c fuzz.Continue) {
			*op = &OtherType{"A", "B"}
		},
		func(m map[string]OtherType, c fuzz.Continue) {
			m["Works Because"] = OtherType{
				"Fuzzer",
				"Preallocated",
			}
		},
	)
	object := MyType{}
	f.Fuzz(&object)
	bytes, err := json.MarshalIndent(&object, "", "    ")
	if err != nil {
		fmt.Printf("error: %v\n", err)
	}
	fmt.Printf("%s\n", string(bytes))
	// Output:
	// {
	//     "Pointer": {
	//         "A": "A",
	//         "B": "B"
	//     },
	//     "Map": {
	//         "Works Because": {
	//             "A": "Fuzzer",
	//             "B": "Preallocated"
	//         }
	//     },
	//     "PointerMap": {
	//         "Works Because": {
	//             "A": "Fuzzer",
	//             "B": "Preallocated"
	//         }
	//     },
	//     "Slice": [
	//         {
	//             "A": "Foo",
	//             "B": "Bar"
	//         }
	//     ],
	//     "SlicePointer": [
	//         {
	//             "A": "A",
	//             "B": "B"
	//         }
	//     ],
	//     "PointerSlicePointer": [
	//         {
	//             "A": "A",
	//             "B": "B"
	//         }
	//     ]
	// }
}

func ExampleMap() {
	f := fuzz.New().NilChance(0).NumElements(1, 1)
	var myMap map[struct{ A, B, C int }]string
	f.Fuzz(&myMap)
	fmt.Printf("myMap has %v element(s).\n", len(myMap))
	// Output:
	// myMap has 1 element(s).
}

func ExampleSingle() {
	f := fuzz.New()
	var i int
	f.Fuzz(&i)

	// Technically, we'd expect this to fail one out of 2 billion attempts...
	fmt.Printf("(i == 0) == %v", i == 0)
	// Output:
	// (i == 0) == false
}

func ExampleEnum() {
	type MyEnum string
	const (
		A MyEnum = "A"
		B MyEnum = "B"
	)
	type MyInfo struct {
		Type  MyEnum
		AInfo *string
		BInfo *string
	}

	f := fuzz.New().NilChance(0).Funcs(
		func(e *MyInfo, c fuzz.Continue) {
			// Note c's embedded Rand allows for direct use.
			// We could also use c.RandBool() here.
			switch c.Intn(2) {
			case 0:
				e.Type = A
				c.Fuzz(&e.AInfo)
			case 1:
				e.Type = B
				c.Fuzz(&e.BInfo)
			}
		},
	)

	for i := 0; i < 100; i++ {
		var myObject MyInfo
		f.Fuzz(&myObject)
		switch myObject.Type {
		case A:
			if myObject.AInfo == nil {
				fmt.Println("AInfo should have been set!")
			}
			if myObject.BInfo != nil {
				fmt.Println("BInfo should NOT have been set!")
			}
		case B:
			if myObject.BInfo == nil {
				fmt.Println("BInfo should have been set!")
			}
			if myObject.AInfo != nil {
				fmt.Println("AInfo should NOT have been set!")
			}
		default:
			fmt.Println("Invalid enum value!")
		}
	}
	// Output:
}
