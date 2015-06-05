/**
 *  Copyright 2014 Paul Querna
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ff

type SweetInterface interface {
	Cats() int
}

type Cats struct {
	FieldOnCats int
}

func (c *Cats) Cats() int {
	return 42
}

type Embed struct {
	SuperBool bool
}

type Everything struct {
	Embed
	Bool             bool
	Int              int
	Int8             int8
	Int16            int16
	Int32            int32
	Int64            int64
	Uint             uint
	Uint8            uint8
	Uint16           uint16
	Uint32           uint32
	Uint64           uint64
	Uintptr          uintptr
	Float32          float32
	Float64          float64
	Array            []int
	Map              map[string]int
	String           string
	StringPointer    *string
	Int64Pointer     *int64
	FooStruct        *Foo
	MySweetInterface SweetInterface
	nonexported
}

type nonexported struct {
	Something int8
}

type Foo struct {
	Bar int
}

func NewEverything(e *Everything) {
	e.SuperBool = true
	e.Bool = true
	e.Int = 1
	e.Int8 = 2
	e.Int16 = 3
	e.Int32 = -4
	e.Int64 = 2 ^ 59
	e.Uint = 100
	e.Uint8 = 101
	e.Uint16 = 102
	e.Uint64 = 103
	e.Uintptr = 104
	e.Float32 = 3.14
	e.Float64 = 3.15
	e.Array = []int{1, 2, 3}
	e.Map = map[string]int{
		"foo": 1,
		"bar": 2,
	}
	e.String = "snowman->☃"
	e.FooStruct = &Foo{Bar: 1}
	e.Something = 99
	e.MySweetInterface = &Cats{}
}
