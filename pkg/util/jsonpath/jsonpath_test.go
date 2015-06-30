/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package jsonpath

import (
	"bytes"
	"encoding/json"
	"testing"
)

type jsonpathTest struct {
	name     string
	template string
	input    interface{}
	expect   string
}

type book struct {
	Category string
	Author   string
	Title    string
	Price    float32
}

type bicycle struct {
	Color string
	Price float32
}

type store struct {
	Book    []book
	Bicycle bicycle
	Name    string
	Labels  map[string]int
}

func testJSONPath(tests []jsonpathTest, t *testing.T) {
	for _, test := range tests {
		j := NewJSONPath(test.name)
		err := j.Parse(test.template)
		if err != nil {
			t.Errorf("in %s, parse %s error %v", test.name, test.template, err)
		}
		buf := new(bytes.Buffer)
		err = j.Execute(buf, test.input)
		if err != nil {
			t.Errorf("in %s, execute error %v", test.name, err)
		}
		out := buf.String()
		if out != test.expect {
			t.Errorf(`in %s, expect to get "%s", got "%s"`, test.name, test.expect, out)
		}
	}
}

func TestStoreData(t *testing.T) {
	storeData := store{
		Name: "jsonpath",
		Book: []book{
			{"reference", "Nigel Rees", "Sayings of the Centurey", 8.95},
			{"fiction", "Evelyn Waugh", "Sword of Honour", 12.99},
			{"fiction", "Herman Melville", "Moby Dick", 8.99},
		},
		Bicycle: bicycle{"red", 19.95},
		Labels: map[string]int{
			"engieer":  10,
			"web/html": 15,
			"k8s-app":  20,
		},
	}

	storeTests := []jsonpathTest{
		{"plain", "hello jsonpath", nil, "hello jsonpath"},
		{"recursive", "${..}", []int{1, 2, 3}, "[1, 2, 3]"},
		{"filter", "${[?(@<5)]}", []int{2, 6, 3, 7}, "2 3"},
		{"quote", `${"${"}`, nil, "${"},
		{"union", "${[1,3,4]}", []int{0, 1, 2, 3, 4}, "1 3 4"},
		{"array", "${[0:2]}", []string{"Monday", "Tudesday"}, "Monday Tudesday"},
		{"variable", "hello ${.Name}", storeData, "hello jsonpath"},
		{"dict/", "${.Labels.web/html}", storeData, "15"},
		{"dict-", "${.Labels.k8s-app}", storeData, "20"},
		{"nest", "${.Bicycle.Color}", storeData, "red"},
		{"allarray", "${.Book[*].Author}", storeData, "Nigel Rees Evelyn Waugh Herman Melville"},
		{"allfileds", "${.Bicycle.*}", storeData, "red 19.95"},
		{"recurfileds", "${..Price}", storeData, "8.95 12.99 8.99 19.95"},
		{"lastarray", "${.Book[-1:]}", storeData,
			"{Category: fiction, Author: Herman Melville, Title: Moby Dick, Price: 8.99}"},
		{"recurarray", "${..Book[2]}", storeData,
			"{Category: fiction, Author: Herman Melville, Title: Moby Dick, Price: 8.99}"},
	}
	testJSONPath(storeTests, t)
}

func TestPoints(t *testing.T) {
	var pointsJSON = []byte(`[
		{"id": "i1", "x":4, "y":-5},
		{"id": "i2", "x":-2, "y":-5, "z":1},
		{"id": "i3", "x":  8, "y":  3 },
		{"id": "i4", "x": -6, "y": -1 },
		{"id": "i5", "x":  0, "y":  2, "z": 1 },
		{"id": "i6", "x":  1, "y":  4 }
	]`)
	var pointsData interface{}
	err := json.Unmarshal(pointsJSON, &pointsData)
	if err != nil {
		t.Error(err)
	}
	t.Log(pointsData)
	pointsTests := []jsonpathTest{
		{"existsfilter", "${[?(@.z)].id}", pointsData, "i2 i5"},
		{"arrayfield", "${[0]['id']}", pointsData, "i1"},
	}
	testJSONPath(pointsTests, t)
}
