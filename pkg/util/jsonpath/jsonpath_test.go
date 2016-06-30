/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"
	"reflect"
	"sort"
	"strings"
	"testing"
)

type jsonpathTest struct {
	name     string
	template string
	input    interface{}
	expect   string
}

func testJSONPath(tests []jsonpathTest, t *testing.T) {
	for _, test := range tests {
		j := New(test.name)
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

// testJSONPathSortOutput test cases related to map, the results may print in random order
func testJSONPathSortOutput(tests []jsonpathTest, t *testing.T) {
	for _, test := range tests {
		j := New(test.name)
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
		//since map is visited in random order, we need to sort the results.
		sortedOut := strings.Fields(out)
		sort.Strings(sortedOut)
		sortedExpect := strings.Fields(test.expect)
		sort.Strings(sortedExpect)
		if !reflect.DeepEqual(sortedOut, sortedExpect) {
			t.Errorf(`in %s, expect to get "%s", got "%s"`, test.name, test.expect, out)
		}
	}
}

func testFailJSONPath(tests []jsonpathTest, t *testing.T) {
	for _, test := range tests {
		j := New(test.name)
		err := j.Parse(test.template)
		if err != nil {
			t.Errorf("in %s, parse %s error %v", test.name, test.template, err)
		}
		buf := new(bytes.Buffer)
		err = j.Execute(buf, test.input)
		var out string
		if err == nil {
			out = "nil"
		} else {
			out = err.Error()
		}
		if out != test.expect {
			t.Errorf("in %s, expect to get error %q, got %q", test.name, test.expect, out)
		}
	}
}

type book struct {
	Category string
	Author   string
	Title    string
	Price    float32
}

func (b book) String() string {
	return fmt.Sprintf("{Category: %s, Author: %s, Title: %s, Price: %v}", b.Category, b.Author, b.Title, b.Price)
}

type bicycle struct {
	Color string
	Price float32
}

type empName string
type job string
type store struct {
	Book      []book
	Bicycle   bicycle
	Name      string
	Labels    map[string]int
	Employees map[empName]job
}

func TestStructInput(t *testing.T) {

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
		Employees: map[empName]job{
			"jason": "manager",
			"dan":   "clerk",
		},
	}

	storeTests := []jsonpathTest{
		{"plain", "hello jsonpath", nil, "hello jsonpath"},
		{"recursive", "{..}", []int{1, 2, 3}, "[1 2 3]"},
		{"filter", "{[?(@<5)]}", []int{2, 6, 3, 7}, "2 3"},
		{"quote", `{"{"}`, nil, "{"},
		{"union", "{[1,3,4]}", []int{0, 1, 2, 3, 4}, "1 3 4"},
		{"array", "{[0:2]}", []string{"Monday", "Tudesday"}, "Monday Tudesday"},
		{"variable", "hello {.Name}", storeData, "hello jsonpath"},
		{"dict/", "{$.Labels.web/html}", storeData, "15"},
		{"dict/", "{$.Employees.jason}", storeData, "manager"},
		{"dict/", "{$.Employees.dan}", storeData, "clerk"},
		{"dict-", "{.Labels.k8s-app}", storeData, "20"},
		{"nest", "{.Bicycle.Color}", storeData, "red"},
		{"allarray", "{.Book[*].Author}", storeData, "Nigel Rees Evelyn Waugh Herman Melville"},
		{"allfileds", "{.Bicycle.*}", storeData, "red 19.95"},
		{"recurfileds", "{..Price}", storeData, "8.95 12.99 8.99 19.95"},
		{"lastarray", "{.Book[-1:]}", storeData,
			"{Category: fiction, Author: Herman Melville, Title: Moby Dick, Price: 8.99}"},
		{"recurarray", "{..Book[2]}", storeData,
			"{Category: fiction, Author: Herman Melville, Title: Moby Dick, Price: 8.99}"},
	}
	testJSONPath(storeTests, t)

	failStoreTests := []jsonpathTest{
		{"invalid identfier", "{hello}", storeData, "unrecognized identifier hello"},
		{"nonexistent field", "{.hello}", storeData, "hello is not found"},
		{"invalid array", "{.Labels[0]}", storeData, "map[string]int is not array or slice"},
		{"invalid filter operator", "{.Book[?(@.Price<>10)]}", storeData, "unrecognized filter operator <>"},
		{"redundent end", "{range .Labels.*}{@}{end}{end}", storeData, "not in range, nothing to end"},
	}
	testFailJSONPath(failStoreTests, t)
}

func TestJSONInput(t *testing.T) {
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
	pointsTests := []jsonpathTest{
		{"exists filter", "{[?(@.z)].id}", pointsData, "i2 i5"},
		{"bracket key", "{[0]['id']}", pointsData, "i1"},
	}
	testJSONPath(pointsTests, t)
}

// TestKubernetes tests some use cases from kubernetes
func TestKubernetes(t *testing.T) {
	var input = []byte(`{
	  "kind": "List",
	  "items":[
		{
		  "kind":"None",
		  "metadata":{"name":"127.0.0.1"},
		  "status":{
			"capacity":{"cpu":"4"},
			"addresses":[{"type": "LegacyHostIP", "address":"127.0.0.1"}]
		  }
		},
		{
		  "kind":"None",
		  "metadata":{"name":"127.0.0.2"},
		  "status":{
			"capacity":{"cpu":"8"},
			"addresses":[
			  {"type": "LegacyHostIP", "address":"127.0.0.2"},
			  {"type": "another", "address":"127.0.0.3"}
			]
		  }
		}
	  ],
	  "users":[
	    {
	      "name": "myself",
	      "user": {}
	    },
	    {
	      "name": "e2e",
	      "user": {"username": "admin", "password": "secret"}
	  	}
	  ]
	}`)
	var nodesData interface{}
	err := json.Unmarshal(input, &nodesData)
	if err != nil {
		t.Error(err)
	}

	nodesTests := []jsonpathTest{
		{"range item", `{range .items[*]}{.metadata.name}, {end}{.kind}`, nodesData, "127.0.0.1, 127.0.0.2, List"},
		{"range item with quote", `{range .items[*]}{.metadata.name}{"\t"}{end}`, nodesData, "127.0.0.1\t127.0.0.2\t"},
		{"range addresss", `{.items[*].status.addresses[*].address}`, nodesData,
			"127.0.0.1 127.0.0.2 127.0.0.3"},
		{"double range", `{range .items[*]}{range .status.addresses[*]}{.address}, {end}{end}`, nodesData,
			"127.0.0.1, 127.0.0.2, 127.0.0.3, "},
		{"item name", `{.items[*].metadata.name}`, nodesData, "127.0.0.1 127.0.0.2"},
		{"union nodes capacity", `{.items[*]['metadata.name', 'status.capacity']}`, nodesData,
			"127.0.0.1 127.0.0.2 map[cpu:4] map[cpu:8]"},
		{"range nodes capacity", `{range .items[*]}[{.metadata.name}, {.status.capacity}] {end}`, nodesData,
			"[127.0.0.1, map[cpu:4]] [127.0.0.2, map[cpu:8]] "},
		{"user password", `{.users[?(@.name=="e2e")].user.password}`, &nodesData, "secret"},
	}
	testJSONPath(nodesTests, t)

	randomPrintOrderTests := []jsonpathTest{
		{"recursive name", "{..name}", nodesData, `127.0.0.1 127.0.0.2 myself e2e`},
	}
	testJSONPathSortOutput(randomPrintOrderTests, t)
}
