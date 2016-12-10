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

func testJSONPath(tests []jsonpathTest, allowMissingKeys bool, t *testing.T) {
	for _, test := range tests {
		j := New(test.name)
		j.AllowMissingKeys(allowMissingKeys)
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
			t.Errorf(`in %s, expect to get "%q", got "%q"`, test.name, test.expect, out)
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
			t.Errorf(`in %s, expect to get "%q", got "%q"`, test.name, test.expect, out)
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
		{"plain", "hello jsonpath", nil, "[\n    \"hello jsonpath\"\n]\n"},
		{"recursive", "{..}", []int{1, 2, 3}, "[\n    [\n        1,\n        2,\n        3\n    ]\n]\n"},
		{"filter", "{[?(@<5)]}", []int{2, 6, 3, 7}, "[\n    2,\n    3\n]\n"},
		{"quote", `{"{"}`, nil, "[\n    \"{\"\n]\n"},
		{"union", "{[1,3,4]}", []int{0, 1, 2, 3, 4}, "[\n    1,\n    3,\n    4\n]\n"},
		{"array", "{[0:2]}", []string{"Monday", "Tuesday"}, "[\n    \"Monday\",\n    \"Tuesday\"\n]\n"},
		{"variable", "hello {.Name}", storeData, "[\n    \"hello \"\n]\n[\n    \"jsonpath\"\n]\n"},
		{"dict/", "{$.Labels.web/html}", storeData, "[\n    15\n]\n"},
		{"dict/", "{$.Employees.jason}", storeData, "[\n    \"manager\"\n]\n"},
		{"dict/", "{$.Employees.dan}", storeData, "[\n    \"clerk\"\n]\n"},
		{"dict-", "{.Labels.k8s-app}", storeData, "[\n    20\n]\n"},
		{"nest", "{.Bicycle.Color}", storeData, "[\n    \"red\"\n]\n"},
		{"allarray", "{.Book[*].Author}", storeData, "[\n    \"Nigel Rees\",\n    \"Evelyn Waugh\",\n    \"Herman Melville\"\n]\n"},
		{"allfileds", "{.Bicycle.*}", storeData, "[\n    \"red\",\n    19.95\n]\n"},
		{"recurfileds", "{..Price}", storeData, "[\n    8.95,\n    12.99,\n    8.99,\n    19.95\n]\n"},
		{"lastarray", "{.Book[-1:]}", storeData,
			"[\n    {\n        \"Category\": \"fiction\",\n        \"Author\": \"Herman Melville\",\n        \"Title\": \"Moby Dick\",\n        \"Price\": 8.99\n    }\n]\n"},
		{"recurarray", "{..Book[2]}", storeData,
			"[\n    {\n        \"Category\": \"fiction\",\n        \"Author\": \"Herman Melville\",\n        \"Title\": \"Moby Dick\",\n        \"Price\": 8.99\n    }\n]\n"},
	}
	testJSONPath(storeTests, false, t)

	missingKeyTests := []jsonpathTest{
		{"nonexistent field", "{.hello}", storeData, "[]\n"},
	}
	testJSONPath(missingKeyTests, true, t)

	failStoreTests := []jsonpathTest{
		{"invalid identifier", "{hello}", storeData, "unrecognized identifier hello"},
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
		{"exists filter", "{[?(@.z)].id}", pointsData, "[\n    \"i2\",\n    \"i5\"\n]\n"},
		{"bracket key", "{[0]['id']}", pointsData, "[\n    \"i1\"\n]\n"},
	}
	testJSONPath(pointsTests, false, t)
}

// TestKubernetes tests some use cases from kubernetes
func TestKubernetes(t *testing.T) {
	var input = []byte(`{
	  "kind": "List",
	  "items":[
		{
		  "kind":"None",
		  "metadata":{
		    "name":"127.0.0.1",
			"labels":{
			  "kubernetes.io/hostname":"127.0.0.1"
			}
		  },
		  "status":{
			"capacity":{"cpu":"4"},
			"addresses":[{"type": "LegacyHostIP", "address":"127.0.0.1"}]
		  }
		},
		{
		  "kind":"None",
		  "metadata":{
			"name":"127.0.0.2",
			"labels":{
			  "kubernetes.io/hostname":"127.0.0.2"
			}
		  },
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
		{"range item", `{range .items[*]}{.metadata.name}, {end}{.kind}`, nodesData, "[\n    \"127.0.0.1\"\n]\n[\n    \", \"\n]\n[\n    \"127.0.0.2\"\n]\n[\n    \", \"\n]\n[]\n[\n    \"List\"\n]\n"},
		{"range item with quote", `{range .items[*]}{.metadata.name}{"\t"}{end}`, nodesData, "[\n    \"127.0.0.1\"\n]\n[\n    \"\\t\"\n]\n[\n    \"127.0.0.2\"\n]\n[\n    \"\\t\"\n]\n[]\n"},
		{"range addresss", `{.items[*].status.addresses[*].address}`, nodesData,
			"[\n    \"127.0.0.1\",\n    \"127.0.0.2\",\n    \"127.0.0.3\"\n]\n"},
		{"double range", `{range .items[*]}{range .status.addresses[*]}{.address}, {end}{end}`, nodesData,
			"[\n    \"127.0.0.1\"\n]\n[\n    \", \"\n]\n[\n    \"127.0.0.2\"\n]\n[\n    \", \"\n]\n[\n    \"127.0.0.3\"\n]\n[\n    \", \"\n]\n[]\n[]\n"},
		{"item name", `{.items[*].metadata.name}`, nodesData, "[\n    \"127.0.0.1\",\n    \"127.0.0.2\"\n]\n"},
		{"union nodes capacity", `{.items[*]['metadata.name', 'status.capacity']}`, nodesData,
			"[\n    \"127.0.0.1\",\n    \"127.0.0.2\",\n    {\n        \"cpu\": \"4\"\n    },\n    {\n        \"cpu\": \"8\"\n    }\n]\n"},
		{"range nodes capacity", `{range .items[*]}[{.metadata.name}, {.status.capacity}] {end}`, nodesData,
			"[\n    \"[\"\n]\n[\n    \"127.0.0.1\"\n]\n[\n    \", \"\n]\n[\n    {\n        \"cpu\": \"4\"\n    }\n]\n[\n    \"] \"\n]\n[\n    \"[\"\n]\n[\n    \"127.0.0.2\"\n]\n[\n    \", \"\n]\n[\n    {\n        \"cpu\": \"8\"\n    }\n]\n[\n    \"] \"\n]\n[]\n"},
		{"user password", `{.users[?(@.name=="e2e")].user.password}`, &nodesData, "[\n    \"secret\"\n]\n"},
		{"hostname", `{.items[0].metadata.labels.kubernetes\.io/hostname}`, &nodesData, "[\n    \"127.0.0.1\"\n]\n"},
		{"hostname filter", `{.items[?(@.metadata.labels.kubernetes\.io/hostname=="127.0.0.1")].kind}`, &nodesData, "[\n    \"None\"\n]\n"},
	}
	testJSONPath(nodesTests, false, t)

	randomPrintOrderTests := []jsonpathTest{
		{"recursive name", "{..name}", nodesData, "[\n    \"127.0.0.1\",\n    \"127.0.0.2\",\n    \"myself\",\n    \"e2e\"\n]\n"},
	}
	testJSONPathSortOutput(randomPrintOrderTests, t)
}
