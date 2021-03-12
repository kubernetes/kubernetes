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
	name        string
	template    string
	input       interface{}
	expect      string
	expectError bool
}

func testJSONPath(tests []jsonpathTest, allowMissingKeys bool, t *testing.T) {
	for _, test := range tests {
		j := New(test.name)
		j.AllowMissingKeys(allowMissingKeys)
		err := j.Parse(test.template)
		if err != nil {
			if !test.expectError {
				t.Errorf("in %s, parse %s error %v", test.name, test.template, err)
			}
			continue
		}
		buf := new(bytes.Buffer)
		err = j.Execute(buf, test.input)
		if test.expectError {
			if err == nil {
				t.Errorf(`in %s, expected execute error, got %q`, test.name, buf)
			}
			continue
		} else if err != nil {
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

func TestTypesInput(t *testing.T) {
	types := map[string]interface{}{
		"bools":      []bool{true, false, true, false},
		"integers":   []int{1, 2, 3, 4},
		"floats":     []float64{1.0, 2.2, 3.3, 4.0},
		"strings":    []string{"one", "two", "three", "four"},
		"interfaces": []interface{}{true, "one", 1, 1.1},
		"maps": []map[string]interface{}{
			{"name": "one", "value": 1},
			{"name": "two", "value": 2.02},
			{"name": "three", "value": 3.03},
			{"name": "four", "value": 4.04},
		},
		"structs": []struct {
			Name  string      `json:"name"`
			Value interface{} `json:"value"`
			Type  string      `json:"type"`
		}{
			{Name: "one", Value: 1, Type: "integer"},
			{Name: "two", Value: 2.002, Type: "float"},
			{Name: "three", Value: 3, Type: "integer"},
			{Name: "four", Value: 4.004, Type: "float"},
		},
	}

	sliceTests := []jsonpathTest{
		// boolean slice tests
		{"boolSlice", `{ .bools }`, types, `[true,false,true,false]`, false},
		{"boolSliceIndex", `{ .bools[0] }`, types, `true`, false},
		{"boolSliceIndex", `{ .bools[-1] }`, types, `false`, false},
		{"boolSubSlice", `{ .bools[0:2] }`, types, `true false`, false},
		{"boolSubSliceFirst2", `{ .bools[:2] }`, types, `true false`, false},
		{"boolSubSliceStep2", `{ .bools[:4:2] }`, types, `true true`, false},
		// integer slice tests
		{"integerSlice", `{ .integers }`, types, `[1,2,3,4]`, false},
		{"integerSliceIndex", `{ .integers[0] }`, types, `1`, false},
		{"integerSliceIndexReverse", `{ .integers[-2] }`, types, `3`, false},
		{"integerSubSliceFirst2", `{ .integers[0:2] }`, types, `1 2`, false},
		{"integerSubSliceFirst2Alt", `{ .integers[:2] }`, types, `1 2`, false},
		{"integerSubSliceStep2", `{ .integers[:4:2] }`, types, `1 3`, false},
		// float slice tests
		{"floatSlice", `{ .floats }`, types, `[1,2.2,3.3,4]`, false},
		{"floatSliceIndex", `{ .floats[0] }`, types, `1`, false},
		{"floatSliceIndexReverse", `{ .floats[-2] }`, types, `3.3`, false},
		{"floatSubSliceFirst2", `{ .floats[0:2] }`, types, `1 2.2`, false},
		{"floatSubSliceFirst2Alt", `{ .floats[:2] }`, types, `1 2.2`, false},
		{"floatSubSliceStep2", `{ .floats[:4:2] }`, types, `1 3.3`, false},
		// strings slice tests
		{"stringSlice", `{ .strings }`, types, `["one","two","three","four"]`, false},
		{"stringSliceIndex", `{ .strings[0] }`, types, `one`, false},
		{"stringSliceIndexReverse", `{ .strings[-2] }`, types, `three`, false},
		{"stringSubSliceFirst2", `{ .strings[0:2] }`, types, `one two`, false},
		{"stringSubSliceFirst2Alt", `{ .strings[:2] }`, types, `one two`, false},
		{"stringSubSliceStep2", `{ .strings[:4:2] }`, types, `one three`, false},
		// interfaces slice tests
		{"interfaceSlice", `{ .interfaces }`, types, `[true,"one",1,1.1]`, false},
		{"interfaceSliceIndex", `{ .interfaces[0] }`, types, `true`, false},
		{"interfaceSliceIndexReverse", `{ .interfaces[-2] }`, types, `1`, false},
		{"interfaceSubSliceFirst2", `{ .interfaces[0:2] }`, types, `true one`, false},
		{"interfaceSubSliceFirst2Alt", `{ .interfaces[:2] }`, types, `true one`, false},
		{"interfaceSubSliceStep2", `{ .interfaces[:4:2] }`, types, `true 1`, false},
		// maps slice tests
		{"mapSlice", `{ .maps }`, types,
			`[{"name":"one","value":1},{"name":"two","value":2.02},{"name":"three","value":3.03},{"name":"four","value":4.04}]`, false},
		{"mapSliceIndex", `{ .maps[0] }`, types, `{"name":"one","value":1}`, false},
		{"mapSliceIndexReverse", `{ .maps[-2] }`, types, `{"name":"three","value":3.03}`, false},
		{"mapSubSliceFirst2", `{ .maps[0:2] }`, types, `{"name":"one","value":1} {"name":"two","value":2.02}`, false},
		{"mapSubSliceFirst2Alt", `{ .maps[:2] }`, types, `{"name":"one","value":1} {"name":"two","value":2.02}`, false},
		{"mapSubSliceStepOdd", `{ .maps[::2] }`, types, `{"name":"one","value":1} {"name":"three","value":3.03}`, false},
		{"mapSubSliceStepEven", `{ .maps[1::2] }`, types, `{"name":"two","value":2.02} {"name":"four","value":4.04}`, false},
		// structs slice tests
		{"structSlice", `{ .structs }`, types,
			`[{"name":"one","value":1,"type":"integer"},{"name":"two","value":2.002,"type":"float"},{"name":"three","value":3,"type":"integer"},{"name":"four","value":4.004,"type":"float"}]`, false},
		{"structSliceIndex", `{ .structs[0] }`, types, `{"name":"one","value":1,"type":"integer"}`, false},
		{"structSliceIndexReverse", `{ .structs[-2] }`, types, `{"name":"three","value":3,"type":"integer"}`, false},
		{"structSubSliceFirst2", `{ .structs[0:2] }`, types,
			`{"name":"one","value":1,"type":"integer"} {"name":"two","value":2.002,"type":"float"}`, false},
		{"structSubSliceFirst2Alt", `{ .structs[:2] }`, types,
			`{"name":"one","value":1,"type":"integer"} {"name":"two","value":2.002,"type":"float"}`, false},
		{"structSubSliceStepOdd", `{ .structs[::2] }`, types,
			`{"name":"one","value":1,"type":"integer"} {"name":"three","value":3,"type":"integer"}`, false},
		{"structSubSliceStepEven", `{ .structs[1::2] }`, types,
			`{"name":"two","value":2.002,"type":"float"} {"name":"four","value":4.004,"type":"float"}`, false},
	}

	testJSONPath(sliceTests, false, t)
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
	IsNew bool
}

type empName string
type job string
type store struct {
	Book      []book
	Bicycle   []bicycle
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
		Bicycle: []bicycle{
			{"red", 19.95, true},
			{"green", 20.01, false},
		},
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
		{"plain", "hello jsonpath", nil, "hello jsonpath", false},
		{"recursive", "{..}", []int{1, 2, 3}, "[1,2,3]", false},
		{"filter", "{[?(@<5)]}", []int{2, 6, 3, 7}, "2 3", false},
		{"quote", `{"{"}`, nil, "{", false},
		{"union", "{[1,3,4]}", []int{0, 1, 2, 3, 4}, "1 3 4", false},
		{"array", "{[0:2]}", []string{"Monday", "Tudesday"}, "Monday Tudesday", false},
		{"variable", "hello {.Name}", storeData, "hello jsonpath", false},
		{"dict/", "{$.Labels.web/html}", storeData, "15", false},
		{"dict/", "{$.Employees.jason}", storeData, "manager", false},
		{"dict/", "{$.Employees.dan}", storeData, "clerk", false},
		{"dict-", "{.Labels.k8s-app}", storeData, "20", false},
		{"nest", "{.Bicycle[*].Color}", storeData, "red green", false},
		{"allarray", "{.Book[*].Author}", storeData, "Nigel Rees Evelyn Waugh Herman Melville", false},
		{"allfields", `{range .Bicycle[*]}{ "{" }{ @.* }{ "} " }{end}`, storeData, "{red 19.95 true} {green 20.01 false} ", false},
		{"recurfields", "{..Price}", storeData, "8.95 12.99 8.99 19.95 20.01", false},
		{"recurdotfields", "{...Price}", storeData, "8.95 12.99 8.99 19.95 20.01", false},
		{"superrecurfields", "{............................................................Price}", storeData, "", true},
		{"allstructsSlice", "{.Bicycle}", storeData,
			`[{"Color":"red","Price":19.95,"IsNew":true},{"Color":"green","Price":20.01,"IsNew":false}]`, false},
		{"allstructs", `{range .Bicycle[*]}{ @ }{ " " }{end}`, storeData,
			`{"Color":"red","Price":19.95,"IsNew":true} {"Color":"green","Price":20.01,"IsNew":false} `, false},
		{"lastarray", "{.Book[-1:]}", storeData,
			`{"Category":"fiction","Author":"Herman Melville","Title":"Moby Dick","Price":8.99}`, false},
		{"recurarray", "{..Book[2]}", storeData,
			`{"Category":"fiction","Author":"Herman Melville","Title":"Moby Dick","Price":8.99}`, false},
		{"bool", "{.Bicycle[?(@.IsNew==true)]}", storeData, `{"Color":"red","Price":19.95,"IsNew":true}`, false},
	}

	testJSONPath(storeTests, false, t)

	missingKeyTests := []jsonpathTest{
		{"nonexistent field", "{.hello}", storeData, "", false},
	}
	testJSONPath(missingKeyTests, true, t)

	failStoreTests := []jsonpathTest{
		{"invalid identifier", "{hello}", storeData, "unrecognized identifier hello", false},
		{"nonexistent field", "{.hello}", storeData, "hello is not found", false},
		{"invalid array", "{.Labels[0]}", storeData, "map[string]int is not array or slice", false},
		{"invalid filter operator", "{.Book[?(@.Price<>10)]}", storeData, "unrecognized filter operator <>", false},
		{"redundant end", "{range .Labels.*}{@}{end}{end}", storeData, "not in range, nothing to end", false},
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
		{"exists filter", "{[?(@.z)].id}", pointsData, "i2 i5", false},
		{"bracket key", "{[0]['id']}", pointsData, "i1", false},
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
			"ready": true,
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
			"ready": false,
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
		{"range item", `{range .items[*]}{.metadata.name}, {end}{.kind}`, nodesData, "127.0.0.1, 127.0.0.2, List", false},
		{"range item with quote", `{range .items[*]}{.metadata.name}{"\t"}{end}`, nodesData, "127.0.0.1\t127.0.0.2\t", false},
		{"range addresss", `{.items[*].status.addresses[*].address}`, nodesData,
			"127.0.0.1 127.0.0.2 127.0.0.3", false},
		{"double range", `{range .items[*]}{range .status.addresses[*]}{.address}, {end}{end}`, nodesData,
			"127.0.0.1, 127.0.0.2, 127.0.0.3, ", false},
		{"item name", `{.items[*].metadata.name}`, nodesData, "127.0.0.1 127.0.0.2", false},
		{"union nodes capacity", `{.items[*]['metadata.name', 'status.capacity']}`, nodesData,
			`127.0.0.1 127.0.0.2 {"cpu":"4"} {"cpu":"8"}`, false},
		{"range nodes capacity", `{range .items[*]}[{.metadata.name}, {.status.capacity}] {end}`, nodesData,
			`[127.0.0.1, {"cpu":"4"}] [127.0.0.2, {"cpu":"8"}] `, false},
		{"user password", `{.users[?(@.name=="e2e")].user.password}`, &nodesData, "secret", false},
		{"hostname", `{.items[0].metadata.labels.kubernetes\.io/hostname}`, &nodesData, "127.0.0.1", false},
		{"hostname filter", `{.items[?(@.metadata.labels.kubernetes\.io/hostname=="127.0.0.1")].kind}`, &nodesData, "None", false},
		{"bool item", `{.items[?(@..ready==true)].metadata.name}`, &nodesData, "127.0.0.1", false},
	}
	testJSONPath(nodesTests, false, t)

	randomPrintOrderTests := []jsonpathTest{
		{"recursive name", "{..name}", nodesData, `127.0.0.1 127.0.0.2 myself e2e`, false},
	}
	testJSONPathSortOutput(randomPrintOrderTests, t)
}

func TestEmptyRange(t *testing.T) {
	var input = []byte(`{"items":[]}`)
	var emptyList interface{}
	err := json.Unmarshal(input, &emptyList)
	if err != nil {
		t.Error(err)
	}

	tests := []jsonpathTest{
		{"empty range", `{range .items[*]}{.metadata.name}{end}`, &emptyList, "", false},
		{"empty nested range", `{range .items[*]}{.metadata.name}{":"}{range @.spec.containers[*]}{.name}{","}{end}{"+"}{end}`, &emptyList, "", false},
	}
	testJSONPath(tests, true, t)
}

func TestNestedRanges(t *testing.T) {
	var input = []byte(`{
		"items": [
			{
				"metadata": {
					"name": "pod1"
				},
				"spec": {
					"containers": [
						{
							"name": "foo",
							"another": [
								{ "name": "value1" },
								{ "name": "value2" }
							]
						},
						{
							"name": "bar",
							"another": [
								{ "name": "value1" },
								{ "name": "value2" }
							]
						}
					]
                }
			},
			{
				"metadata": {
					"name": "pod2"
				},
				"spec": {
					"containers": [
						{
							"name": "baz",
							"another": [
								{ "name": "value1" },
								{ "name": "value2" }
							]
						}
					]
                }
			}
		]
	}`)
	var data interface{}
	err := json.Unmarshal(input, &data)
	if err != nil {
		t.Error(err)
	}

	testJSONPath(
		[]jsonpathTest{
			{
				"nested range with a trailing newline",
				`{range .items[*]}` +
					`{.metadata.name}` +
					`{":"}` +
					`{range @.spec.containers[*]}` +
					`{.name}` +
					`{","}` +
					`{end}` +
					`{"+"}` +
					`{end}`,
				data,
				"pod1:foo,bar,+pod2:baz,+",
				false,
			},
		},
		false,
		t,
	)

	testJSONPath(
		[]jsonpathTest{
			{
				"nested range with a trailing character within another nested range with a trailing newline",
				`{range .items[*]}` +
					`{.metadata.name}` +
					`{"~"}` +
					`{range @.spec.containers[*]}` +
					`{.name}` +
					`{":"}` +
					`{range @.another[*]}` +
					`{.name}` +
					`{","}` +
					`{end}` +
					`{"+"}` +
					`{end}` +
					`{"#"}` +
					`{end}`,
				data,
				"pod1~foo:value1,value2,+bar:value1,value2,+#pod2~baz:value1,value2,+#",
				false,
			},
		},
		false,
		t,
	)

	testJSONPath(
		[]jsonpathTest{
			{
				"two nested ranges at the same level with a trailing newline",
				`{range .items[*]}` +
					`{.metadata.name}` +
					`{"\t"}` +
					`{range @.spec.containers[*]}` +
					`{.name}` +
					`{" "}` +
					`{end}` +
					`{"\t"}` +
					`{range @.spec.containers[*]}` +
					`{.name}` +
					`{" "}` +
					`{end}` +
					`{"\n"}` +
					`{end}`,
				data,
				"pod1\tfoo bar \tfoo bar \npod2\tbaz \tbaz \n",
				false,
			},
		},
		false,
		t,
	)
}

func TestFilterPartialMatchesSometimesMissingAnnotations(t *testing.T) {
	// for https://issues.k8s.io/45546
	var input = []byte(`{
		"kind": "List",
		"items": [
			{
				"kind": "Pod",
				"metadata": {
					"name": "pod1",
					"annotations": {
						"color": "blue"
					}
				}
			},
			{
				"kind": "Pod",
				"metadata": {
					"name": "pod2"
				}
			},
			{
				"kind": "Pod",
				"metadata": {
					"name": "pod3",
					"annotations": {
						"color": "green"
					}
				}
			},
			{
				"kind": "Pod",
				"metadata": {
					"name": "pod4",
					"annotations": {
						"color": "blue"
					}
				}
			}
		]
	}`)
	var data interface{}
	err := json.Unmarshal(input, &data)
	if err != nil {
		t.Fatal(err)
	}

	testJSONPath(
		[]jsonpathTest{
			{
				"filter, should only match a subset, some items don't have annotations, tolerate missing items",
				`{.items[?(@.metadata.annotations.color=="blue")].metadata.name}`,
				data,
				"pod1 pod4",
				false, // expect no error
			},
		},
		true, // allow missing keys
		t,
	)

	testJSONPath(
		[]jsonpathTest{
			{
				"filter, should only match a subset, some items don't have annotations, error on missing items",
				`{.items[?(@.metadata.annotations.color=="blue")].metadata.name}`,
				data,
				"",
				true, // expect an error
			},
		},
		false, // don't allow missing keys
		t,
	)
}

func TestNegativeIndex(t *testing.T) {
	var input = []byte(
		`{
			"apiVersion": "v1",
			"kind": "Pod",
			"spec": {
				"containers": [
					{
						"image": "radial/busyboxplus:curl",
						"name": "fake0"
					},
					{
						"image": "radial/busyboxplus:curl",
						"name": "fake1"
					},
					{
						"image": "radial/busyboxplus:curl",
						"name": "fake2"
					},
					{
						"image": "radial/busyboxplus:curl",
						"name": "fake3"
					}]}}`)

	var data interface{}
	err := json.Unmarshal(input, &data)
	if err != nil {
		t.Fatal(err)
	}

	testJSONPath(
		[]jsonpathTest{
			{
				"test containers[0], it equals containers[0]",
				`{.spec.containers[0].name}`,
				data,
				"fake0",
				false,
			},
			{
				"test containers[0:0], it equals the empty set",
				`{.spec.containers[0:0].name}`,
				data,
				"",
				false,
			},
			{
				"test containers[0:-1], it equals containers[0:3]",
				`{.spec.containers[0:-1].name}`,
				data,
				"fake0 fake1 fake2",
				false,
			},
			{
				"test containers[-1:0], expect error",
				`{.spec.containers[-1:0].name}`,
				data,
				"",
				true,
			},
			{
				"test containers[-1], it equals containers[3]",
				`{.spec.containers[-1].name}`,
				data,
				"fake3",
				false,
			},
			{
				"test containers[-1:], it equals containers[3:]",
				`{.spec.containers[-1:].name}`,
				data,
				"fake3",
				false,
			},
			{
				"test containers[-2], it equals containers[2]",
				`{.spec.containers[-2].name}`,
				data,
				"fake2",
				false,
			},
			{
				"test containers[-2:], it equals containers[2:]",
				`{.spec.containers[-2:].name}`,
				data,
				"fake2 fake3",
				false,
			},
			{
				"test containers[-3], it equals containers[1]",
				`{.spec.containers[-3].name}`,
				data,
				"fake1",
				false,
			},
			{
				"test containers[-4], it equals containers[0]",
				`{.spec.containers[-4].name}`,
				data,
				"fake0",
				false,
			},
			{
				"test containers[-4:], it equals containers[0:]",
				`{.spec.containers[-4:].name}`,
				data,
				"fake0 fake1 fake2 fake3",
				false,
			},
			{
				"test containers[-5], expect a error cause it out of bounds",
				`{.spec.containers[-5].name}`,
				data,
				"",
				true, // expect error
			},
			{
				"test containers[5:5], expect empty set",
				`{.spec.containers[5:5].name}`,
				data,
				"",
				false,
			},
			{
				"test containers[-5:-5], expect empty set",
				`{.spec.containers[-5:-5].name}`,
				data,
				"",
				false,
			},
			{
				"test containers[3:1], expect a error cause start index is greater than end index",
				`{.spec.containers[3:1].name}`,
				data,
				"",
				true,
			},
			{
				"test containers[-1:-2], it equals containers[3:2], expect a error cause start index is greater than end index",
				`{.spec.containers[-1:-2].name}`,
				data,
				"",
				true,
			},
		},
		false,
		t,
	)
}

func TestStep(t *testing.T) {
	var input = []byte(
		`{
			"apiVersion": "v1",
			"kind": "Pod",
			"spec": {
				"containers": [
					{
						"image": "radial/busyboxplus:curl",
						"name": "fake0"
					},
					{
						"image": "radial/busyboxplus:curl",
						"name": "fake1"
					},
					{
						"image": "radial/busyboxplus:curl",
						"name": "fake2"
					},
					{
						"image": "radial/busyboxplus:curl",
						"name": "fake3"
					},
					{
						"image": "radial/busyboxplus:curl",
						"name": "fake4"
					},
					{
						"image": "radial/busyboxplus:curl",
						"name": "fake5"
					}]}}`)

	var data interface{}
	err := json.Unmarshal(input, &data)
	if err != nil {
		t.Fatal(err)
	}

	testJSONPath(
		[]jsonpathTest{
			{
				"test containers[0:], it equals containers[0:6:1]",
				`{.spec.containers[0:].name}`,
				data,
				"fake0 fake1 fake2 fake3 fake4 fake5",
				false,
			},
			{
				"test containers[0:6:], it equals containers[0:6:1]",
				`{.spec.containers[0:6:].name}`,
				data,
				"fake0 fake1 fake2 fake3 fake4 fake5",
				false,
			},
			{
				"test containers[0:6:1]",
				`{.spec.containers[0:6:1].name}`,
				data,
				"fake0 fake1 fake2 fake3 fake4 fake5",
				false,
			},
			{
				"test containers[0:6:0], it errors",
				`{.spec.containers[0:6:0].name}`,
				data,
				"",
				true,
			},
			{
				"test containers[0:6:-1], it errors",
				`{.spec.containers[0:6:-1].name}`,
				data,
				"",
				true,
			},
			{
				"test containers[1:4:2]",
				`{.spec.containers[1:4:2].name}`,
				data,
				"fake1 fake3",
				false,
			},
			{
				"test containers[1:4:3]",
				`{.spec.containers[1:4:3].name}`,
				data,
				"fake1",
				false,
			},
			{
				"test containers[1:4:4]",
				`{.spec.containers[1:4:4].name}`,
				data,
				"fake1",
				false,
			},
			{
				"test containers[0:6:2]",
				`{.spec.containers[0:6:2].name}`,
				data,
				"fake0 fake2 fake4",
				false,
			},
			{
				"test containers[0:6:3]",
				`{.spec.containers[0:6:3].name}`,
				data,
				"fake0 fake3",
				false,
			},
			{
				"test containers[0:6:5]",
				`{.spec.containers[0:6:5].name}`,
				data,
				"fake0 fake5",
				false,
			},
			{
				"test containers[0:6:6]",
				`{.spec.containers[0:6:6].name}`,
				data,
				"fake0",
				false,
			},
		},
		false,
		t,
	)
}
