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

package print

import (
	"strings"
	"testing"
)

func TestPrettyPrintObject(t *testing.T) {
	testCases := []struct {
		input0 interface{}
		input1 interface{}
		input2 interface{}
		output []string
	}{
		{
			"expected",
			map[string]string{
				"testkey1": "testvalue1",
			},
			map[string]string{
				"outkey1": "outkey2",
			},
			[]string{"testkey1", "testvalue1", "outkey1", "outkey2"},
		},
		{
			"Deployment",
			"deploymentName",
			struct {
				apiVersion string
				metadata   map[string]string
				labels     []string
				spec       interface{}
			}{
				apiVersion: "apps/v1beta2",
				metadata:   map[string]string{"key": "value"},
				labels:     []string{"app=test, service=test"},
				spec: struct {
					objName  string
					objValue string
				}{
					"objn",
					"objv",
				},
			},
			[]string{"Deployment", "deploymentName", "apiVersion:apps/v1beta2", "metadata:map[key:value]", "labels:[app=test, service=test]", "spec:{objName:objn objValue:objv}"},
		},
		{
			"Replica Set",
			"RSName",
			struct {
				kind     string
				metadata map[string]string
				list     []string
				spec     interface{}
			}{
				kind:     "rs",
				metadata: map[string]string{"key": "value"},
				list:     []string{"list, array"},
				spec: struct {
					replicas string
					selector []string
					template interface{}
				}{
					"3",
					[]string{"app", "service"},
					struct {
						containers string
					}{
						"nginx",
					},
				},
			},
			[]string{"Replica Set", "RSName", "kind:rs", "metadata:map[key:value]", "list:[list, array]", "replicas:3", "selector:[app service]", "template:{containers:nginx}"},
		},
	}

	for i, tc := range testCases {
		out := PrettyPrintObject(tc.input0.(string), tc.input1, tc.input2)
		for _, s := range tc.output {
			if !strings.Contains(out, s) {
				t.Errorf("case[%d] (%v) test failed: ", i, tc.input0.(string))
			}
		}

	}
}

func TestPrettyPrintQuotedObject(t *testing.T) {

	testCases := []struct {
		name string
		prop interface{}
	}{
		{
			name: "testCase1",
			prop: struct {
				kind     string
				metadata interface{}
				spec     interface{}
			}{
				kind: "'subStruct'",
				metadata: struct {
					labels interface{}
				}{
					labels: []string{"key1='value1'", "key2='value2'", "key3='value3'"},
				},
				spec: map[string]string{
					"'Index'": "'Node'",
				},
			},
		},
		{
			name: "testCase2",
			prop: struct {
				list []string
				dic  interface{}
			}{
				list: []string{"index1", "index2", "index3"},
				dic: map[int]string{
					1: "Value",
				},
			},
		},
	}
	expected := map[int]string{
		0: "{\"testCase1\" {\"'subStruct'\" {[\"key1='value1'\" \"key2='value2'\" \"key3='value3'\"]} map[\"'Index'\":\"'Node'\"]}}",
		1: "{\"testCase2\" {[\"index1\" \"index2\" \"index3\"] map['\\x01':\"Value\"]}}",
	}
	for i := 0; i < len(testCases); i++ {
		out := PrettyPrintQuotedObject("TestPrintQuotedObject", testCases[i])
		if !strings.Contains(out, expected[i]) {
			t.Errorf("case[%d] : (%v) test failed: ", i, expected[i+1])
		}
	}

}

func TestPrettyPrintStructObject(t *testing.T) {
	testCases := []interface{}{
		struct {
			key   string
			value string
		}{
			"testKey",
			"testValue",
		},
		struct {
			name     string
			property string
			filed    string
		}{
			"object-name",
			"prop-test",
			"field-test",
		},
		struct {
			spec map[string]string
		}{
			map[string]string{
				"key": "value",
			},
		},
		[]string{"index1", "index2", "index3", "index4"},
	}
	expected := map[int]string{
		0: "{\"testKey\" \"testValue\"}",
		1: "{\"object-name\" \"prop-test\" \"field-test\"}",
		2: "map[\"key\":\"value\"]",
		3: "[\"index1\" \"index2\" \"index3\" \"index4\"]",
	}
	for i := 0; i < len(testCases); i++ {
		out := PrettyPrintQuotedObject("TestPrintStructObject", testCases[i])
		if !strings.Contains(out, expected[i]) {
			t.Errorf("case[%d] : (%v) test failed: ", i, expected[i+1])
		}
	}
}
