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

package util

import (
	"reflect"
	"testing"
)

func TestParseTags(t *testing.T) {
	testCases := map[string]struct {
		lines       []string
		expectTags  Tags
		expectError bool
	}{
		"genclient": {
			lines:      []string{`+genclient`},
			expectTags: Tags{GenerateClient: true},
		},
		"genclient=true": {
			lines:       []string{`+genclient=true`},
			expectError: true,
		},
		"nonNamespaced=true": {
			lines:       []string{`+genclient=true`, `+nonNamespaced=true`},
			expectError: true,
		},
		"readonly=true": {
			lines:       []string{`+genclient=true`, `+readonly=true`},
			expectError: true,
		},
		"genclient:nonNamespaced": {
			lines:      []string{`+genclient`, `+genclient:nonNamespaced`},
			expectTags: Tags{GenerateClient: true, NonNamespaced: true},
		},
		"genclient:noVerbs": {
			lines:      []string{`+genclient`, `+genclient:noVerbs`},
			expectTags: Tags{GenerateClient: true, NoVerbs: true},
		},
		"genclient:noStatus": {
			lines:      []string{`+genclient`, `+genclient:noStatus`},
			expectTags: Tags{GenerateClient: true, NoStatus: true},
		},
		"genclient:onlyVerbs": {
			lines:      []string{`+genclient`, `+genclient:onlyVerbs=create,delete`},
			expectTags: Tags{GenerateClient: true, SkipVerbs: []string{"update", "updateStatus", "deleteCollection", "get", "list", "watch", "patch", "apply", "applyStatus"}},
		},
		"genclient:readonly": {
			lines:      []string{`+genclient`, `+genclient:readonly`},
			expectTags: Tags{GenerateClient: true, SkipVerbs: []string{"create", "update", "updateStatus", "delete", "deleteCollection", "patch", "apply", "applyStatus"}},
		},
		"genclient:conflict": {
			lines:       []string{`+genclient`, `+genclient:onlyVerbs=create`, `+genclient:skipVerbs=create`},
			expectError: true,
		},
		"genclient:invalid": {
			lines:       []string{`+genclient`, `+genclient:invalid`},
			expectError: true,
		},
	}
	for key, c := range testCases {
		result, err := ParseClientGenTags(c.lines)
		if err != nil && !c.expectError {
			t.Fatalf("unexpected error: %v", err)
		}
		if !c.expectError && !reflect.DeepEqual(result, c.expectTags) {
			t.Errorf("[%s] expected %#v to be %#v", key, result, c.expectTags)
		}
	}
}

func TestParseTagsExtension(t *testing.T) {
	testCases := map[string]struct {
		lines              []string
		expectedExtensions []extension
		expectError        bool
	}{
		"simplest extension": {
			lines:              []string{`+genclient:method=Foo,verb=create`},
			expectedExtensions: []extension{{VerbName: "Foo", VerbType: "create"}},
		},
		"multiple extensions": {
			lines:              []string{`+genclient:method=Foo,verb=create`, `+genclient:method=Bar,verb=get`},
			expectedExtensions: []extension{{VerbName: "Foo", VerbType: "create"}, {VerbName: "Bar", VerbType: "get"}},
		},
		"extension without verb": {
			lines:       []string{`+genclient:method`},
			expectError: true,
		},
		"extension without verb type": {
			lines:       []string{`+genclient:method=Foo`},
			expectError: true,
		},
		"sub-resource extension": {
			lines:              []string{`+genclient:method=Foo,verb=create,subresource=bar`},
			expectedExtensions: []extension{{VerbName: "Foo", VerbType: "create", SubResourcePath: "bar"}},
		},
		"output type extension": {
			lines:              []string{`+genclient:method=Foos,verb=list,result=Bars`},
			expectedExtensions: []extension{{VerbName: "Foos", VerbType: "list", ResultTypeOverride: "Bars"}},
		},
		"input type extension": {
			lines:              []string{`+genclient:method=Foo,verb=update,input=Bar`},
			expectedExtensions: []extension{{VerbName: "Foo", VerbType: "update", InputTypeOverride: "Bar"}},
		},
		"unknown verb type extension": {
			lines:              []string{`+genclient:method=Foo,verb=explode`},
			expectedExtensions: nil,
			expectError:        true,
		},
		"invalid verb extension": {
			lines:              []string{`+genclient:method=Foo,unknown=bar`},
			expectedExtensions: nil,
			expectError:        true,
		},
		"empty verb extension subresource": {
			lines:              []string{`+genclient:method=Foo,verb=get,subresource=`},
			expectedExtensions: nil,
			expectError:        true,
		},
	}
	for key, c := range testCases {
		result, err := ParseClientGenTags(c.lines)
		if err != nil && !c.expectError {
			t.Fatalf("[%s] unexpected error: %v", key, err)
		}
		if err != nil && c.expectError {
			t.Logf("[%s] got expected error: %+v", key, err)
		}
		if !c.expectError && !reflect.DeepEqual(result.Extensions, c.expectedExtensions) {
			t.Errorf("[%s] expected %#+v to be %#+v", key, result.Extensions, c.expectedExtensions)
		}
	}
}
