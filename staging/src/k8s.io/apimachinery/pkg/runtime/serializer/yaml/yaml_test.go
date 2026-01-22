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

package yaml

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/yaml"
	sigsyaml "sigs.k8s.io/yaml"
)

type testcase struct {
	name  string
	data  []byte
	error string

	benchmark bool
}

func testcases() []testcase {
	return []testcase{
		{
			name:  "arrays of string aliases",
			error: "excessive aliasing",
			data: []byte(`
apiVersion: v1
data:
a: &a ["webwebwebwebwebweb","webwebwebwebwebweb","webwebwebwebwebweb","webwebwebwebwebweb","webwebwebwebwebweb","webwebwebwebwebweb","webwebwebwebwebweb","webwebwebwebwebweb","webwebwebwebwebweb"]
b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a]
c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b]
d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c]
e: &e [*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d]
f: &f [*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e]
g: &g [*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f]
h: &h [*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g]
i: &i [*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h]
kind: ConfigMap
metadata:
name: yaml-bomb
namespace: default
`),
		},
		{
			name:  "arrays of empty string aliases",
			error: "excessive aliasing",
			data: []byte(`
apiVersion: v1
data:
a: &a ["","","","","","","","",""]
b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a]
c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b]
d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c]
e: &e [*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d]
f: &f [*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e]
g: &g [*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f]
h: &h [*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g]
i: &i [*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h]
kind: ConfigMap
metadata:
name: yaml-bomb
namespace: default
`),
			benchmark: true,
		},
		{
			name:  "arrays of null aliases",
			error: "excessive aliasing",
			data: []byte(`
apiVersion: v1
data:
a: &a [null,null,null,null,null,null,null,null,null]
b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a]
c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b]
d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c]
e: &e [*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d]
f: &f [*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e]
g: &g [*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f]
h: &h [*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g]
i: &i [*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h]
kind: ConfigMap
metadata:
name: yaml-bomb
namespace: default
`),
			benchmark: true,
		},
		{
			name:  "arrays of zero int aliases",
			error: "excessive aliasing",
			data: []byte(`
apiVersion: v1
data:
a: &a [0,0,0,0,0,0,0,0,0]
b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a]
c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b]
d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c]
e: &e [*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d]
f: &f [*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e]
g: &g [*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f]
h: &h [*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g]
i: &i [*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h]
kind: ConfigMap
metadata:
name: yaml-bomb
namespace: default
`),
			benchmark: true,
		},
		{
			name:  "arrays of zero float aliases",
			error: "excessive aliasing",
			data: []byte(`
apiVersion: v1
data:
a: &a [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a]
c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b]
d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c]
e: &e [*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d]
f: &f [*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e]
g: &g [*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f]
h: &h [*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g]
i: &i [*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h]
kind: ConfigMap
metadata:
name: yaml-bomb
namespace: default
`),
			benchmark: true,
		},
		{
			name:  "arrays of big float aliases",
			error: "excessive aliasing",
			data: []byte(`
apiVersion: v1
data:
a: &a [1234567890.12345678,1234567890.12345678,1234567890.12345678,1234567890.12345678,1234567890.12345678,1234567890.12345678,1234567890.12345678,1234567890.12345678,1234567890.12345678]
b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a]
c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b]
d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c]
e: &e [*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d]
f: &f [*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e]
g: &g [*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f]
h: &h [*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g]
i: &i [*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h]
kind: ConfigMap
metadata:
name: yaml-bomb
namespace: default
`),
			benchmark: true,
		},
		{
			name:  "arrays of bool aliases",
			error: "excessive aliasing",
			data: []byte(`
apiVersion: v1
data:
a: &a [true,true,true,true,true,true,true,true,true]
b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a,*a]
c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b,*b]
d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c,*c]
e: &e [*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d,*d]
f: &f [*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e,*e]
g: &g [*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f,*f]
h: &h [*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g,*g]
i: &i [*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h,*h]
kind: ConfigMap
metadata:
name: yaml-bomb
namespace: default
`),
			benchmark: true,
		},
		{
			name:  "map key aliases",
			error: "excessive aliasing",
			data: []byte(`
apiVersion: v1
data:
a: &a {"verylongkey1":"","verylongkey2":"","verylongkey3":"","verylongkey4":"","verylongkey5":"","verylongkey6":"","verylongkey7":"","verylongkey8":"","verylongkey9":""}
b: &b {"verylongkey1":*a,"verylongkey2":*a,"verylongkey3":*a,"verylongkey4":*a,"verylongkey5":*a,"verylongkey6":*a,"verylongkey7":*a,"verylongkey8":*a,"verylongkey9":*a}
c: &c {"verylongkey1":*b,"verylongkey2":*b,"verylongkey3":*b,"verylongkey4":*b,"verylongkey5":*b,"verylongkey6":*b,"verylongkey7":*b,"verylongkey8":*b,"verylongkey9":*b}
d: &d {"verylongkey1":*c,"verylongkey2":*c,"verylongkey3":*c,"verylongkey4":*c,"verylongkey5":*c,"verylongkey6":*c,"verylongkey7":*c,"verylongkey8":*c,"verylongkey9":*c}
e: &e {"verylongkey1":*d,"verylongkey2":*d,"verylongkey3":*d,"verylongkey4":*d,"verylongkey5":*d,"verylongkey6":*d,"verylongkey7":*d,"verylongkey8":*d,"verylongkey9":*d}
f: &f {"verylongkey1":*e,"verylongkey2":*e,"verylongkey3":*e,"verylongkey4":*e,"verylongkey5":*e,"verylongkey6":*e,"verylongkey7":*e,"verylongkey8":*e,"verylongkey9":*e}
g: &g {"verylongkey1":*f,"verylongkey2":*f,"verylongkey3":*f,"verylongkey4":*f,"verylongkey5":*f,"verylongkey6":*f,"verylongkey7":*f,"verylongkey8":*f,"verylongkey9":*f}
h: &h {"verylongkey1":*g,"verylongkey2":*g,"verylongkey3":*g,"verylongkey4":*g,"verylongkey5":*g,"verylongkey6":*g,"verylongkey7":*g,"verylongkey8":*g,"verylongkey9":*g}
i: &i {"verylongkey1":*h,"verylongkey2":*h,"verylongkey3":*h,"verylongkey4":*h,"verylongkey5":*h,"verylongkey6":*h,"verylongkey7":*h,"verylongkey8":*h,"verylongkey9":*h}
kind: ConfigMap
metadata:
name: yaml-bomb
namespace: default
`),
			benchmark: true,
		},
		{
			name:  "map value aliases",
			error: "excessive aliasing",
			data: []byte(`
apiVersion: v1
data:
a: &a {"1":"verylongmapvalue","2":"verylongmapvalue","3":"verylongmapvalue","4":"verylongmapvalue","5":"verylongmapvalue","6":"verylongmapvalue","7":"verylongmapvalue","8":"verylongmapvalue","9":"verylongmapvalue"}
b: &b {"1":*a,"2":*a,"3":*a,"4":*a,"5":*a,"6":*a,"7":*a,"8":*a,"9":*a}
c: &c {"1":*b,"2":*b,"3":*b,"4":*b,"5":*b,"6":*b,"7":*b,"8":*b,"9":*b}
d: &d {"1":*c,"2":*c,"3":*c,"4":*c,"5":*c,"6":*c,"7":*c,"8":*c,"9":*c}
e: &e {"1":*d,"2":*d,"3":*d,"4":*d,"5":*d,"6":*d,"7":*d,"8":*d,"9":*d}
f: &f {"1":*e,"2":*e,"3":*e,"4":*e,"5":*e,"6":*e,"7":*e,"8":*e,"9":*e}
g: &g {"1":*f,"2":*f,"3":*f,"4":*f,"5":*f,"6":*f,"7":*f,"8":*f,"9":*f}
h: &h {"1":*g,"2":*g,"3":*g,"4":*g,"5":*g,"6":*g,"7":*g,"8":*g,"9":*g}
i: &i {"1":*h,"2":*h,"3":*h,"4":*h,"5":*h,"6":*h,"7":*h,"8":*h,"9":*h}
kind: ConfigMap
metadata:
name: yaml-bomb
namespace: default
`),
			benchmark: true,
		},
		{
			name:  "nested map aliases",
			error: "excessive aliasing",
			data: []byte(`
apiVersion: v1
data:
a: &a {"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
b: &b {"1":*a,"2":*a,"3":*a,"4":*a,"5":*a,"6":*a,"7":*a,"8":*a,"9":*a}
c: &c {"1":*b,"2":*b,"3":*b,"4":*b,"5":*b,"6":*b,"7":*b,"8":*b,"9":*b}
d: &d {"1":*c,"2":*c,"3":*c,"4":*c,"5":*c,"6":*c,"7":*c,"8":*c,"9":*c}
e: &e {"1":*d,"2":*d,"3":*d,"4":*d,"5":*d,"6":*d,"7":*d,"8":*d,"9":*d}
f: &f {"1":*e,"2":*e,"3":*e,"4":*e,"5":*e,"6":*e,"7":*e,"8":*e,"9":*e}
g: &g {"1":*f,"2":*f,"3":*f,"4":*f,"5":*f,"6":*f,"7":*f,"8":*f,"9":*f}
h: &h {"1":*g,"2":*g,"3":*g,"4":*g,"5":*g,"6":*g,"7":*g,"8":*g,"9":*g}
i: &i {"1":*h,"2":*h,"3":*h,"4":*h,"5":*h,"6":*h,"7":*h,"8":*h,"9":*h}
kind: ConfigMap
metadata:
name: yaml-bomb
namespace: default
`),
			benchmark: true,
		},
		{
			name:  "nested slice aliases",
			error: "excessive aliasing",
			data: []byte(`
apiVersion: v1
data:
a: &a [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[""]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
b: &b [[[[[[[[[[*a]]]]]]]]],[[[[[[[[[*a]]]]]]]]],[[[[[[[[[*a]]]]]]]]],[[[[[[[[[*a]]]]]]]]],[[[[[[[[[*a]]]]]]]]]]
c: &c [[[[[[[[[[*b]]]]]]]]],[[[[[[[[[*b]]]]]]]]],[[[[[[[[[*b]]]]]]]]],[[[[[[[[[*b]]]]]]]]],[[[[[[[[[*b]]]]]]]]]]
d: &d [[[[[[[[[[*c]]]]]]]]],[[[[[[[[[*c]]]]]]]]],[[[[[[[[[*c]]]]]]]]],[[[[[[[[[*c]]]]]]]]],[[[[[[[[[*c]]]]]]]]]]
e: &e [[[[[[[[[[*d]]]]]]]]],[[[[[[[[[*d]]]]]]]]],[[[[[[[[[*d]]]]]]]]],[[[[[[[[[*d]]]]]]]]],[[[[[[[[[*d]]]]]]]]]]
f: &f [[[[[[[[[[*e]]]]]]]]],[[[[[[[[[*e]]]]]]]]],[[[[[[[[[*e]]]]]]]]],[[[[[[[[[*e]]]]]]]]],[[[[[[[[[*e]]]]]]]]]]
g: &g [[[[[[[[[[*f]]]]]]]]],[[[[[[[[[*f]]]]]]]]],[[[[[[[[[*f]]]]]]]]],[[[[[[[[[*f]]]]]]]]],[[[[[[[[[*f]]]]]]]]]]
h: &h [[[[[[[[[[*g]]]]]]]]],[[[[[[[[[*g]]]]]]]]],[[[[[[[[[*g]]]]]]]]],[[[[[[[[[*g]]]]]]]]],[[[[[[[[[*g]]]]]]]]]]
i: &i [[[[[[[[[[*h]]]]]]]]],[[[[[[[[[*h]]]]]]]]],[[[[[[[[[*h]]]]]]]]],[[[[[[[[[*h]]]]]]]]],[[[[[[[[[*h]]]]]]]]]]
kind: ConfigMap
metadata:
name: yaml-bomb
namespace: default
`),
			benchmark: true,
		},
		{
			name:      "3MB map without alias",
			data:      []byte(`a: &a [{a}` + strings.Repeat(`,{a}`, 3*1024*1024/4) + `]`),
			benchmark: true,
		},
		{
			name:  "3MB map with alias",
			error: "excessive aliasing",
			data: []byte(`
a: &a [{a}` + strings.Repeat(`,{a}`, 3*1024*1024/4) + `]
b: &b [*a]`),
			benchmark: true,
		},
		{
			name:  "deeply nested slices",
			error: "max depth",
			data:  []byte(strings.Repeat(`[`, 3*1024*1024)),
		},
		{
			name:  "deeply nested maps",
			error: "max depth",
			data:  []byte("x: " + strings.Repeat(`{`, 3*1024*1024)),
		},
		{
			name:  "deeply nested indents",
			error: "max depth",
			data:  []byte(strings.Repeat(`- `, 3*1024*1024)),
		},
		{
			name:      "3MB of 1000-indent lines",
			data:      []byte(strings.Repeat(strings.Repeat(`- `, 1000)+"\n", 3*1024/2)),
			benchmark: true,
		},
		{
			name:      "3MB of empty slices",
			data:      []byte(`[` + strings.Repeat(`[],`, 3*1024*1024/3-2) + `[]]`),
			benchmark: true,
		},
		{
			name:      "3MB of slices",
			data:      []byte(`[` + strings.Repeat(`[0],`, 3*1024*1024/4-2) + `[0]]`),
			benchmark: true,
		},
		{
			name:      "3MB of empty maps",
			data:      []byte(`[` + strings.Repeat(`{},`, 3*1024*1024/3-2) + `{}]`),
			benchmark: true,
		},
		{
			name:      "3MB of maps",
			data:      []byte(`[` + strings.Repeat(`{a},`, 3*1024*1024/4-2) + `{a}]`),
			benchmark: true,
		},
		{
			name:      "3MB of ints",
			data:      []byte(`[` + strings.Repeat(`0,`, 3*1024*1024/2-2) + `0]`),
			benchmark: true,
		},
		{
			name:      "3MB of floats",
			data:      []byte(`[` + strings.Repeat(`0.0,`, 3*1024*1024/4-2) + `0.0]`),
			benchmark: true,
		},
		{
			name:      "3MB of bools",
			data:      []byte(`[` + strings.Repeat(`true,`, 3*1024*1024/5-2) + `true]`),
			benchmark: true,
		},
		{
			name:      "3MB of empty strings",
			data:      []byte(`[` + strings.Repeat(`"",`, 3*1024*1024/3-2) + `""]`),
			benchmark: true,
		},
		{
			name:      "3MB of strings",
			data:      []byte(`[` + strings.Repeat(`"abcdefghijklmnopqrstuvwxyz012",`, 3*1024*1024/30-2) + `"abcdefghijklmnopqrstuvwxyz012"]`),
			benchmark: true,
		},
		{
			name:      "3MB of nulls",
			data:      []byte(`[` + strings.Repeat(`null,`, 3*1024*1024/5-2) + `null]`),
			benchmark: true,
		},
	}
}

var decoders = map[string]func([]byte) ([]byte, error){
	"sigsyaml": sigsyaml.YAMLToJSON,
	"utilyaml": yaml.ToJSON,
}

func TestYAMLLimits(t *testing.T) {
	for _, tc := range testcases() {
		if tc.benchmark {
			continue
		}
		t.Run(tc.name, func(t *testing.T) {
			for decoderName, decoder := range decoders {
				t.Run(decoderName, func(t *testing.T) {
					_, err := decoder(tc.data)
					if len(tc.error) == 0 {
						if err != nil {
							t.Errorf("unexpected error: %v", err)
						}
					} else {
						if err == nil || !strings.Contains(err.Error(), tc.error) {
							t.Errorf("expected %q error, got %v", tc.error, err)
						}
					}
				})
			}
		})
	}
}

func BenchmarkYAMLLimits(b *testing.B) {
	for _, tc := range testcases() {
		b.Run(tc.name, func(b *testing.B) {
			for decoderName, decoder := range decoders {
				b.Run(decoderName, func(b *testing.B) {
					for i := 0; i < b.N; i++ {
						_, err := decoder(tc.data)
						if len(tc.error) == 0 {
							if err != nil {
								b.Errorf("unexpected error: %v", err)
							}
						} else {
							if err == nil || !strings.Contains(err.Error(), tc.error) {
								b.Errorf("expected %q error, got %v", tc.error, err)
							}
						}
					}
				})
			}
		})
	}
}
