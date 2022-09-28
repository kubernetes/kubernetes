/*
Copyright 2022 The Kubernetes Authors.

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

package test

// ZaprOutputMappingDirect provides a mapping from klog output to the
// corresponding zapr output when zapr is called directly.
//
// Experimental
//
// Notice: This package is EXPERIMENTAL and may be changed or removed in a
// later release.
func ZaprOutputMappingDirect() map[string]string {
	return map[string]string{
		`I output.go:<LINE>] "test" akey="<&>"
`: `{"caller":"test/output.go:<LINE>","msg":"test","v":0,"akey":"<&>"}
`,

		`E output.go:<LINE>] "test" err="whoops"
`: `{"caller":"test/output.go:<LINE>","msg":"test","err":"whoops"}
`,

		`I output.go:<LINE>] "helper" akey="avalue"
`: `{"caller":"test/output.go:<LINE>","msg":"helper","v":0,"akey":"avalue"}
`,

		`I output.go:<LINE>] "test" akey="avalue" akey="avalue2"
`: `{"caller":"test/output.go:<LINE>","msg":"test","v":0,"akey":"avalue","akey":"avalue2"}
`,

		`I output.go:<LINE>] "hello/world: test" akey="avalue"
`: `{"logger":"hello.world","caller":"test/output.go:<LINE>","msg":"test","v":0,"akey":"avalue"}
`,

		`I output.go:<LINE>] "test" X="y" duration="1m0s" A="b"
`: `{"caller":"test/output.go:<LINE>","msg":"test","duration":"1h0m0s","X":"y","v":0,"duration":"1m0s","A":"b"}
`,

		`I output.go:<LINE>] "test" akey9="avalue9" akey8="avalue8" akey1="avalue1" akey5="avalue5" akey4="avalue4"
`: `{"caller":"test/output.go:<LINE>","msg":"test","akey9":"avalue9","akey8":"avalue8","akey1":"avalue1","v":0,"akey5":"avalue5","akey4":"avalue4"}
`,

		`I output.go:<LINE>] "test"
`: `{"caller":"test/output.go:<LINE>","msg":"test","v":0}
`,

		`I output.go:<LINE>] "\"quoted\"" key="\"quoted value\""
`: `{"caller":"test/output.go:<LINE>","msg":"\"quoted\"","v":0,"key":"\"quoted value\""}
`,

		`I output.go:<LINE>] "test" err="whoops"
`: `{"caller":"test/output.go:<LINE>","msg":"test","v":0,"err":"whoops"}
`,

		`I output.go:<LINE>] "test" pod="kube-system/pod-1"
`: `{"caller":"test/output.go:<LINE>","msg":"test","v":0,"pod":{"name":"pod-1","namespace":"kube-system"}}
`,

		`I output.go:<LINE>] "test" pods=[kube-system/pod-1 kube-system/pod-2]
`: `{"caller":"test/output.go:<LINE>","msg":"test","v":0,"pods":[{"name":"pod-1","namespace":"kube-system"},{"name":"pod-2","namespace":"kube-system"}]}
`,

		`I output.go:<LINE>] "test" pods="[kube-system/pod-1 kube-system/pod-2]"
`: `{"caller":"test/output.go:<LINE>","msg":"test","v":0,"pods":[{"name":"pod-1","namespace":"kube-system"},{"name":"pod-2","namespace":"kube-system"}]}
`,

		`I output.go:<LINE>] "test" pods="[]"
`: `{"caller":"test/output.go:<LINE>","msg":"test","v":0,"pods":null}
`,

		`I output.go:<LINE>] "test" pods="<KObjSlice needs a slice, got type int>"
`: `{"caller":"test/output.go:<LINE>","msg":"test","v":0,"pods":"<KObjSlice needs a slice, got type int>"}
`,

		`I output.go:<LINE>] "test" ints="<KObjSlice needs a slice of values implementing KMetadata, got type int>"
`: `{"caller":"test/output.go:<LINE>","msg":"test","v":0,"ints":"<KObjSlice needs a slice of values implementing KMetadata, got type int>"}
`,

		`I output.go:<LINE>] "test" pods="[kube-system/pod-1 <nil>]"
`: `{"caller":"test/output.go:<LINE>","msg":"test","v":0,"pods":[{"name":"pod-1","namespace":"kube-system"},null]}
`,

		`I output.go:<LINE>] "test" akey="avalue"
`: `{"caller":"test/output.go:<LINE>","msg":"test","v":0,"akey":"avalue"}
`,

		`I output.go:<LINE>] "me: test" akey="avalue"
`: `{"logger":"me","caller":"test/output.go:<LINE>","msg":"test","v":0,"akey":"avalue"}
`,

		`I output.go:<LINE>] "test" akey="avalue2"
`: `{"caller":"test/output.go:<LINE>","msg":"test","akey":"avalue","v":0,"akey":"avalue2"}
`,

		`I output.go:<LINE>] "you see me"
`: `{"caller":"test/output.go:<LINE>","msg":"you see me","v":9}
`,

		`I output.go:<LINE>] "test" firstKey=1
I output.go:<LINE>] "test" firstKey=1 secondKey=2
I output.go:<LINE>] "test" firstKey=1
I output.go:<LINE>] "test" firstKey=1 secondKey=3
`: `{"caller":"test/output.go:<LINE>","msg":"test","firstKey":1,"v":0}
{"caller":"test/output.go:<LINE>","msg":"test","firstKey":1,"secondKey":2,"v":0}
{"caller":"test/output.go:<LINE>","msg":"test","firstKey":1,"v":0}
{"caller":"test/output.go:<LINE>","msg":"test","firstKey":1,"secondKey":3,"v":0}
`,

		`I output.go:<LINE>] "odd WithValues" keyWithoutValue="(MISSING)"
I output.go:<LINE>] "odd WithValues" keyWithoutValue="(MISSING)" anotherKeyWithoutValue="(MISSING)"
I output.go:<LINE>] "odd WithValues" keyWithoutValue="(MISSING)"
`: `{"caller":"test/output.go:<WITH-VALUES>","msg":"odd number of arguments passed as key-value pairs for logging","ignored key":"keyWithoutValue"}
{"caller":"test/output.go:<WITH-VALUES-2>","msg":"odd number of arguments passed as key-value pairs for logging","ignored key":"anotherKeyWithoutValue"}
{"caller":"test/output.go:<LINE>","msg":"odd WithValues","v":0}
{"caller":"test/output.go:<LINE>","msg":"odd WithValues","v":0}
{"caller":"test/output.go:<LINE>","msg":"odd WithValues","v":0}
`,

		`I output.go:<LINE>] "odd arguments" akey="avalue" akey2="(MISSING)"
`: `{"caller":"test/output.go:<LINE>","msg":"odd number of arguments passed as key-value pairs for logging","ignored key":"akey2"}
{"caller":"test/output.go:<LINE>","msg":"odd arguments","v":0,"akey":"avalue"}
`,

		`I output.go:<LINE>] "both odd" basekey1="basevar1" basekey2="(MISSING)" akey="avalue" akey2="(MISSING)"
`: `{"caller":"test/output.go:<WITH-VALUES>","msg":"odd number of arguments passed as key-value pairs for logging","ignored key":"basekey2"}
{"caller":"test/output.go:<LINE>","msg":"odd number of arguments passed as key-value pairs for logging","basekey1":"basevar1","ignored key":"akey2"}
{"caller":"test/output.go:<LINE>","msg":"both odd","basekey1":"basevar1","v":0,"akey":"avalue"}
`,

		`I output.go:<LINE>] "marshaler nil" obj="<panic: value method k8s.io/klog/v2.ObjectRef.String called using nil *ObjectRef pointer>"
`: `{"caller":"test/output.go:<LINE>","msg":"marshaler nil","v":0,"objError":"PANIC=value method k8s.io/klog/v2.ObjectRef.MarshalLog called using nil *ObjectRef pointer"}
`,

		// zap replaces a panic for a nil object with <nil>.
		`E output.go:<LINE>] "error nil" err="<panic: runtime error: invalid memory address or nil pointer dereference>"
`: `{"caller":"test/output.go:<LINE>","msg":"error nil","err":"<nil>"}
`,

		`I output.go:<LINE>] "stringer nil" stringer="<panic: runtime error: invalid memory address or nil pointer dereference>"
`: `{"caller":"test/output.go:<LINE>","msg":"stringer nil","v":0,"stringer":"<nil>"}
`,

		`I output.go:<LINE>] "stringer panic" stringer="<panic: fake String panic>"
`: `{"caller":"test/output.go:<LINE>","msg":"stringer panic","v":0,"stringerError":"PANIC=fake String panic"}
`,

		`E output.go:<LINE>] "error panic" err="<panic: fake Error panic>"
`: `{"caller":"test/output.go:<LINE>","msg":"error panic","errError":"PANIC=fake Error panic"}
`,

		`I output.go:<LINE>] "marshaler panic" obj="<panic: fake MarshalLog panic>"
`: `{"caller":"test/output.go:<LINE>","msg":"marshaler panic","v":0,"objError":"PANIC=fake MarshalLog panic"}
`,

		`I output.go:<LINE>] "marshaler recursion" obj={}
`: `{"caller":"test/output.go:<LINE>","msg":"marshaler recursion","v":0,"obj":{}}
`,

		// klog.Info
		`I output.go:<LINE>] "helloworld\n"
`: `{"caller":"test/output.go:<LINE>","msg":"helloworld\n","v":0}
`,

		// klog.Infoln
		`I output.go:<LINE>] "hello world\n"
`: `{"caller":"test/output.go:<LINE>","msg":"hello world\n","v":0}
`,

		// klog.Error
		`E output.go:<LINE>] "helloworld\n"
`: `{"caller":"test/output.go:<LINE>","msg":"helloworld\n"}
`,

		// klog.Errorln
		`E output.go:<LINE>] "hello world\n"
`: `{"caller":"test/output.go:<LINE>","msg":"hello world\n"}
`,

		// klog.ErrorS
		`E output.go:<LINE>] "world" err="hello"
`: `{"caller":"test/output.go:<LINE>","msg":"world","err":"hello"}
`,

		// klog.InfoS
		`I output.go:<LINE>] "hello" what="world"
`: `{"caller":"test/output.go:<LINE>","msg":"hello","v":0,"what":"world"}
`,

		// klog.V(1).Info
		`I output.go:<LINE>] "hellooneworld\n"
`: `{"caller":"test/output.go:<LINE>","msg":"hellooneworld\n","v":1}
`,

		// klog.V(1).Infoln
		`I output.go:<LINE>] "hello one world\n"
`: `{"caller":"test/output.go:<LINE>","msg":"hello one world\n","v":1}
`,

		// klog.V(1).ErrorS
		`E output.go:<LINE>] "one world" err="hello"
`: `{"caller":"test/output.go:<LINE>","msg":"one world","err":"hello"}
`,

		// klog.V(1).InfoS
		`I output.go:<LINE>] "hello" what="one world"
`: `{"caller":"test/output.go:<LINE>","msg":"hello","v":1,"what":"one world"}
`,

		`I output.go:<LINE>] "integer keys" %!s(int=1)="value" %!s(int=2)="value2" akey="avalue" akey2="(MISSING)"
`: `{"caller":"test/output.go:<WITH-VALUES>","msg":"non-string key argument passed to logging, ignoring all later arguments","invalid key":1}
{"caller":"test/output.go:<LINE>","msg":"odd number of arguments passed as key-value pairs for logging","ignored key":"akey2"}
{"caller":"test/output.go:<LINE>","msg":"integer keys","v":0,"akey":"avalue"}
`,

		`I output.go:<LINE>] "struct keys" {name}="value" test="other value" key="val"
`: `{"caller":"test/output.go:<WITH-VALUES>","msg":"non-string key argument passed to logging, ignoring all later arguments","invalid key":{}}
{"caller":"test/output.go:<LINE>","msg":"struct keys","v":0,"key":"val"}
`,
		`I output.go:<LINE>] "map keys" map[test:%!s(bool=true)]="test"
`: `{"caller":"test/output.go:<LINE>","msg":"non-string key argument passed to logging, ignoring all later arguments","invalid key":{"test":true}}
{"caller":"test/output.go:<LINE>","msg":"map keys","v":0}
`,
	}
}

// ZaprOutputMappingIndirect provides a mapping from klog output to the
// corresponding zapr output when zapr is called indirectly through
// klog.
//
// This is different from ZaprOutputMappingDirect because:
// - WithName gets added to the message by Output.
// - zap uses . as separator instead of / between WithName values,
//   here we get slashes because Output concatenates these values.
// - WithValues are added to the normal key/value parameters by
//   Output, which puts them after "v".
// - Output does that without emitting the warning that we get
//   from zapr.
// - zap drops keys with missing values, here we get "(MISSING)".
// - zap does not de-duplicate key/value pairs, here klog does that
//   for it.
//
// Experimental
//
// Notice: This package is EXPERIMENTAL and may be changed or removed in a
// later release.
func ZaprOutputMappingIndirect() map[string]string {
	mapping := ZaprOutputMappingDirect()

	for key, value := range map[string]string{
		`I output.go:<LINE>] "hello/world: test" akey="avalue"
`: `{"caller":"test/output.go:<LINE>","msg":"hello/world: test","v":0,"akey":"avalue"}
`,

		`I output.go:<LINE>] "me: test" akey="avalue"
`: `{"caller":"test/output.go:<LINE>","msg":"me: test","v":0,"akey":"avalue"}
`,

		`I output.go:<LINE>] "odd parameters" basekey1="basevar1" basekey2="(MISSING)" akey="avalue" akey2="(MISSING)"
`: `{"caller":"test/output.go:<LINE>","msg":"odd number of arguments passed as key-value pairs for logging","ignored key":"akey2"}
{"caller":"test/output.go:<LINE>","msg":"odd parameters","v":0,"basekey1":"basevar1","basekey2":"(MISSING)","akey":"avalue"}
`,

		`I output.go:<LINE>] "odd WithValues" keyWithoutValue="(MISSING)"
I output.go:<LINE>] "odd WithValues" keyWithoutValue="(MISSING)" anotherKeyWithoutValue="(MISSING)"
I output.go:<LINE>] "odd WithValues" keyWithoutValue="(MISSING)"
`: `{"caller":"test/output.go:<LINE>","msg":"odd WithValues","v":0,"keyWithoutValue":"(MISSING)"}
{"caller":"test/output.go:<LINE>","msg":"odd WithValues","v":0,"keyWithoutValue":"(MISSING)","anotherKeyWithoutValue":"(MISSING)"}
{"caller":"test/output.go:<LINE>","msg":"odd WithValues","v":0,"keyWithoutValue":"(MISSING)"}
`,

		`I output.go:<LINE>] "both odd" basekey1="basevar1" basekey2="(MISSING)" akey="avalue" akey2="(MISSING)"
`: `{"caller":"test/output.go:<LINE>","msg":"odd number of arguments passed as key-value pairs for logging","ignored key":"akey2"}
{"caller":"test/output.go:<LINE>","msg":"both odd","v":0,"basekey1":"basevar1","basekey2":"(MISSING)","akey":"avalue"}
`,

		`I output.go:<LINE>] "test" akey9="avalue9" akey8="avalue8" akey1="avalue1" akey5="avalue5" akey4="avalue4"
`: `{"caller":"test/output.go:<LINE>","msg":"test","v":0,"akey9":"avalue9","akey8":"avalue8","akey1":"avalue1","akey5":"avalue5","akey4":"avalue4"}
`,

		`I output.go:<LINE>] "test" akey="avalue2"
`: `{"caller":"test/output.go:<LINE>","msg":"test","v":0,"akey":"avalue2"}
`,

		`I output.go:<LINE>] "test" X="y" duration="1m0s" A="b"
`: `{"caller":"test/output.go:<LINE>","msg":"test","v":0,"X":"y","duration":"1m0s","A":"b"}
`,

		`I output.go:<LINE>] "test" firstKey=1
I output.go:<LINE>] "test" firstKey=1 secondKey=2
I output.go:<LINE>] "test" firstKey=1
I output.go:<LINE>] "test" firstKey=1 secondKey=3
`: `{"caller":"test/output.go:<LINE>","msg":"test","v":0,"firstKey":1}
{"caller":"test/output.go:<LINE>","msg":"test","v":0,"firstKey":1,"secondKey":2}
{"caller":"test/output.go:<LINE>","msg":"test","v":0,"firstKey":1}
{"caller":"test/output.go:<LINE>","msg":"test","v":0,"firstKey":1,"secondKey":3}
`,
		`I output.go:<LINE>] "integer keys" %!s(int=1)="value" %!s(int=2)="value2" akey="avalue" akey2="(MISSING)"
`: `{"caller":"test/output.go:<LINE>","msg":"non-string key argument passed to logging, ignoring all later arguments","invalid key":1}
{"caller":"test/output.go:<LINE>","msg":"integer keys","v":0}
`,
		`I output.go:<LINE>] "struct keys" {name}="value" test="other value" key="val"
`: `{"caller":"test/output.go:<LINE>","msg":"non-string key argument passed to logging, ignoring all later arguments","invalid key":{}}
{"caller":"test/output.go:<LINE>","msg":"struct keys","v":0}
`,
		`I output.go:<LINE>] "map keys" map[test:%!s(bool=true)]="test"
`: `{"caller":"test/output.go:<LINE>","msg":"non-string key argument passed to logging, ignoring all later arguments","invalid key":{"test":true}}
{"caller":"test/output.go:<LINE>","msg":"map keys","v":0}
`,
	} {
		mapping[key] = value
	}
	return mapping
}
