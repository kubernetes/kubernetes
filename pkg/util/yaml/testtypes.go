/*
Copyright 2017 The Kubernetes Authors.

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

/*
These types are for tesing purposes only. Because the gengo parser does not look at *_test.go files,
these types have to live in a real .go file, unforutnately.
*/

type MyConfig struct {
	// c1.1
	// c1.2
	ToBeEmbedded `json:",inline"`
	// c2.1
	// c2.2
	AStruct someStruct `json:"aStruct"`
	// c3.1
	// c3.2
	IgnoredField string `json:"-"`
	// c4.1
	// c4.2
	AStringSlice []string `json:"aStringSlice"`
	// c5.1
	// c5.2
	AStructSlice []someStruct `json:"aStructSlice"`
	// c6.1
	// c6.2
	AStructMap map[string]someStruct `json:"aStructMap"`
	// c7.1
	// c7.2
	EmptyString string `json:"emptyString"`
	// c8.1
	// c8.2
	PointerToEmptyString *string `json:"pointerToEmptyString"`
	// c9.1
	// c9.2
	PointerToString *string `json:"pointerToString"`
	// c10.1
	// c10.2
	PointerToNullStruct *someStruct `json:"pointerToNullStruct"`
	// c11.1
	// c11.2
	PointerToStruct *someStruct `json:"pointerToStruct"`
	// c20.1
	// c20.2
	AString string `json:"aString"`
	// c21.1
	// c21.2
	AnInt     int    `json:"anInt"`
	NoComment string `json:"noComment"`
}

type ToBeEmbedded struct {
	// c12.1
	// c12.2
	EmbeddedString string `json:"embeddedString"`
	// c13.1
	// c13.2
	EmbeddedStruct someStruct `json:"embeddedStruct"`
}

type someStruct struct {
	// c14.1
	// c14.2
	Nested someOtherStruct `json:"nested"`
	// c15.1
	// c15.2
	AString string `json:"aString"`
	// c16.1
	// c16.2
	AStringSlice []string `json:"aStringSlice"`
	// c17.1
	// c17.2
	AStructSlice []someOtherStruct `json:"aStructSlice"`
	// c18.1
	// c18.2
	AStructMap map[string]someOtherStruct `json:"aStructMap"`
}

type someOtherStruct struct {
	// c19.1
	// c19.2
	AnInt int `json:"anInt"`
}
