/*
Copyright (c) 2014 VMware, Inc. All Rights Reserved.

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

package progress

import "testing"

func TestPrefix(t *testing.T) {
	var r Report

	ch := make(chan Report, 1)
	s := Prefix(dummySinker{ch}, "prefix").Sink()

	// No detail
	s <- dummyReport{d: ""}
	r = <-ch
	if r.Detail() != "prefix" {
		t.Errorf("Expected detail to be prefixed")
	}

	// With detail
	s <- dummyReport{d: "something"}
	r = <-ch
	if r.Detail() != "prefix: something" {
		t.Errorf("Expected detail to be prefixed")
	}
}
