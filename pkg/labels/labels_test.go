/*
Copyright 2014 Google Inc. All rights reserved.

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

package labels

import (
	"testing"
)

func matches(t *testing.T, ls LabelSet, want string) {
	if ls.String() != want {
		t.Errorf("Expected '%s', but got '%s'", want, ls.String())
	}
}

func TestLabelSetString(t *testing.T) {
	matches(t, LabelSet{"x": "y"}, "x=y")
	matches(t, LabelSet{"foo": "bar"}, "foo=bar")
	matches(t, LabelSet{"foo": "bar", "baz": "qup"}, "foo=bar,baz=qup")

	// TODO: Make our label representation robust enough to handel labels
	// with ",=!" characters in their names.
}

func TestLabelGet(t *testing.T) {
	ls := LabelSet{"x": "y"}
	if ls.Get("x") != "y" {
		t.Errorf("LabelSet.Get is broken")
	}
}
