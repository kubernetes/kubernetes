// Copyright 2016 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package optional

import "testing"

func TestConvertSuccess(t *testing.T) {
	if got, want := ToBool(false), false; got != want {
		t.Errorf("got %v, want %v", got, want)
	}
	if got, want := ToString(""), ""; got != want {
		t.Errorf("got %v, want %v", got, want)
	}
	if got, want := ToInt(0), 0; got != want {
		t.Errorf("got %v, want %v", got, want)
	}
	if got, want := ToUint(uint(0)), uint(0); got != want {
		t.Errorf("got %v, want %v", got, want)
	}
	if got, want := ToFloat64(0.0), 0.0; got != want {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestConvertFailure(t *testing.T) {
	for _, f := range []func(){
		func() { ToBool(nil) },
		func() { ToBool(3) },
		func() { ToString(nil) },
		func() { ToString(3) },
		func() { ToInt(nil) },
		func() { ToInt("") },
		func() { ToUint(nil) },
		func() { ToUint("") },
		func() { ToFloat64(nil) },
		func() { ToFloat64("") },
	} {
		if !panics(f) {
			t.Error("got no panic, want panic")
		}
	}
}

func panics(f func()) (b bool) {
	defer func() {
		if recover() != nil {
			b = true
		}
	}()
	f()
	return false
}
