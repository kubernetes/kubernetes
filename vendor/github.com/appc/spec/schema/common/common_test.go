// Copyright 2015 The appc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package common

import (
	"testing"
)

func TestMakeQueryString(t *testing.T) {
	tests := []struct {
		input  string
		output string
		werr   bool
	}{
		{
			input:  "",
			output: "",
			werr:   true,
		},
		{
			input:  "version,label=v1+v2",
			output: "",
			werr:   true,
		},
		{
			input:  "foo=bar",
			output: "foo=bar",
			werr:   false,
		},
		{
			input:  "foo=bar,boo=baz",
			output: "foo=bar&boo=baz",
			werr:   false,
		},
		{
			input:  "foo=Ümlaut",
			output: "foo=%C3%9Cmlaut",
			werr:   false,
		},
		{
			input:  "version=1.0.0,label=v1+v2",
			output: "version=1.0.0&label=v1%2Bv2",
			werr:   false,
		},
		{
			input:  "name=db,source=/tmp$1",
			output: "name=db&source=%2Ftmp%241",
			werr:   false,
		},
		{
			input:  "greeting-fr=Ça va?,greeting-es=¿Cómo estás?",
			output: "greeting-fr=%C3%87a+va%3F&greeting-es=%C2%BFC%C3%B3mo+est%C3%A1s%3F",
			werr:   false,
		},
	}

	for i, tt := range tests {
		o, err := MakeQueryString(tt.input)
		gerr := (err != nil)
		if o != tt.output {
			t.Errorf("#%d: expected `%v` got `%v`", i, tt.output, o)
		}
		if gerr != tt.werr {
			t.Errorf("#%d: gerr=%t, want %t (err=%v)", i, gerr, tt.werr, err)
		}
	}
}
