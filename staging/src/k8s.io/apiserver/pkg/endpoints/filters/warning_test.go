/*
Copyright 2020 The Kubernetes Authors.

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

package filters

import (
	"net/http/httptest"
	"reflect"
	"testing"
)

func Test_recorder_AddWarning(t *testing.T) {
	type args struct {
		agent string
		text  string
	}
	tests := []struct {
		name   string
		args   []args
		expect []string
	}{
		{
			name:   "empty",
			args:   []args{},
			expect: nil,
		},
		{
			name:   "empty values",
			args:   []args{{agent: "", text: ""}},
			expect: nil,
		},
		{
			name:   "empty agent",
			args:   []args{{agent: "", text: "mytext"}},
			expect: []string{`299 - "mytext"`},
		},
		{
			name:   "simple",
			args:   []args{{agent: "myagent", text: "mytext"}},
			expect: []string{`299 myagent "mytext"`},
		},
		{
			name: "duplicate text",
			args: []args{
				{agent: "myagent", text: "mytext"},
				{agent: "myagent2", text: "mytext"},
			},
			expect: []string{`299 myagent "mytext"`},
		},
		{
			name: "multiple",
			args: []args{
				{agent: "", text: "mytext1"},
				{agent: "", text: "mytext2"},
				{agent: "", text: "mytext3"},
			},
			expect: []string{
				`299 - "mytext1"`,
				`299 - "mytext2"`,
				`299 - "mytext3"`,
			},
		},

		{
			name:   "invalid agent",
			args:   []args{{agent: "my agent", text: "mytext"}},
			expect: nil,
		},
		{
			name:   "invalid text",
			args:   []args{{agent: "myagent", text: "my\ntext"}},
			expect: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			responseRecorder := httptest.NewRecorder()
			warningRecorder := &recorder{writer: responseRecorder}
			for _, arg := range tt.args {
				warningRecorder.AddWarning(arg.agent, arg.text)
			}
			if !reflect.DeepEqual(tt.expect, responseRecorder.Header()["Warning"]) {
				t.Errorf("expected\n%#v\ngot\n%#v", tt.expect, responseRecorder.Header()["Warning"])
			}
		})
	}
}
