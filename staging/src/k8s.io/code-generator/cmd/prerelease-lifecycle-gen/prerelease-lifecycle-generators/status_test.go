/*
Copyright 2023 The Kubernetes Authors.

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

package prereleaselifecyclegenerators

import (
	"reflect"
	"testing"

	"k8s.io/gengo/types"
)

var mockType = &types.Type{
	CommentLines: []string{
		"RandomType defines a random structure in Kubernetes",
		"It should be used just when you need something different than 42",
	},
	SecondClosestCommentLines: []string{},
}

func Test_extractKubeVersionTag(t *testing.T) {
	tests := []struct {
		name        string
		tagName     string
		tagComments []string
		wantValue   *tagValue
		wantMajor   int
		wantMinor   int
		wantErr     bool
	}{
		{
			name:    "not found tag should generate an error",
			tagName: "someVersionTag:version",
			tagComments: []string{
				"+someOtherTag:version=1.5",
			},
			wantValue: nil,
			wantErr:   true,
		},
		{
			name:    "found tag should return correctly",
			tagName: "someVersionTag:version",
			tagComments: []string{
				"+someVersionTag:version=1.5",
			},
			wantValue: &tagValue{
				value: "1.5",
			},
			wantMajor: 1,
			wantMinor: 5,
			wantErr:   false,
		},
		/*{
			name:    "multiple declarations of same tag should return an error",
			tagName: "someVersionTag:version",
			tagComments: []string{
				"+someVersionTag:version=1.5",
				"+someVersionTag:version=v1.7",
			},
			wantValue: nil,
			wantErr:   true, // TODO: Today it is klog.Fatal, check how to capture it
		},
		{
			name:    "multiple values on same tag should return an error",
			tagName: "someVersionTag:version",
			tagComments: []string{
				"+someVersionTag:version=1.5,something",
			},
			wantValue: nil,
			wantErr:   true, // TODO: Today it is klog.Fatal, check how to capture it
		},*/
		{
			name:    "wrong tag major value should return an error",
			tagName: "someVersionTag:version",
			tagComments: []string{
				"+someVersionTag:version=.5",
			},
			wantErr: true,
		},
		{
			name:    "wrong tag minor value should return an error",
			tagName: "someVersionTag:version",
			tagComments: []string{
				"+someVersionTag:version=1.",
			},
			wantErr: true,
		},
		{
			name:    "wrong tag format should return an error",
			tagName: "someVersionTag:version",
			tagComments: []string{
				"+someVersionTag:version=1.5.7",
			},
			wantErr: true,
		},
		{
			name:    "wrong tag major int value should return an error",
			tagName: "someVersionTag:version",
			tagComments: []string{
				"+someVersionTag:version=blah.5",
			},
			wantErr: true,
		},
		{
			name:    "wrong tag minor int value should return an error",
			tagName: "someVersionTag:version",
			tagComments: []string{
				"+someVersionTag:version=1.blah",
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockType.SecondClosestCommentLines = tt.tagComments
			gotTag, gotMajor, gotMinor, err := extractKubeVersionTag(tt.tagName, mockType)
			if (err != nil) != tt.wantErr {
				t.Errorf("extractKubeVersionTag() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}
			if !reflect.DeepEqual(gotTag, tt.wantValue) {
				t.Errorf("extractKubeVersionTag() got = %v, want %v", gotTag, tt.wantValue)
			}
			if gotMajor != tt.wantMajor {
				t.Errorf("extractKubeVersionTag() got1 = %v, want %v", gotMajor, tt.wantMajor)
			}
			if gotMinor != tt.wantMinor {
				t.Errorf("extractKubeVersionTag() got2 = %v, want %v", gotMinor, tt.wantMinor)
			}
		})
	}
}
