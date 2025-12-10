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
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/types"
	"k8s.io/klog/v2"
)

var mockType = &types.Type{
	CommentLines: []string{
		"RandomType defines a random structure in Kubernetes",
		"It should be used just when you need something different than 42",
	},
	SecondClosestCommentLines: []string{},
}

func TestArgsFromType(t *testing.T) {
	type testcase struct {
		name          string
		t             *types.Type
		expected      generator.Args
		expectedError string
	}

	tests := []testcase{
		{
			name: "no comments",
			t: &types.Type{
				Name: types.Name{
					Name:    "Simple",
					Package: "k8s.io/apis/core/v1",
				},
			},
			expectedError: `missing`,
		},
		{
			name: "GA type",
			t: &types.Type{
				Name: types.Name{
					Name:    "Simple",
					Package: "k8s.io/apis/core/v1",
				},
				CommentLines: []string{
					"+k8s:prerelease-lifecycle-gen:introduced=1.5",
				},
			},
			expected: generator.Args{
				"introducedMajor": 1,
				"introducedMinor": 5,
			},
		},
		{
			name: "GA type v2",
			t: &types.Type{
				Name: types.Name{
					Name:    "Simple",
					Package: "k8s.io/apis/core/v2",
				},
				CommentLines: []string{
					"+k8s:prerelease-lifecycle-gen:introduced=1.5",
				},
			},
			expected: generator.Args{
				"introducedMajor": 1,
				"introducedMinor": 5,
			},
		},
		{
			name: "GA type - explicit deprecated",
			t: &types.Type{
				Name: types.Name{
					Name:    "Simple",
					Package: "k8s.io/apis/core/v1",
				},
				CommentLines: []string{
					"+k8s:prerelease-lifecycle-gen:introduced=1.5",
					"+k8s:prerelease-lifecycle-gen:deprecated=1.7",
				},
			},
			expected: generator.Args{
				"introducedMajor": 1,
				"introducedMinor": 5,
				"deprecatedMajor": 1,
				"deprecatedMinor": 7,
			},
		},
		{
			name: "GA type - explicit removed",
			t: &types.Type{
				Name: types.Name{
					Name:    "Simple",
					Package: "k8s.io/apis/core/v1",
				},
				CommentLines: []string{
					"+k8s:prerelease-lifecycle-gen:introduced=1.5",
					"+k8s:prerelease-lifecycle-gen:removed=1.9",
				},
			},
			expected: generator.Args{
				"introducedMajor": 1,
				"introducedMinor": 5,
				"removedMajor":    1,
				"removedMinor":    9,
			},
		},
		{
			name: "GA type - explicit",
			t: &types.Type{
				Name: types.Name{
					Name:    "Simple",
					Package: "k8s.io/apis/core/v1",
				},
				CommentLines: []string{
					"+k8s:prerelease-lifecycle-gen:introduced=1.5",
					"+k8s:prerelease-lifecycle-gen:deprecated=1.7",
					"+k8s:prerelease-lifecycle-gen:removed=1.9",
				},
			},
			expected: generator.Args{
				"introducedMajor": 1,
				"introducedMinor": 5,
				"deprecatedMajor": 1,
				"deprecatedMinor": 7,
				"removedMajor":    1,
				"removedMinor":    9,
			},
		},
		{
			name: "beta type - defaulted",
			t: &types.Type{
				Name: types.Name{
					Name:    "Simple",
					Package: "k8s.io/apis/core/v1beta1",
				},
				CommentLines: []string{
					"+k8s:prerelease-lifecycle-gen:introduced=1.5",
				},
			},
			expected: generator.Args{
				"introducedMajor": 1,
				"introducedMinor": 5,
				"deprecatedMajor": 1,
				"deprecatedMinor": 8,
				"removedMajor":    1,
				"removedMinor":    11,
			},
		},
		{
			name: "beta type - explicit",
			t: &types.Type{
				Name: types.Name{
					Name:    "Simple",
					Package: "k8s.io/apis/core/v1beta1",
				},
				CommentLines: []string{
					"+k8s:prerelease-lifecycle-gen:introduced=1.5",
					"+k8s:prerelease-lifecycle-gen:deprecated=1.7",
					"+k8s:prerelease-lifecycle-gen:removed=1.9",
				},
			},
			expected: generator.Args{
				"introducedMajor": 1,
				"introducedMinor": 5,
				"deprecatedMajor": 1,
				"deprecatedMinor": 7,
				"removedMajor":    1,
				"removedMinor":    9,
			},
		},
		{
			name: "beta type - explicit deprecated only",
			t: &types.Type{
				Name: types.Name{
					Name:    "Simple",
					Package: "k8s.io/apis/core/v1beta1",
				},
				CommentLines: []string{
					"+k8s:prerelease-lifecycle-gen:introduced=1.5",
					"+k8s:prerelease-lifecycle-gen:deprecated=1.7",
				},
			},
			expected: generator.Args{
				"introducedMajor": 1,
				"introducedMinor": 5,
				"deprecatedMajor": 1,
				"deprecatedMinor": 7,
				"removedMajor":    1,
				"removedMinor":    10,
			},
		},
		{
			name: "beta type - explicit removed only",
			t: &types.Type{
				Name: types.Name{
					Name:    "Simple",
					Package: "k8s.io/apis/core/v1beta1",
				},
				CommentLines: []string{
					"+k8s:prerelease-lifecycle-gen:introduced=1.5",
					"+k8s:prerelease-lifecycle-gen:removed=1.9",
				},
			},
			expected: generator.Args{
				"introducedMajor": 1,
				"introducedMinor": 5,
				"deprecatedMajor": 1,
				"deprecatedMinor": 8,
				"removedMajor":    1,
				"removedMinor":    9,
			},
		},
		{
			name: "alpha type - defaulted",
			t: &types.Type{
				Name: types.Name{
					Name:    "Simple",
					Package: "k8s.io/apis/core/v1alpha1",
				},
				CommentLines: []string{
					"+k8s:prerelease-lifecycle-gen:introduced=1.5",
				},
			},
			expected: generator.Args{
				"introducedMajor": 1,
				"introducedMinor": 5,
				"deprecatedMajor": 1,
				"deprecatedMinor": 8,
				"removedMajor":    1,
				"removedMinor":    11,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.expected != nil {
				test.expected["type"] = test.t
			}
			gen := genPreleaseLifecycle{}
			args, err := gen.argsFromType(nil, test.t)
			if test.expectedError != "" {
				if err == nil {
					t.Errorf("expected error, got none")
				} else if !strings.Contains(err.Error(), test.expectedError) {
					t.Errorf("expected error %q, got %q", test.expectedError, err.Error())
				}
				return
			}
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(test.expected, args); diff != "" {
				t.Error(diff)
			}
		})
	}
}

func Test_extractKubeVersionTag(t *testing.T) {
	oldKlogOsExit := klog.OsExit
	defer func() {
		klog.OsExit = oldKlogOsExit
	}()
	klog.OsExit = customExit

	tests := []struct {
		name        string
		tagName     string
		tagComments []string
		wantValue   *tagValue
		wantMajor   int
		wantMinor   int
		wantErr     bool
		wantFatal   bool
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
		{
			name:    "multiple declarations of same tag should return an error",
			tagName: "someVersionTag:version",
			tagComments: []string{
				"+someVersionTag:version=1.5",
				"+someVersionTag:version=v1.7",
			},
			wantValue: nil,
			wantFatal: true,
		},
		{
			name:    "multiple values on same tag should return an error",
			tagName: "someVersionTag:version",
			tagComments: []string{
				"+someVersionTag:version=1.5,something",
			},
			wantValue: nil,
			wantFatal: true,
		},
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
			gotTag, gotMajor, gotMinor, err, fatalErr := safeExtractKubeVersionTag(tt.tagName, mockType)
			if (fatalErr != nil) != tt.wantFatal {
				t.Errorf("extractKubeVersionTag() fatalErr = %v, wantFatal %v", fatalErr, tt.wantFatal)
				return
			}
			if tt.wantFatal {
				return
			}
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

func customExit(exitCode int) {
	panic(strconv.Itoa(exitCode))
}

func safeExtractKubeVersionTag(tagName string, t *types.Type) (value *tagValue, major int, minor int, err error, localErr error) {
	defer func() {
		if e := recover(); e != nil {
			localErr = fmt.Errorf("extractKubeVersionTag returned error: %v", e)
		}
	}()
	value, major, minor, err = extractKubeVersionTag(tagName, t)
	return
}

func safeExtractTag(t *testing.T, tagName string, comments []string) (value *tagValue, err error) {
	defer func() {
		if e := recover(); e != nil {
			err = fmt.Errorf("extractTag returned error: %v", e)
		}
	}()
	value = extractTag(tagName, comments)
	return
}

func Test_extractTag(t *testing.T) {
	oldKlogOsExit := klog.OsExit
	defer func() {
		klog.OsExit = oldKlogOsExit
	}()
	klog.OsExit = customExit

	comments := []string{
		"+variable=7",
		"+anotherVariable=8",
		"+yetAnotherVariable=9",
		"variableWithoutMarker=10",
		"+variableWithoutValue",
		"+variable=11",
		"+multi-valuedVariable=12,13,14",
		"+strangeVariable=15,=16",
	}

	tests := []struct {
		name         string
		tagComments  []string
		variableName string
		wantError    bool
		wantValue    *tagValue
	}{
		{
			name:         "variable with explicit value",
			tagComments:  comments,
			variableName: "anotherVariable",
			wantValue:    &tagValue{value: "8"},
		},
		{
			name:         "variable without explicit value",
			tagComments:  comments,
			variableName: "variableWithoutValue",
			wantValue:    &tagValue{value: ""},
		},
		{
			name:         "variable not present in comments",
			tagComments:  comments,
			variableName: "variableOutOfNowhere",
			wantValue:    nil,
		},
		{
			name:         "variable without marker test",
			tagComments:  comments,
			variableName: "variableWithoutMarker",
			wantValue:    nil,
		},
		{
			name:         "abort duplicated variable",
			tagComments:  comments,
			variableName: "variable",
			wantError:    true,
			wantValue:    nil,
		},
		{
			name:         "abort variable with multiple values",
			tagComments:  comments,
			variableName: "multi-valuedVariable",
			wantError:    true,
			wantValue:    nil,
		},
		{
			name:         "this test documents strange behaviour", // TODO: Is this behaviour intended?
			tagComments:  comments,
			variableName: "strangeVariable",
			wantValue:    &tagValue{value: "15"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotTag, err := safeExtractTag(t, tt.variableName, tt.tagComments)
			if (err != nil) != tt.wantError {
				t.Errorf("extractTag() err = %v, wantError = %v.", gotTag, tt.wantError)
				return
			}
			if tt.wantError {
				return
			}
			if !reflect.DeepEqual(gotTag, tt.wantValue) {
				t.Errorf("extractTag() got = %v, want %v", gotTag, tt.wantValue)
			}
		})
	}
}

func Test_extractEnabledTypeTag(t *testing.T) {
	someComments := []string{
		"+variable=7",
		"+k8s:prerelease-lifecycle-gen=8",
	}
	moreComments := []string{
		"+yetAnotherVariable=9",
		"variableWithoutMarker=10",
		"+variableWithoutValue",
		"+variable=11",
		"+multi-valuedVariable=12,13,14",
	}

	tests := []struct {
		name      string
		mockType  *types.Type
		wantValue *tagValue
	}{
		{
			name:      "desired info in main comments",
			mockType:  &types.Type{CommentLines: someComments, SecondClosestCommentLines: moreComments},
			wantValue: &tagValue{value: "8"},
		},
		{
			name:      "secondary comments empty",
			mockType:  &types.Type{CommentLines: someComments},
			wantValue: &tagValue{value: "8"},
		},
		{
			name:      "main comments empty",
			mockType:  &types.Type{SecondClosestCommentLines: someComments},
			wantValue: &tagValue{value: "8"},
		},
		{
			name:      "lack of desired info, empty secondary comments",
			mockType:  &types.Type{CommentLines: moreComments},
			wantValue: nil,
		},
		{
			name:      "lack of desired info, empty main comments",
			mockType:  &types.Type{SecondClosestCommentLines: moreComments},
			wantValue: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotTag := extractEnabledTypeTag(tt.mockType)
			if !reflect.DeepEqual(gotTag, tt.wantValue) {
				t.Errorf("extractEnabledTypeTag() got = %v, want %v", gotTag, tt.wantValue)
			}
		})
	}
}

func Test_extractReplacementTag(t *testing.T) {
	replacementTag := "+k8s:prerelease-lifecycle-gen:replacement"
	tests := []struct {
		name               string
		mainComments       []string
		secondaryComments  []string
		wantGroup          string
		wantVersion        string
		wantKind           string
		wantHasReplacement bool
		wantErr            bool
	}{
		{
			name:               "no replacement tag",
			mainComments:       []string{"randomText=7"},
			secondaryComments:  []string{"importantFlag=8.8.8.8"},
			wantGroup:          "",
			wantVersion:        "",
			wantKind:           "",
			wantHasReplacement: false,
			wantErr:            false,
		},
		{
			name:               "replacement tag correct",
			mainComments:       []string{fmt.Sprintf("%v=my_group,v1,KindOf", replacementTag)},
			secondaryComments:  []string{},
			wantGroup:          "my_group",
			wantVersion:        "v1",
			wantKind:           "KindOf",
			wantHasReplacement: true,
			wantErr:            false,
		},
		{
			name:               "correct replacement tag in secondary comments",
			mainComments:       []string{},
			secondaryComments:  []string{fmt.Sprintf("%v=my_group,v1,KindOf", replacementTag)},
			wantGroup:          "my_group",
			wantVersion:        "v1",
			wantKind:           "KindOf",
			wantHasReplacement: true,
			wantErr:            false,
		},
		{
			name:               "4 values instead of 3",
			mainComments:       []string{fmt.Sprintf("%v=my_group,v1,KindOf,subKind", replacementTag)},
			secondaryComments:  []string{},
			wantGroup:          "",
			wantVersion:        "",
			wantKind:           "",
			wantHasReplacement: false,
			wantErr:            true,
		},
		{
			name:               "4 values instead of 3 in secondary comments",
			mainComments:       []string{},
			secondaryComments:  []string{fmt.Sprintf("%v=my_group,v1,KindOf,subKind", replacementTag)},
			wantGroup:          "",
			wantVersion:        "",
			wantKind:           "",
			wantHasReplacement: false,
			wantErr:            true,
		},
		{
			name:               "2 values instead of 3",
			mainComments:       []string{fmt.Sprintf("%v=my_group,v1", replacementTag)},
			secondaryComments:  []string{},
			wantGroup:          "",
			wantVersion:        "",
			wantKind:           "",
			wantHasReplacement: false,
			wantErr:            true,
		},
		{
			name:               "group name not all upper",
			mainComments:       []string{fmt.Sprintf("%v=myGroup,v1,KindOf", replacementTag)},
			secondaryComments:  []string{},
			wantGroup:          "",
			wantVersion:        "",
			wantKind:           "",
			wantHasReplacement: false,
			wantErr:            true,
		},
		{
			name:               "version name does not start with v",
			mainComments:       []string{fmt.Sprintf("%v=my_group,bestVersion,KindOf", replacementTag)},
			secondaryComments:  []string{},
			wantGroup:          "",
			wantVersion:        "",
			wantKind:           "",
			wantHasReplacement: false,
			wantErr:            true,
		},
		{
			name:               "kind name does not start with capital",
			mainComments:       []string{fmt.Sprintf("%v=my_group,v1,kindOf", replacementTag)},
			secondaryComments:  []string{},
			wantGroup:          "",
			wantVersion:        "",
			wantKind:           "",
			wantHasReplacement: false,
			wantErr:            true,
		},
		{
			name:               "empty group name", // TODO: is it a valid input or a bug?
			mainComments:       []string{fmt.Sprintf("%v=,v1,KindOf", replacementTag)},
			secondaryComments:  []string{},
			wantGroup:          "",
			wantVersion:        "v1",
			wantKind:           "KindOf",
			wantHasReplacement: true,
			wantErr:            false,
		},
		{
			name:               "empty version",
			mainComments:       []string{fmt.Sprintf("%v=my_group,,KindOf", replacementTag)},
			secondaryComments:  []string{},
			wantGroup:          "",
			wantVersion:        "",
			wantKind:           "",
			wantHasReplacement: false,
			wantErr:            true,
		},
		{
			name:               "empty kind",
			mainComments:       []string{fmt.Sprintf("%v=my_group,v1,", replacementTag)},
			secondaryComments:  []string{},
			wantGroup:          "",
			wantVersion:        "",
			wantKind:           "",
			wantHasReplacement: false,
			wantErr:            true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			replacementGroup, replacementVersion, replacementKind, hasReplacement, err := extractReplacementTag(&types.Type{
				CommentLines:              tt.mainComments,
				SecondClosestCommentLines: tt.secondaryComments,
			})
			if replacementGroup != tt.wantGroup {
				t.Errorf("extractReplacementTag() group got = %v, want %v", replacementGroup, tt.wantGroup)
			}
			if replacementVersion != tt.wantVersion {
				t.Errorf("extractReplacementTag() version got = %v, want %v", replacementVersion, tt.wantVersion)
			}
			if replacementKind != tt.wantKind {
				t.Errorf("extractReplacementTag() kind got = %v, want %v", replacementKind, tt.wantKind)
			}
			if hasReplacement != tt.wantHasReplacement {
				t.Errorf("extractReplacementTag() hasReplacement got = %v, want %v", hasReplacement, tt.wantHasReplacement)
			}
			if (err != nil) != tt.wantErr {
				t.Errorf("extractReplacementTag() err got = %v, want %v", err, tt.wantErr)
			}
		})
	}
}

func Test_isAPIType(t *testing.T) {
	tests := []struct {
		name string
		t    *types.Type
		want bool
	}{
		{
			name: "private name is not apitype",
			want: false,
			t: &types.Type{
				Name: types.Name{
					Name: "notpublic",
				},
			},
		},
		{
			name: "non struct is not apitype",
			want: false,
			t: &types.Type{
				Name: types.Name{
					Name: "Public",
				},
				Kind: types.Slice,
			},
		},
		{
			name: "contains member type",
			want: true,
			t: &types.Type{
				Name: types.Name{
					Name: "Public",
				},
				Kind: types.Struct,
				Members: []types.Member{
					{
						Embedded: true,
						Name:     "TypeMeta",
					},
				},
			},
		},

		{
			name: "contains no type",
			want: false,
			t: &types.Type{
				Name: types.Name{
					Name: "Public",
				},
				Kind: types.Struct,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isAPIType(tt.t); got != tt.want {
				t.Errorf("isAPIType() = %v, want %v", got, tt.want)
			}
		})
	}
}
