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

package plugins

import (
	"strconv"
	"testing"
)

func TestPlugin(t *testing.T) {
	tests := []struct {
		plugin      *Plugin
		expectedErr error
	}{
		{
			plugin: &Plugin{
				Description: Description{
					Name:      "test",
					ShortDesc: "The test",
					Command:   "echo 1",
				},
			},
		},
		{
			plugin: &Plugin{
				Description: Description{
					Name:      "test",
					ShortDesc: "The test",
				},
			},
			expectedErr: ErrIncompletePlugin,
		},
		{
			plugin:      &Plugin{},
			expectedErr: ErrIncompletePlugin,
		},
		{
			plugin: &Plugin{
				Description: Description{
					Name:      "test spaces",
					ShortDesc: "The test",
					Command:   "echo 1",
				},
			},
			expectedErr: ErrInvalidPluginName,
		},
		{
			plugin: &Plugin{
				Description: Description{
					Name:      "test",
					ShortDesc: "The test",
					Command:   "echo 1",
					Flags: []Flag{
						{
							Name: "aflag",
						},
					},
				},
			},
			expectedErr: ErrIncompleteFlag,
		},
		{
			plugin: &Plugin{
				Description: Description{
					Name:      "test",
					ShortDesc: "The test",
					Command:   "echo 1",
					Flags: []Flag{
						{
							Name: "a flag",
							Desc: "Invalid flag",
						},
					},
				},
			},
			expectedErr: ErrInvalidFlagName,
		},
		{
			plugin: &Plugin{
				Description: Description{
					Name:      "test",
					ShortDesc: "The test",
					Command:   "echo 1",
					Flags: []Flag{
						{
							Name:      "aflag",
							Desc:      "Invalid shorthand",
							Shorthand: "aa",
						},
					},
				},
			},
			expectedErr: ErrInvalidFlagShorthand,
		},
		{
			plugin: &Plugin{
				Description: Description{
					Name:      "test",
					ShortDesc: "The test",
					Command:   "echo 1",
					Flags: []Flag{
						{
							Name:      "aflag",
							Desc:      "Invalid shorthand",
							Shorthand: "2",
						},
					},
				},
			},
			expectedErr: ErrInvalidFlagShorthand,
		},
		{
			plugin: &Plugin{
				Description: Description{
					Name:      "test",
					ShortDesc: "The test",
					Command:   "echo 1",
					Flags: []Flag{
						{
							Name: "aflag",
							Desc: "Invalid flag type",
							Type: "stringArray",
						},
					},
				},
			},
			expectedErr: ErrInvalidFlagType,
		},
		{
			plugin: &Plugin{
				Description: Description{
					Name:      "test",
					ShortDesc: "The test",
					Command:   "echo 1",
					Flags: []Flag{
						{
							Name:     "aflag",
							Desc:     "boolean flag with a default value",
							Type:     BoolFlagType,
							DefValue: "true",
						},
					},
				},
			},
		},
		{
			plugin: &Plugin{
				Description: Description{
					Name:      "test",
					ShortDesc: "The test",
					Command:   "echo 1",
					Flags: []Flag{
						{
							Name:     "aflag",
							Desc:     "Invalid flag default value",
							Type:     BoolFlagType,
							DefValue: "yes",
						},
					},
				},
			},
			expectedErr: ErrInvalidFlagDefaultValue,
		},
		{
			plugin: &Plugin{
				Description: Description{
					Name:      "test",
					ShortDesc: "The test",
					Command:   "echo 1",
					Flags: []Flag{
						{
							Name:      "aflag",
							Desc:      "A flag",
							Shorthand: "a",
						},
						{
							Name: "boolflag",
							Desc: "A bool flag",
							Type: BoolFlagType,
						},
					},
					Tree: Plugins{
						&Plugin{
							Description: Description{
								Name:      "child",
								ShortDesc: "The child",
								LongDesc:  "The child long desc",
								Example:   "You can use it like this but you're not supposed to",
								Command:   "echo 1",
								Flags: []Flag{
									{
										Name: "childflag",
										Desc: "A child flag",
									},
									{
										Name:      "childshorthand",
										Desc:      "A child shorthand flag",
										Shorthand: "s",
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, test := range tests {
		err := test.plugin.Validate()
		if err != test.expectedErr {
			t.Errorf("%s: expected error %v, got %v", test.plugin.Name, test.expectedErr, err)
		}
	}
}

func TestPluginFlagTypeDefaultsToString(t *testing.T) {
	f := Flag{
		Name: "aflag",
		Desc: "a flag with an unspecified type",
		Type: "",
	}

	err := f.Validate()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if f.Type != StringFlagType {
		t.Fatalf("expected flag type to default to %v, got %v", StringFlagType, f.Type)
	}
}

func TestPluginBoolFlagDefaultsToFalse(t *testing.T) {
	f := Flag{
		Name:     "aflag",
		Desc:     "a bool flag with an unspecified default value",
		Type:     BoolFlagType,
		DefValue: "",
	}

	err := f.Validate()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	defVal, err := strconv.ParseBool(f.DefValue)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if defVal != false {
		t.Fatalf("expected bool flag to default to %v, got %v", false, defVal)
	}
}
