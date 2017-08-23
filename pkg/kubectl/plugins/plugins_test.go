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

import "testing"

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
			expectedErr: IncompletePluginError,
		},
		{
			plugin:      &Plugin{},
			expectedErr: IncompletePluginError,
		},
		{
			plugin: &Plugin{
				Description: Description{
					Name:      "test spaces",
					ShortDesc: "The test",
					Command:   "echo 1",
				},
			},
			expectedErr: InvalidPluginNameError,
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
			expectedErr: IncompleteFlagError,
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
			expectedErr: InvalidFlagNameError,
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
			expectedErr: InvalidFlagShorthandError,
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
			expectedErr: InvalidFlagShorthandError,
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
