/*
Copyright 2024 The Kubernetes Authors.

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

package validators

import (
	"strings"
	"testing"

	"k8s.io/gengo/v2/codetags"
)

func TestTypeCheck(t *testing.T) {

	mkt := func(args []codetags.Arg, valueType codetags.ValueType) codetags.Tag {
		return codetags.Tag{
			Args:      args,
			ValueType: valueType,
		}
	}

	mkta := func(name string, value string, argType codetags.ArgType) codetags.Arg {
		return codetags.Arg{
			Name:  name,
			Value: value,
			Type:  argType,
		}
	}

	mkd := func(args []TagArgDoc, payloadsRequired bool, payloadsType codetags.ValueType) TagDoc {
		return TagDoc{
			Args:             args,
			PayloadsRequired: payloadsRequired,
			PayloadsType:     payloadsType,
		}
	}

	mkda := func(name string, argType codetags.ArgType, required bool) TagArgDoc {
		return TagArgDoc{
			Name:     name,
			Type:     argType,
			Required: required,
		}
	}
	tests := []struct {
		name    string
		tag     codetags.Tag
		doc     TagDoc
		wantErr string
	}{
		{
			name: "valid tag with no args",
			tag:  mkt(nil, codetags.ValueTypeNone),
			doc:  mkd(nil, false, codetags.ValueTypeNone),
		},

		// named args
		{
			name: "valid tag with required named args",
			tag: mkt([]codetags.Arg{
				mkta("arg1", "value1", codetags.ArgTypeString),
				mkta("arg2", "value2", codetags.ArgTypeString),
			}, codetags.ValueTypeNone),
			doc: mkd([]TagArgDoc{
				mkda("arg1", codetags.ArgTypeString, true),
				mkda("arg2", codetags.ArgTypeString, true),
			}, false, codetags.ValueTypeNone),
		},
		{
			name: "valid tag with optional named args",
			tag: mkt([]codetags.Arg{
				mkta("arg1", "value1", codetags.ArgTypeString),
				mkta("arg2", "value2", codetags.ArgTypeString),
			}, codetags.ValueTypeNone),
			doc: mkd([]TagArgDoc{
				mkda("arg1", codetags.ArgTypeString, false),
				mkda("arg2", codetags.ArgTypeString, false),
			}, false, codetags.ValueTypeNone),
		},
		{
			name: "valid tag without optional named args",
			tag:  mkt([]codetags.Arg{}, codetags.ValueTypeNone),
			doc: mkd([]TagArgDoc{
				mkda("arg1", codetags.ArgTypeString, false),
				mkda("arg2", codetags.ArgTypeString, false),
			}, false, codetags.ValueTypeNone),
		},
		{
			name: "missing required named argument",
			tag:  mkt(nil, codetags.ValueTypeNone),
			doc: mkd([]TagArgDoc{
				mkda("arg1", codetags.ArgTypeString, true),
			}, false, codetags.ValueTypeNone),
			wantErr: "missing named argument",
		},
		{
			name: "named argument with wrong type",
			tag: mkt([]codetags.Arg{
				mkta("arg1", "value1", codetags.ArgTypeString),
			}, codetags.ValueTypeNone),
			doc: mkd([]TagArgDoc{
				mkda("arg1", codetags.ArgTypeInt, true),
			}, false, codetags.ValueTypeNone),
			wantErr: "has wrong type: got string, want int",
		},
		{
			name: "unrecognized named argument",
			tag: mkt([]codetags.Arg{
				mkta("arg1", "value1", codetags.ArgTypeString),
				mkta("arg2", "value2", codetags.ArgTypeString),
			}, codetags.ValueTypeNone),
			doc: mkd([]TagArgDoc{
				mkda("arg1", codetags.ArgTypeString, true),
			}, false, codetags.ValueTypeNone),
			wantErr: "unrecognized named argument",
		},

		// positional arg
		{
			name: "valid tag with required positional arg",
			tag: mkt([]codetags.Arg{
				mkta("", "value1", codetags.ArgTypeString),
			}, codetags.ValueTypeNone),
			doc: mkd([]TagArgDoc{
				mkda("", codetags.ArgTypeString, true),
			}, false, codetags.ValueTypeNone),
		},
		{
			name: "valid tag with optional positional arg",
			tag: mkt([]codetags.Arg{
				mkta("", "value1", codetags.ArgTypeString),
			}, codetags.ValueTypeNone),
			doc: mkd([]TagArgDoc{
				mkda("", codetags.ArgTypeString, false),
			}, false, codetags.ValueTypeNone),
		},
		{
			name: "valid tag without optional positional arg",
			tag:  mkt([]codetags.Arg{}, codetags.ValueTypeNone),
			doc: mkd([]TagArgDoc{
				mkda("", codetags.ArgTypeString, false),
			}, false, codetags.ValueTypeNone),
		},
		{
			name: "missing required positional argument",
			tag:  mkt(nil, codetags.ValueTypeNone),
			doc: mkd([]TagArgDoc{
				mkda("", codetags.ArgTypeString, true),
			}, false, codetags.ValueTypeNone),
			wantErr: "missing required positional argument",
		},
		{
			name: "positional argument with wrong type",
			tag: mkt([]codetags.Arg{
				mkta("", "value1", codetags.ArgTypeString),
			}, codetags.ValueTypeNone),
			doc: mkd([]TagArgDoc{
				mkda("", codetags.ArgTypeInt, true),
			}, false, codetags.ValueTypeNone),
			wantErr: "has wrong type: got string, want int",
		},

		// values
		{
			name: "valid required tag value",
			tag:  mkt(nil, codetags.ValueTypeString),
			doc:  mkd(nil, true, codetags.ValueTypeString),
		},
		{
			name: "valid with optional tag value",
			tag:  mkt(nil, codetags.ValueTypeString),
			doc:  mkd(nil, false, codetags.ValueTypeString),
		},
		{
			name: "valid without optional tag value",
			tag:  mkt(nil, codetags.ValueTypeNone),
			doc:  mkd(nil, false, codetags.ValueTypeString),
		},
		{
			name:    "missing required tag value",
			tag:     mkt(nil, codetags.ValueTypeNone),
			doc:     mkd(nil, true, codetags.ValueTypeString),
			wantErr: "missing required tag value",
		},
		{
			name:    "tag value with wrong type",
			tag:     mkt(nil, codetags.ValueTypeString),
			doc:     mkd(nil, false, codetags.ValueTypeInt),
			wantErr: "tag value has wrong type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := typeCheck(tt.tag, tt.doc)
			if (len(tt.wantErr) == 0) != (err == nil) {
				t.Errorf("typeCheck() error = %v, wantErr = %v", err, tt.wantErr)
			}
			if err != nil && tt.wantErr != "" {
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Errorf("typeCheck() error = %v, wantErr = %v", err, tt.wantErr)
				}
			}
		})
	}
}
