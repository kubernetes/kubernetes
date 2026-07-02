/*
Copyright 2025 The Kubernetes Authors.

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

package storage

import (
	"fmt"
	"testing"
)

func TestIsPreconditionErrorForField(t *testing.T) {
	type args struct {
		err   error
		field string
	}
	tests := []struct {
		name                          string
		args                          args
		wantInvalidObjError           bool
		wantPreconditionErrorForField bool
	}{
		{
			name:                          "nil",
			args:                          args{err: nil, field: "otherfield"},
			wantInvalidObjError:           false,
			wantPreconditionErrorForField: false,
		},
		{
			name:                          "non-storage error",
			args:                          args{err: fmt.Errorf("test"), field: "otherfield"},
			wantInvalidObjError:           false,
			wantPreconditionErrorForField: false,
		},
		{
			name:                          "non-precondition storage error",
			args:                          args{err: &StorageError{}, field: "otherfield"},
			wantInvalidObjError:           false,
			wantPreconditionErrorForField: false,
		},
		{
			name:                          "non-precondition storage error",
			args:                          args{err: &StorageError{err: fmt.Errorf("test")}, field: "otherfield"},
			wantInvalidObjError:           false,
			wantPreconditionErrorForField: false,
		},
		{
			name:                          "invalid obj error",
			args:                          args{err: NewInvalidObjError("mykey", "myerr"), field: "otherfield"},
			wantInvalidObjError:           true,
			wantPreconditionErrorForField: false,
		},
		{
			name:                          "precondition storage error other field",
			args:                          args{err: NewPreconditionError("mykey", "myfield", "preconditionValue", "objectValue"), field: "otherfield"},
			wantInvalidObjError:           true,
			wantPreconditionErrorForField: false,
		},
		{
			name:                          "precondition storage error field",
			args:                          args{err: NewPreconditionError("mykey", "myfield", "preconditionValue", "objectValue"), field: "myfield"},
			wantInvalidObjError:           true,
			wantPreconditionErrorForField: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsInvalidObj(tt.args.err); got != tt.wantInvalidObjError {
				t.Errorf("IsInvalidObj() = %v, want %v", got, tt.wantInvalidObjError)
			}
			if got := IsPreconditionErrorForField(tt.args.err, tt.args.field); got != tt.wantPreconditionErrorForField {
				t.Errorf("IsPreconditionErrorForField() = %v, want %v", got, tt.wantPreconditionErrorForField)
			}
		})
	}
}
