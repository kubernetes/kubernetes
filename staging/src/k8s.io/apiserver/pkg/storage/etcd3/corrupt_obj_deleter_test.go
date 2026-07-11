/*
Copyright The Kubernetes Authors.

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

package etcd3

import (
	"errors"
	"testing"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
)

func TestAggregatedStorageErrorMessage(t *testing.T) {
	tests := []struct {
		name     string
		abortErr error
		want     string
	}{
		{
			name: "aggregation completed without an abort",
			want: "unable to transform or decode 2 objects: {\n\tfoo is corrupt\n\tbar is corrupt\n}",
		},
		{
			name:     "aggregation aborted after reaching the limit",
			abortErr: errTooMany,
			want:     "unable to transform or decode 2 objects: {\n\tfoo is corrupt\n\tbar is corrupt\n}, aborted: too many errors, the list is truncated",
		},
		{
			name:     "aggregation aborted due to an unexpected error",
			abortErr: errors.New("etcd is down"),
			want:     "unable to transform or decode 2 objects: {\n\tfoo is corrupt\n\tbar is corrupt\n}, aborted: etcd is down",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := &aggregatedStorageError{
				resourcePrefix: "list",
				errs:           utilerrors.NewAggregate([]error{errors.New("foo is corrupt"), errors.New("bar is corrupt")}),
				abortErr:       test.abortErr,
			}
			if want, got := test.want, err.Error(); want != got {
				t.Errorf("unexpected error message, want:\n%s\ngot:\n%s", want, got)
			}
		})
	}
}
