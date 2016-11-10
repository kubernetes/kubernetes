/*
Copyright 2016 The Kubernetes Authors.

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

package retry

import (
	"fmt"
	"testing"

	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/util/wait"
)

func TestRetryOnConflict(t *testing.T) {
	opts := wait.Backoff{Factor: 1.0, Steps: 3}
	conflictErr := errors.NewConflict(unversioned.GroupResource{Resource: "test"}, "other", nil)

	// never returns
	err := RetryOnConflict(opts, func() error {
		return conflictErr
	})
	if err != conflictErr {
		t.Errorf("unexpected error: %v", err)
	}

	// returns immediately
	i := 0
	err = RetryOnConflict(opts, func() error {
		i++
		return nil
	})
	if err != nil || i != 1 {
		t.Errorf("unexpected error: %v", err)
	}

	// returns immediately on error
	testErr := fmt.Errorf("some other error")
	err = RetryOnConflict(opts, func() error {
		return testErr
	})
	if err != testErr {
		t.Errorf("unexpected error: %v", err)
	}

	// keeps retrying
	i = 0
	err = RetryOnConflict(opts, func() error {
		if i < 2 {
			i++
			return errors.NewConflict(unversioned.GroupResource{Resource: "test"}, "other", nil)
		}
		return nil
	})
	if err != nil || i != 2 {
		t.Errorf("unexpected error: %v", err)
	}
}
