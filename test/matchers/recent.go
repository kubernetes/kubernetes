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

package matchers

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api/unversioned"

	"github.com/onsi/gomega/format"
)

type RecentMatcher struct {
	Duration time.Duration
}

func (m *RecentMatcher) Match(actual interface{}) (success bool, err error) {
	if t, ok := actual.(unversioned.Time); ok {
		return m.Match(t.Time)
	}
	t, ok := actual.(time.Time)
	if !ok {
		return false, fmt.Errorf("Expected a time.Time. Got:\n%s", format.Object(actual, 1))
	}

	now := time.Now()
	if now.Sub(t) > m.Duration {
		return false, fmt.Errorf("%v is too old! now: %v", t, now)
	} else if t.After(now) {
		return false, fmt.Errorf("%v is in the future! now: %v", t, now)
	}

	return true, nil
}

func (m *RecentMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "to be in the last", m.Duration)
}

func (m *RecentMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to be in the last", m.Duration)
}
