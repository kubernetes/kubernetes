/*
Copyright 2021 The Kubernetes Authors.

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

package healthz

import (
	"fmt"
	"net/http"
	"testing"
)

type checkWithMessage struct {
	message string
}

func (c *checkWithMessage) Check(_ *http.Request) error {
	return fmt.Errorf("%s", c.message)
}

func TestNamedHealthChecker(t *testing.T) {
	named := NamedHealthChecker("foo", &checkWithMessage{message: "hello"})
	if named.Name() != "foo" {
		t.Errorf("expected: %v, got: %v", "foo", named.Name())
	}
	if err := named.Check(nil); err.Error() != "hello" {
		t.Errorf("expected: %v, got: %v", "hello", err.Error())
	}
}
