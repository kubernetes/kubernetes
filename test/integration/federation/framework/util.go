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

package framework

import (
	"testing"
	"time"
)

const (
	DefaultWaitInterval = 50 * time.Millisecond
)

// Setup is likely to be fixture-specific, but Teardown needs to be
// consistent to enable TeardownOnPanic.
type TestFixture interface {
	Teardown(t *testing.T)
}

// TeardownOnPanic can be used to ensure cleanup on setup failure.
func TeardownOnPanic(t *testing.T, f TestFixture) {
	if r := recover(); r != nil {
		f.Teardown(t)
		panic(r)
	}
}

// TestLogger defines operations common across integration and e2e testing
type TestLogger interface {
	Fatalf(format string, args ...interface{})
	Fatal(msg string)
	Logf(format string, args ...interface{})
}

type IntegrationLogger struct {
	t *testing.T
}

func (tr *IntegrationLogger) Logf(format string, args ...interface{}) {
	tr.t.Logf(format, args...)
}

func (tr *IntegrationLogger) Fatalf(format string, args ...interface{}) {
	tr.t.Fatalf(format, args...)
}

func (tr *IntegrationLogger) Fatal(msg string) {
	tr.t.Fatal(msg)
}
