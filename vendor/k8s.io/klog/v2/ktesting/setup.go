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

package ktesting

import (
	"context"

	"github.com/go-logr/logr"
)

// DefaultConfig is the global default logging configuration for a unit
// test. It is used by NewTestContext and k8s.io/klogr/testing/init.
//
// # Experimental
//
// Notice: This variable is EXPERIMENTAL and may be changed or removed in a
// later release.
var DefaultConfig = NewConfig()

// NewTestContext returns a logger and context for use in a unit test case or
// benchmark. The tl parameter can be a testing.T or testing.B pointer that
// will receive all log output. Importing k8s.io/klogr/testing/init will add
// command line flags that modify the configuration of that log output.
//
// # Experimental
//
// Notice: This function is EXPERIMENTAL and may be changed or removed in a
// later release.
func NewTestContext(tl TL) (logr.Logger, context.Context) {
	logger := NewLogger(tl, DefaultConfig)
	ctx := logr.NewContext(context.Background(), logger)
	return logger, ctx

}
