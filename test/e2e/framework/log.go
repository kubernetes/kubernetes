/*
Copyright 2019 The Kubernetes Authors.

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
	"fmt"

	"github.com/onsi/ginkgo/v2"
)

// Logf logs the info.
//
// Use this instead of `klog.Infof` because stack unwinding automatically
// skips over helper functions which marked themselves as helper by
// calling [ginkgo.GinkgoHelper].
func Logf(format string, args ...interface{}) {
	log(1, fmt.Sprintf(format, args...))
}

// Failf logs the fail info, including a stack trace starts with its direct caller
// (for example, for call chain f -> g -> Failf("foo", ...) error would be logged for "g").
func Failf(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	skip := 1
	ginkgo.Fail(msg, skip)
	panic("unreachable")
}

// Fail is an alias for ginkgo.Fail.
var Fail = ginkgo.Fail
