// Copyright 2022 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package backend

import (
	"os"
	"runtime/debug"
	"strings"

	"go.uber.org/zap"
)

const (
	ENV_VERIFY           = "ETCD_VERIFY"
	ENV_VERIFY_ALL_VALUE = "all"
	ENV_VERIFY_LOCK      = "lock"
)

func ValidateCalledInsideApply(lg *zap.Logger) {
	if !verifyLockEnabled() {
		return
	}
	if !insideApply() {
		lg.Panic("Called outside of APPLY!", zap.Stack("stacktrace"))
	}
}

func ValidateCalledOutSideApply(lg *zap.Logger) {
	if !verifyLockEnabled() {
		return
	}
	if insideApply() {
		lg.Panic("Called inside of APPLY!", zap.Stack("stacktrace"))
	}
}

func ValidateCalledInsideUnittest(lg *zap.Logger) {
	if !verifyLockEnabled() {
		return
	}
	if !insideUnittest() {
		lg.Fatal("Lock called outside of unit test!", zap.Stack("stacktrace"))
	}
}

func verifyLockEnabled() bool {
	return os.Getenv(ENV_VERIFY) == ENV_VERIFY_ALL_VALUE || os.Getenv(ENV_VERIFY) == ENV_VERIFY_LOCK
}

func insideApply() bool {
	stackTraceStr := string(debug.Stack())
	return strings.Contains(stackTraceStr, ".applyEntries")
}

func insideUnittest() bool {
	stackTraceStr := string(debug.Stack())
	return strings.Contains(stackTraceStr, "_test.go") && !strings.Contains(stackTraceStr, "tests/")
}
