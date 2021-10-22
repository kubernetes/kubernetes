// Copyright 2020 The Kubernetes Authors.
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

package klog

// These helper functions must be in a separate source file because the
// tests in klog_test.go compare the logged source code file name against
// "klog_test.go". "klog_wrappers_test.go" must *not* be logged.

func myInfoS(msg string, keyAndValues ...interface{}) {
	InfoSDepth(1, msg, keyAndValues...)
}

func myErrorS(err error, msg string, keyAndValues ...interface{}) {
	ErrorSDepth(1, err, msg, keyAndValues...)
}
