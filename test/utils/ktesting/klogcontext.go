/*
Copyright 2024 The Kubernetes Authors.

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
	"fmt"
	"strings"
	"time"
)

var timeNow = time.Now // Can be stubbed out for testing.

// withKlogHeader creates a TB where the same "I<date> <time>]" prefix
// gets added to all log output, just as in the klog test logger.
// This is used internally when constructing a TContext for unit testing.
func withKlogHeader(tb TB) TB {
	return klogTB{
		TB: tb,
	}
}

type klogTB struct {
	TB
}

func (k klogTB) Log(args ...any) {
	k.Helper()
	k.TB.Log(header() + strings.TrimSpace(fmt.Sprintln(args...)))
}

func (k klogTB) Logf(format string, args ...any) {
	k.Helper()
	k.TB.Log(header() + strings.TrimSpace(fmt.Sprintf(format, args...)))
}

func (k klogTB) Error(args ...any) {
	k.Helper()
	k.TB.Error(header() + strings.TrimSpace(fmt.Sprintln(args...)))
}

func (k klogTB) Errorf(format string, args ...any) {
	k.Helper()
	k.TB.Error(header() + strings.TrimSpace(fmt.Sprintf(format, args...)))
}

func (k klogTB) Fatal(args ...any) {
	k.Helper()
	k.TB.Fatal(header() + strings.TrimSpace(fmt.Sprintln(args...)))
}

func (k klogTB) Fatalf(format string, args ...any) {
	k.Helper()
	k.TB.Fatal(header() + strings.TrimSpace(fmt.Sprintf(format, args...)))
}

func header() string {
	now := timeNow()
	_, month, day := now.Date()
	hour, minute, second := now.Clock()
	return fmt.Sprintf("I%02d%02d %02d:%02d:%02d.%06d] ",
		month, day, hour, minute, second, now.Nanosecond()/1000)
}
