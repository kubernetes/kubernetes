/*
Copyright 2022 The Kubernetes Authors.

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

package e2edra

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	ginkgotypes "github.com/onsi/ginkgo/v2/types"

	"k8s.io/kubernetes/test/utils/ktesting"
)

var (
	TimeNow = time.Now    // Can be stubbed out for testing.
	Pid     = os.Getpid() // Can be stubbed out for testing.
)

// TODO: replace with helper code from https://github.com/kubernetes/kubernetes/pull/122481 should that get merged - or vice versa.
type ginkgoTB struct {
	ktesting.TB
}

var _ ktesting.ContextTB = &ginkgoTB{}

func GinkgoContextTB() ktesting.ContextTB {
	return &ginkgoTB{
		TB: ginkgo.GinkgoT(),
	}
}

// CleanupCtx implements [ktesting.ContextTB.CleanupCtx]. It's identical to
// ginkgo.DeferCleanup.
func (g *ginkgoTB) CleanupCtx(cb func(context.Context)) {
	ginkgo.GinkgoHelper()
	ginkgo.DeferCleanup(cb)
}

// Log overrides the implementation from Ginkgo to ensure consistent output.
func (g *ginkgoTB) Log(args ...any) {
	log(1, fmt.Sprint(args...))
}

// Logf overrides the implementation from Ginkgo to ensure consistent output.
func (g *ginkgoTB) Logf(format string, args ...any) {
	log(1, fmt.Sprintf(format, args...))
}

// log re-implements klog.Info: same header, but stack unwinding
// with support for ginkgo.GinkgoWriter and skipping stack levels.
func log(offset int, msg string) {
	now := TimeNow()
	file, line := unwind(offset + 1)
	if file == "" {
		file = "???"
		line = 1
	} else if slash := strings.LastIndex(file, "/"); slash >= 0 {
		file = file[slash+1:]
	}
	_, month, day := now.Date()
	hour, minute, second := now.Clock()
	header := fmt.Sprintf("I%02d%02d %02d:%02d:%02d.%06d %d %s:%d]",
		month, day, hour, minute, second, now.Nanosecond()/1000, Pid, file, line)

	_, _ = fmt.Fprintln(ginkgo.GinkgoWriter, header, msg)
}

func unwind(skip int) (string, int) {
	location := ginkgotypes.NewCodeLocation(skip + 1)
	return location.FileName, location.LineNumber
}
