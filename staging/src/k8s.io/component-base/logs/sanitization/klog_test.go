/*
Copyright 2020 The Kubernetes Authors.

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

package sanitization

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"regexp"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/klog/v2"
)

func TestKlogIntegration(t *testing.T) {

	fs := flag.FlagSet{}
	klog.InitFlags(&fs)
	err := fs.Set("v", "1")
	if err != nil {
		t.Fatalf("Failed to set verbosity")
	}
	err = fs.Set("logtostderr", "false")
	if err != nil {
		t.Fatalf("Failed to set verbosity")
	}
	tcs := []struct {
		name   string
		fun    func()
		format string
	}{
		{
			name: "Info",
			fun: func() {
				klog.Info("test ", datapolItem())
			},
			format: `I%s %s %d klog_test.go:%d] Log message has been redacted. Log argument #1 contains: [password]`,
		},
		{
			name: "V(1).Info",
			fun: func() {
				klog.V(1).Info("test ", datapolItem())
			},
			format: `I%s %s %d klog_test.go:%d] Log message has been redacted. Log argument #1 contains: [password]`,
		},
		{
			name: "Infof",
			fun: func() {
				klog.Infof("test %v", datapolItem())
			},
			format: `I%s %s %d klog_test.go:%d] Log message has been redacted. Log argument #0 contains: [password]`,
		},
		{
			name: "V(1).Infof",
			fun: func() {
				klog.V(1).Infof("test %v", datapolItem())
			},
			format: `I%s %s %d klog_test.go:%d] Log message has been redacted. Log argument #0 contains: [password]`,
		},
		{
			name: "Infoln",
			fun: func() {
				klog.Infoln("test", datapolItem())
			},
			format: `I%s %s %d klog_test.go:%d] Log message has been redacted. Log argument #1 contains: [password]`,
		},
		{
			name: "V(1).Infoln",
			fun: func() {
				klog.V(1).Infoln("test", datapolItem())
			},
			format: `I%s %s %d klog_test.go:%d] Log message has been redacted. Log argument #1 contains: [password]`,
		},
		{
			name: "InfoDepth",
			fun: func() {
				klog.InfoDepth(1, "test ", datapolItem())
			},
			format: `I%s %s %d klog_test.go:%d] Log message has been redacted. Log argument #1 contains: [password]`,
		},
		{
			name: "InfoS",
			fun: func() {
				klog.InfoS("test", "data", datapolItem())
			},
			format: `I%s %s %d klog_test.go:%d] "Log message has been redacted." key="data" types=[password]`,
		},
		{
			name: "V(1).InfoS",
			fun: func() {
				klog.V(1).InfoS("test", "data", datapolItem())
			},
			format: `I%s %s %d klog_test.go:%d] "Log message has been redacted." key="data" types=[password]`,
		},
		{
			name: "InfoSDepth",
			fun: func() {
				klog.InfoSDepth(1, "test", "data", datapolItem())
			},
			format: `I%s %s %d klog_test.go:%d] "Log message has been redacted." key="data" types=[password]`,
		},
		{
			name: "Warning",
			fun: func() {
				klog.Warning("test ", datapolItem())
			},
			format: `W%s %s %d klog_test.go:%d] Log message has been redacted. Log argument #1 contains: [password]`,
		},
		{
			name: "WarningDepth",
			fun: func() {
				klog.WarningDepth(1, "test ", datapolItem())
			},
			format: `W%s %s %d klog_test.go:%d] Log message has been redacted. Log argument #1 contains: [password]`,
		},
		{
			name: "Warningln",
			fun: func() {
				klog.Warningln("test", datapolItem())
			},
			format: `W%s %s %d klog_test.go:%d] Log message has been redacted. Log argument #1 contains: [password]`,
		},
		{
			name: "Warningf",
			fun: func() {
				klog.Warningf("test %d", datapolItem())
			},
			format: `W%s %s %d klog_test.go:%d] Log message has been redacted. Log argument #0 contains: [password]`,
		},
		{
			name: "Error",
			fun: func() {
				klog.Error("test ", datapolItem())
			},
			format: `E%s %s %d klog_test.go:%d] Log message has been redacted. Log argument #1 contains: [password]`,
		},
		{
			name: "ErrorDepth",
			fun: func() {
				klog.ErrorDepth(1, "test ", datapolItem())
			},
			format: `E%s %s %d klog_test.go:%d] Log message has been redacted. Log argument #1 contains: [password]`,
		},
		{
			name: "Errorln",
			fun: func() {
				klog.Errorln("test", datapolItem())
			},
			format: `E%s %s %d klog_test.go:%d] Log message has been redacted. Log argument #1 contains: [password]`,
		},
		{
			name: "Errorf",
			fun: func() {
				klog.Errorf("test %d", datapolItem())
			},
			format: `E%s %s %d klog_test.go:%d] Log message has been redacted. Log argument #0 contains: [password]`,
		},
		{
			name: "ErrorS",
			fun: func() {
				err := errors.New("fail")
				klog.ErrorS(err, "test", "data", datapolItem())
			},
			format: `E%s %s %d klog_test.go:%d] "Log message has been redacted." err="fail" key="data" types=[password]`,
		},
		{
			name: "ErrorSDepth",
			fun: func() {
				err := errors.New("fail")
				klog.ErrorSDepth(1, err, "test", "data", datapolItem())
			},
			format: `E%s %s %d klog_test.go:%d] "Log message has been redacted." err="fail" key="data" types=[password]`,
		},
	}
	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			var buffer bytes.Buffer
			klog.SetOutputBySeverity("INFO", &buffer)
			defer klog.SetOutputBySeverity("INFO", nil)
			klog.SetLogFilter(&SanitizingFilter{})
			defer klog.SetLogFilter(nil)

			tc.fun()
			var date string
			var time string
			var pid uint64
			var lineNum uint64

			logString := normalizeKlogLine(strings.TrimSuffix(buffer.String(), "\n"))
			n, err := fmt.Sscanf(logString, tc.format, &date, &time, &pid, &lineNum)
			if n != 4 || err != nil {
				t.Errorf("log format error: %d elements, error %s:\n%s", n, err, logString)
			}
			expect := fmt.Sprintf(tc.format, date, time, pid, lineNum)
			if !assert.Equal(t, expect, logString) {
				t.Errorf("Info has wrong format \n expect:%s\n got:%s", expect, logString)
			}

		})
	}
}

var (
	re = regexp.MustCompile(`\s{2,}`)
)

// normalizeKlogLine removes duplicate whitespaces to make lines match no matter the environment.
// Klog padds the log header to try to maintain same width. Depending on how high the process
// pid is it make lead to additional whitespaces.
func normalizeKlogLine(s string) string {
	return re.ReplaceAllString(s, " ")
}
