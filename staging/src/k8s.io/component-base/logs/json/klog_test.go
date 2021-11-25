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

package logs

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"go.uber.org/zap/zapcore"

	"k8s.io/klog/v2"
)

type kmeta struct {
	Name, Namespace string
}

func (k kmeta) GetName() string {
	return k.Name
}

func (k kmeta) GetNamespace() string {
	return k.Namespace
}

var _ klog.KMetadata = kmeta{}

func TestKlogIntegration(t *testing.T) {
	timeNow = func() time.Time {
		return time.Date(1970, time.January, 1, 0, 0, 0, 123, time.UTC)
	}
	fs := flag.FlagSet{}
	klog.InitFlags(&fs)
	err := fs.Set("v", "2")
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
				klog.Info("test ", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test 1\n","v":0}`,
		},
		{
			name: "V(1).Info",
			fun: func() {
				klog.V(1).Info("test ", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test 1\n","v":1}`,
		},
		{
			name: "Infof",
			fun: func() {
				klog.Infof("test %d", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test 1\n","v":0}`,
		},
		{
			name: "V(1).Infof",
			fun: func() {
				klog.V(1).Infof("test %d", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test 1\n","v":1}`,
		},
		{
			name: "Infoln",
			fun: func() {
				klog.Infoln("test", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test 1\n","v":0}`,
		},
		{
			name: "V(1).Infoln",
			fun: func() {
				klog.V(1).Infoln("test", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test 1\n","v":1}`,
		},
		{
			name: "InfoDepth",
			fun: func() {
				klog.InfoDepth(1, "test ", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test 1\n","v":0}`,
		},
		{
			name: "InfoS",
			fun: func() {
				klog.InfoS("test", "count", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test","v":0,"count":1}`,
		},
		{
			name: "V(1).InfoS",
			fun: func() {
				klog.V(1).InfoS("test", "count", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test","v":1,"count":1}`,
		},
		{
			name: "V(2).InfoS",
			fun: func() {
				klog.V(2).InfoS("test", "count", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test","v":2,"count":1}`,
		},
		{
			name: "V(3).InfoS",
			fun: func() {
				klog.V(3).InfoS("test", "count", 1)
			},
			// no output because of threshold 2
		},
		{
			name: "InfoSDepth",
			fun: func() {
				klog.InfoSDepth(1, "test", "count", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test","v":0,"count":1}`,
		},
		{
			name: "KObj",
			fun: func() {
				klog.InfoS("some", "pod", klog.KObj(&kmeta{Name: "pod-1", Namespace: "kube-system"}))
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"some","v":0,"pod":{"name":"pod-1","namespace":"kube-system"}}`,
		},
		{
			name: "KObjs",
			fun: func() {
				klog.InfoS("several", "pods",
					klog.KObjs([]interface{}{
						&kmeta{Name: "pod-1", Namespace: "kube-system"},
						&kmeta{Name: "pod-2", Namespace: "kube-system"},
					}))
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"several","v":0,"pods":[{"name":"pod-1","namespace":"kube-system"},{"name":"pod-2","namespace":"kube-system"}]}`,
		},
		{
			name: "Warning",
			fun: func() {
				klog.Warning("test ", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test 1\n","v":0}`,
		},
		{
			name: "WarningDepth",
			fun: func() {
				klog.WarningDepth(1, "test ", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test 1\n","v":0}`,
		},
		{
			name: "Warningln",
			fun: func() {
				klog.Warningln("test", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test 1\n","v":0}`,
		},
		{
			name: "Warningf",
			fun: func() {
				klog.Warningf("test %d", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test 1\n","v":0}`,
		},
		{
			name: "Error",
			fun: func() {
				klog.Error("test ", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test 1\n"}`,
		},
		{
			name: "ErrorDepth",
			fun: func() {
				klog.ErrorDepth(1, "test ", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test 1\n"}`,
		},
		{
			name: "Errorln",
			fun: func() {
				klog.Errorln("test", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test 1\n"}`,
		},
		{
			name: "Errorf",
			fun: func() {
				klog.Errorf("test %d", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test 1\n"}`,
		},
		{
			name: "ErrorS",
			fun: func() {
				err := errors.New("fail")
				klog.ErrorS(err, "test", "count", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test","count":1,"err":"fail"}`,
		},
		{
			name: "ErrorSDepth",
			fun: func() {
				err := errors.New("fail")
				klog.ErrorSDepth(1, err, "test", "count", 1)
			},
			format: `{"ts":%f,"caller":"json/klog_test.go:%d","msg":"test","count":1,"err":"fail"}`,
		},
	}
	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			var buffer bytes.Buffer
			writer := zapcore.AddSync(&buffer)
			logger, _ := NewJSONLogger(writer, writer)
			klog.SetLogger(logger)
			defer klog.ClearLogger()

			tc.fun()
			var ts float64
			var lineNo int
			logString := strings.TrimSuffix(buffer.String(), "\n")
			if tc.format == "" {
				if logString != "" {
					t.Fatalf("expected no output, got: %s", logString)
				}
				return
			}
			n, err := fmt.Sscanf(logString, tc.format, &ts, &lineNo)
			if n != 2 || err != nil {
				t.Errorf("log format error: %d elements, error %s:\n%s", n, err, logString)
			}
			expect := fmt.Sprintf(tc.format, ts, lineNo)
			if !assert.Equal(t, expect, logString) {
				t.Errorf("Info has wrong format \n expect:%s\n got:%s", expect, logString)
			}

		})
	}
}

// TestKlogV test klog -v(--verbose) func available with json logger
func TestKlogV(t *testing.T) {
	var buffer testBuff
	writer := zapcore.AddSync(&buffer)
	logger, _ := NewJSONLogger(writer, writer)
	klog.SetLogger(logger)
	defer klog.ClearLogger()
	fs := flag.FlagSet{}
	klog.InitFlags(&fs)
	totalLogsWritten := 0

	defer func() {
		err := fs.Set("v", "0")
		if err != nil {
			t.Fatalf("Failed to reset verbosity to 0")
		}
	}()

	for i := 0; i < 11; i++ {
		err := fs.Set("v", fmt.Sprintf("%d", i))
		if err != nil {
			t.Fatalf("Failed to set verbosity")
		}
		for j := 0; j < 11; j++ {
			klog.V(klog.Level(j)).Info("test", "time", time.Microsecond)
			logWritten := buffer.writeCount > 0
			totalLogsWritten += buffer.writeCount
			buffer.writeCount = 0
			if logWritten == (i < j) {
				t.Errorf("klog.V(%d).Info(...) wrote log when -v=%d", j, i)
			}
		}
	}
	if totalLogsWritten != 66 {
		t.Fatalf("Unexpected number of logs written, got %d, expected 66", totalLogsWritten)
	}
}
