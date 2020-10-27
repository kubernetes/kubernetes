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
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"k8s.io/klog/v2"
)

// TestZapLoggerInfo test ZapLogger json info format
func TestZapLoggerInfo(t *testing.T) {
	timeNow = func() time.Time {
		return time.Date(1970, time.January, 1, 0, 0, 0, 123, time.UTC)
	}
	var testDataInfo = []struct {
		msg        string
		format     string
		keysValues []interface{}
	}{
		{
			msg:        "test",
			format:     "{\"ts\":%f,\"msg\":\"test\",\"v\":0,\"ns\":\"default\",\"podnum\":2}\n",
			keysValues: []interface{}{"ns", "default", "podnum", 2},
		},
		{
			msg:        "test for strongly typed Zap field",
			format:     "{\"ts\":%f,\"msg\":\"strongly-typed Zap Field passed to logr\",\"v\":0}\n{\"ts\":0.000123,\"msg\":\"test for strongly typed Zap field\",\"v\":0,\"ns\":\"default\",\"podnum\":2}\n",
			keysValues: []interface{}{"ns", "default", "podnum", 2, zap.Int("attempt", 3), "attempt", "Running", 10},
		},
		{
			msg:        "test for non-string key argument",
			format:     "{\"ts\":%f,\"msg\":\"non-string key argument passed to logging, ignoring all later arguments\",\"v\":0}\n{\"ts\":0.000123,\"msg\":\"test for non-string key argument\",\"v\":0,\"ns\":\"default\",\"podnum\":2}\n",
			keysValues: []interface{}{"ns", "default", "podnum", 2, 200, "replica", "Running", 10},
		},
	}

	for _, data := range testDataInfo {
		var buffer bytes.Buffer
		writer := bufio.NewWriter(&buffer)
		var sampleInfoLogger = NewJSONLogger(zapcore.AddSync(writer))
		sampleInfoLogger.Info(data.msg, data.keysValues...)
		writer.Flush()
		logStr := buffer.String()
		var ts float64
		n, err := fmt.Sscanf(logStr, data.format, &ts)
		if n != 1 || err != nil {
			t.Errorf("log format error: %d elements, error %s:\n%s", n, err, logStr)
		}
		expect := fmt.Sprintf(data.format, ts)
		if !assert.Equal(t, expect, logStr) {
			t.Errorf("Info has wrong format \n expect:%s\n got:%s", expect, logStr)
		}
	}
}

// TestZapLoggerEnabled test ZapLogger enabled
func TestZapLoggerEnabled(t *testing.T) {
	var sampleInfoLogger = NewJSONLogger(nil)
	for i := 0; i < 11; i++ {
		if !sampleInfoLogger.V(i).Enabled() {
			t.Errorf("V(%d).Info should be enabled", i)
		}
	}
}

// TestZapLoggerV test ZapLogger V set log level func
func TestZapLoggerV(t *testing.T) {
	timeNow = func() time.Time {
		return time.Date(1970, time.January, 1, 0, 0, 0, 123, time.UTC)
	}

	for i := 0; i < 11; i++ {
		var buffer bytes.Buffer
		writer := bufio.NewWriter(&buffer)
		var sampleInfoLogger = NewJSONLogger(zapcore.AddSync(writer))
		sampleInfoLogger.V(i).Info("test", "ns", "default", "podnum", 2)
		writer.Flush()
		logStr := buffer.String()
		var v int
		var expectFormat string
		expectFormat = "{\"ts\":0.000123,\"msg\":\"test\",\"v\":%d,\"ns\":\"default\",\"podnum\":2}\n"
		n, err := fmt.Sscanf(logStr, expectFormat, &v)
		if n != 1 || err != nil {
			t.Errorf("log format error: %d elements, error %s:\n%s", n, err, logStr)
		}
		if v != i {
			t.Errorf("V(%d).Info...) returned v=%d. expected v=%d", i, v, i)
		}
		expect := fmt.Sprintf(expectFormat, v)
		if !assert.Equal(t, logStr, expect) {
			t.Errorf("V(%d).Info has wrong format \n expect:%s\n got:%s", i, expect, logStr)
		}
		buffer.Reset()
	}
}

// TestZapLoggerError test ZapLogger json error format
func TestZapLoggerError(t *testing.T) {
	var buffer bytes.Buffer
	writer := bufio.NewWriter(&buffer)
	timeNow = func() time.Time {
		return time.Date(1970, time.January, 1, 0, 0, 0, 123, time.UTC)
	}
	var sampleInfoLogger = NewJSONLogger(zapcore.AddSync(writer))
	sampleInfoLogger.Error(fmt.Errorf("ivailid namespace:%s", "default"), "wrong namespace", "ns", "default", "podnum", 2)
	writer.Flush()
	logStr := buffer.String()
	var ts float64
	expectFormat := `{"ts":%f,"msg":"wrong namespace","v":0,"ns":"default","podnum":2,"err":"ivailid namespace:default"}`
	n, err := fmt.Sscanf(logStr, expectFormat, &ts)
	if n != 1 || err != nil {
		t.Errorf("log format error: %d elements, error %s:\n%s", n, err, logStr)
	}
	expect := fmt.Sprintf(expectFormat, ts)
	if !assert.JSONEq(t, expect, logStr) {
		t.Errorf("Info has wrong format \n expect:%s\n got:%s", expect, logStr)
	}
}

// TestKlogV test klog -v(--verbose) func available with json logger
func TestKlogV(t *testing.T) {
	var buffer testBuff
	logger := NewJSONLogger(&buffer)
	klog.SetLogger(logger)
	defer klog.SetLogger(nil)
	fs := flag.FlagSet{}
	klog.InitFlags(&fs)
	totalLogsWritten := 0

	defer fs.Set("v", "0")

	for i := 0; i < 11; i++ {
		err := fs.Set("v", fmt.Sprintf("%d", i))
		if err != nil {
			t.Fatalf("Failed to set verbosity")
		}
		for j := 0; j < 11; j++ {
			klog.V(klog.Level(j)).Info("test")
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

type testBuff struct {
	writeCount int
}

// Sync syncs data to file
func (b *testBuff) Sync() error {
	return nil
}

// Write writes data to buffer
func (b *testBuff) Write(p []byte) (int, error) {
	b.writeCount++
	return len(p), nil
}
