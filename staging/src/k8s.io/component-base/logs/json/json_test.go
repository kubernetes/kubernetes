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
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// TestInfoLoggerInfo test infologger json info format
func TestInfoLoggerInfo(t *testing.T) {
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
			format:     "{\"v\":1,\"ts\":%f,\"msg\":\"test\",\"ns\":\"default\",\"podnum\":2}\n",
			keysValues: []interface{}{"ns", "default", "podnum", 2},
		},
		{
			msg:        "test for strongly typed Zap field",
			format:     "{\"v\":3,\"ts\":%f,\"msg\":\"strongly-typed Zap Field passed to logr\"}\n{\"v\":1,\"ts\":0.000123,\"msg\":\"test for strongly typed Zap field\",\"ns\":\"default\",\"podnum\":2}\n",
			keysValues: []interface{}{"ns", "default", "podnum", 2, zap.Int("attempt", 3), "attempt", "Running", 10},
		},
		{
			msg:        "test for non-string key argument",
			format:     "{\"v\":3,\"ts\":%f,\"msg\":\"non-string key argument passed to logging, ignoring all later arguments\"}\n{\"v\":1,\"ts\":0.000123,\"msg\":\"test for non-string key argument\",\"ns\":\"default\",\"podnum\":2}\n",
			keysValues: []interface{}{"ns", "default", "podnum", 2, 200, "replica", "Running", 10},
		},
	}

	for i, data := range testDataInfo {
		var buffer bytes.Buffer
		writer := bufio.NewWriter(&buffer)
		var sampleInfoLogger = NewJSONLogger(zap.NewExample(), zapcore.AddSync(writer))
		sampleInfoLogger.Info(data.msg, data.keysValues...)
		writer.Flush()
		logStr := buffer.String()
		var ts float64
		fmt.Println(i, logStr)
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

// TestInfoLoggerEnabled test jsonlogger should always enabled
func TestInfoLoggerEnabled(t *testing.T) {
	var sampleInfoLogger = NewJSONLogger(zap.NewExample(), nil)
	if !sampleInfoLogger.Enabled() {
		t.Error("info logger should always enabled")
	}
}

// TestInfoLoggerInfo test infologger V set log level func
func TestZapLoggerV(t *testing.T) {
	var buffer bytes.Buffer
	writer := bufio.NewWriter(&buffer)
	timeNow = func() time.Time {
		return time.Date(1970, time.January, 1, 0, 0, 0, 123, time.UTC)
	}
	var sampleInfoLogger = NewJSONLogger(zap.NewExample(), zapcore.AddSync(writer))
	sampleInfoLogger.V(2).Info("test", "ns", "default", "podnum", 2)
	writer.Flush()
	logStr := buffer.String()
	var ts float64
	expectFormat := `{"v":1,"ts":%f,"msg":"test","ns":"default","podnum":2}`
	n, err := fmt.Sscanf(logStr, expectFormat, &ts)
	if n != 0 || err == nil {
		t.Errorf("log format error: %d elements, error %s:\n%s", n, err, logStr)
	}
	if !assert.Empty(t, logStr) {
		t.Errorf("Info log: %s should empty", logStr)
	}
}

// TestZapLoggerError test infologger json error format
func TestZapLoggerError(t *testing.T) {
	var buffer bytes.Buffer
	writer := bufio.NewWriter(&buffer)
	timeNow = func() time.Time {
		return time.Date(1970, time.January, 1, 0, 0, 0, 123, time.UTC)
	}
	var sampleInfoLogger = NewJSONLogger(zap.NewExample(), zapcore.AddSync(writer))
	sampleInfoLogger.Error(fmt.Errorf("ivailid namespace:%s", "default"), "wrong namespace", "ns", "default", "podnum", 2)
	writer.Flush()
	logStr := buffer.String()
	var ts float64
	expectFormat := `{"v":2,"ts":%f,"msg":"wrong namespace","ns":"default","podnum":2,"err":"ivailid namespace:default"}`
	n, err := fmt.Sscanf(logStr, expectFormat, &ts)
	if n != 1 || err != nil {
		t.Errorf("log format error: %d elements, error %s:\n%s", n, err, logStr)
	}
	expect := fmt.Sprintf(expectFormat, ts)
	if !assert.JSONEq(t, expect, logStr) {
		t.Errorf("Info has wrong format \n expect:%s\n got:%s", expect, logStr)
	}
}
