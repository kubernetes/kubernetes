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
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	logsapi "k8s.io/component-base/logs/api/v1"
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
		names      []string
	}{
		{
			msg:        "test",
			format:     "{\"ts\":%f,\"caller\":\"json/json_test.go:%d\",\"msg\":\"test\",\"v\":0,\"ns\":\"default\",\"podnum\":2}\n",
			keysValues: []interface{}{"ns", "default", "podnum", 2},
		},
		{
			msg:        "test for strongly typed Zap field",
			format:     "{\"ts\":%f,\"caller\":\"json/json_test.go:%d\",\"msg\":\"strongly-typed Zap Field passed to logr\",\"zap field\":{\"Key\":\"attempt\",\"Type\":11,\"Integer\":3,\"String\":\"\",\"Interface\":null}}\n{\"ts\":%f,\"caller\":\"json/json_test.go:%d\",\"msg\":\"test for strongly typed Zap field\",\"v\":0,\"ns\":\"default\",\"podnum\":2}\n",
			keysValues: []interface{}{"ns", "default", "podnum", 2, zap.Int("attempt", 3), "attempt", "Running", 10},
		},
		{
			msg:        "test for non-string key argument",
			format:     "{\"ts\":%f,\"caller\":\"json/json_test.go:%d\",\"msg\":\"non-string key argument passed to logging, ignoring all later arguments\",\"invalid key\":200}\n{\"ts\":%f,\"caller\":\"json/json_test.go:%d\",\"msg\":\"test for non-string key argument\",\"v\":0,\"ns\":\"default\",\"podnum\":2}\n",
			keysValues: []interface{}{"ns", "default", "podnum", 2, 200, "replica", "Running", 10},
		},
		{
			msg:        "test for duration value argument",
			format:     "{\"ts\":%f,\"caller\":\"json/json_test.go:%d\",\"msg\":\"test for duration value argument\",\"v\":0,\"duration\":\"5s\"}\n",
			keysValues: []interface{}{"duration", time.Duration(5 * time.Second)},
		},
		{
			msg:    "test for WithName",
			names:  []string{"hello", "world"},
			format: "{\"ts\":%f,\"logger\":\"hello.world\",\"caller\":\"json/json_test.go:%d\",\"msg\":\"test for WithName\",\"v\":0}\n",
		},
		{
			msg:        "test for duplicate keys",
			format:     "{\"ts\":%f,\"caller\":\"json/json_test.go:%d\",\"msg\":\"test for duplicate keys\",\"v\":0,\"akey\":\"avalue\",\"akey\":\"anothervalue\"}\n",
			keysValues: []interface{}{"akey", "avalue", "akey", "anothervalue"},
		},
	}

	for _, data := range testDataInfo {
		var buffer bytes.Buffer
		writer := zapcore.AddSync(&buffer)
		sampleInfoLogger, _ := NewJSONLogger(0, writer, nil, nil)
		for _, name := range data.names {
			// nolint:logcheck // This intentionally ignore the feature gate and always tests with a name.
			sampleInfoLogger = sampleInfoLogger.WithName(name)
		}
		// nolint:logcheck // The linter cannot and doesn't need to check the key/value pairs.
		sampleInfoLogger.Info(data.msg, data.keysValues...)
		logStr := buffer.String()

		logStrLines := strings.Split(logStr, "\n")
		dataFormatLines := strings.Split(data.format, "\n")
		if !assert.Equal(t, len(logStrLines), len(dataFormatLines)) {
			t.Errorf("Info has wrong format: no. of lines in log is incorrect \n expect:%d\n got:%d", len(dataFormatLines), len(logStrLines))
		}

		for i := range logStrLines {
			if len(logStrLines[i]) == 0 && len(dataFormatLines[i]) == 0 {
				continue
			}
			var ts float64
			var lineNo int
			n, err := fmt.Sscanf(logStrLines[i], dataFormatLines[i], &ts, &lineNo)
			if n != 2 || err != nil {
				t.Errorf("log format error: %d elements, error %s:\n%s", n, err, logStrLines[i])
			}
			expect := fmt.Sprintf(dataFormatLines[i], ts, lineNo)
			if !assert.Equal(t, expect, logStrLines[i]) {
				t.Errorf("Info has wrong format \n expect:%s\n got:%s", expect, logStrLines[i])
			}
		}
	}
}

// TestZapLoggerEnabled test ZapLogger enabled
func TestZapLoggerEnabled(t *testing.T) {
	verbosityLevel := 10
	sampleInfoLogger, _ := NewJSONLogger(logsapi.VerbosityLevel(verbosityLevel), nil, nil, nil)
	for v := 0; v <= verbosityLevel; v++ {
		enabled := sampleInfoLogger.V(v).Enabled()
		expectEnabled := v <= verbosityLevel
		if !expectEnabled && enabled {
			t.Errorf("V(%d).Info should be disabled", v)
		}
		if expectEnabled && !enabled {
			t.Errorf("V(%d).Info should be enabled", v)
		}
	}
}

// TestZapLoggerV test ZapLogger V set log level func
func TestZapLoggerV(t *testing.T) {
	timeNow = func() time.Time {
		return time.Date(1970, time.January, 1, 0, 0, 0, 123, time.UTC)
	}

	verbosityLevel := 10
	for v := 0; v <= verbosityLevel; v++ {
		var buffer bytes.Buffer
		writer := zapcore.AddSync(&buffer)
		sampleInfoLogger, _ := NewJSONLogger(logsapi.VerbosityLevel(verbosityLevel), writer, nil, nil)
		sampleInfoLogger.V(v).Info("test", "ns", "default", "podnum", 2, "time", time.Microsecond)
		logStr := buffer.String()

		shouldHaveLogged := v <= verbosityLevel
		if logged := logStr != ""; logged != shouldHaveLogged {
			if logged {
				t.Fatalf("Expected no output at v=%d, got: %s", v, logStr)
			}
			t.Fatalf("Expected output at v=%d, got none.", v)
		}
		if !shouldHaveLogged {
			continue
		}

		var actualV, lineNo int
		expectFormat := "{\"ts\":0.000123,\"caller\":\"json/json_test.go:%d\",\"msg\":\"test\",\"v\":%d,\"ns\":\"default\",\"podnum\":2,\"time\":\"1µs\"}\n"
		n, err := fmt.Sscanf(logStr, expectFormat, &lineNo, &actualV)
		if n != 2 || err != nil {
			t.Errorf("log format error: %d elements, error %s:\n%s", n, err, logStr)
		}
		if actualV != v {
			t.Errorf("V(%d).Info...) returned v=%d. expected v=%d", v, actualV, v)
		}
		expect := fmt.Sprintf(expectFormat, lineNo, v)
		if !assert.Equal(t, logStr, expect) {
			t.Errorf("V(%d).Info has wrong format \n expect:%s\n got:%s", v, expect, logStr)
		}
		buffer.Reset()
	}
}

// TestZapLoggerError test ZapLogger json error format
func TestZapLoggerError(t *testing.T) {
	var buffer bytes.Buffer
	writer := zapcore.AddSync(&buffer)
	timeNow = func() time.Time {
		return time.Date(1970, time.January, 1, 0, 0, 0, 123, time.UTC)
	}
	sampleInfoLogger, _ := NewJSONLogger(0, writer, nil, nil)
	sampleInfoLogger.Error(fmt.Errorf("invalid namespace:%s", "default"), "wrong namespace", "ns", "default", "podnum", 2, "time", time.Microsecond)
	logStr := buffer.String()
	var ts float64
	var lineNo int
	expectFormat := `{"ts":%f,"caller":"json/json_test.go:%d","msg":"wrong namespace","ns":"default","podnum":2,"time":"1µs","err":"invalid namespace:default"}`
	n, err := fmt.Sscanf(logStr, expectFormat, &ts, &lineNo)
	if n != 2 || err != nil {
		t.Errorf("log format error: %d elements, error %s:\n%s", n, err, logStr)
	}
	expect := fmt.Sprintf(expectFormat, ts, lineNo)
	if !assert.JSONEq(t, expect, logStr) {
		t.Errorf("Info has wrong format \n expect:%s\n got:%s", expect, logStr)
	}
}

func TestZapLoggerStreams(t *testing.T) {
	var infoBuffer, errorBuffer bytes.Buffer
	log, _ := NewJSONLogger(0, zapcore.AddSync(&infoBuffer), zapcore.AddSync(&errorBuffer), nil)

	log.Error(fmt.Errorf("some error"), "failed")
	log.Info("hello world")

	logStr := errorBuffer.String()
	var ts float64
	var lineNo int
	expectFormat := `{"ts":%f,"caller":"json/json_test.go:%d","msg":"failed","err":"some error"}`
	n, err := fmt.Sscanf(logStr, expectFormat, &ts, &lineNo)
	if n != 2 || err != nil {
		t.Errorf("error log format error: %d elements, error %s:\n%s", n, err, logStr)
	}
	expect := fmt.Sprintf(expectFormat, ts, lineNo)
	if !assert.JSONEq(t, expect, logStr) {
		t.Errorf("error log has wrong format \n expect:%s\n got:%s", expect, logStr)
	}

	logStr = infoBuffer.String()
	expectFormat = `{"ts":%f,"caller":"json/json_test.go:%d","msg":"hello world","v":0}`
	n, err = fmt.Sscanf(logStr, expectFormat, &ts, &lineNo)
	if n != 2 || err != nil {
		t.Errorf("info log format error: %d elements, error %s:\n%s", n, err, logStr)
	}
	expect = fmt.Sprintf(expectFormat, ts, lineNo)
	if !assert.JSONEq(t, expect, logStr) {
		t.Errorf("info has wrong format \n expect:%s\n got:%s", expect, logStr)
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
