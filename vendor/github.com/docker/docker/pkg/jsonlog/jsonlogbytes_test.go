package jsonlog

import (
	"bytes"
	"regexp"
	"testing"
)

func TestJSONLogBytesMarshalJSONBuf(t *testing.T) {
	logs := map[*JSONLogBytes]string{
		&JSONLogBytes{Log: []byte(`"A log line with \\"`)}:           `^{\"log\":\"\\\"A log line with \\\\\\\\\\\"\",\"time\":}$`,
		&JSONLogBytes{Log: []byte("A log line")}:                     `^{\"log\":\"A log line\",\"time\":}$`,
		&JSONLogBytes{Log: []byte("A log line with \r")}:             `^{\"log\":\"A log line with \\r\",\"time\":}$`,
		&JSONLogBytes{Log: []byte("A log line with & < >")}:          `^{\"log\":\"A log line with \\u0026 \\u003c \\u003e\",\"time\":}$`,
		&JSONLogBytes{Log: []byte("A log line with utf8 : 🚀 ψ ω β")}: `^{\"log\":\"A log line with utf8 : 🚀 ψ ω β\",\"time\":}$`,
		&JSONLogBytes{Stream: "stdout"}:                              `^{\"stream\":\"stdout\",\"time\":}$`,
		&JSONLogBytes{Stream: "stdout", Log: []byte("A log line")}:   `^{\"log\":\"A log line\",\"stream\":\"stdout\",\"time\":}$`,
		&JSONLogBytes{Created: "time"}:                               `^{\"time\":time}$`,
		&JSONLogBytes{}:                                              `^{\"time\":}$`,
		// These ones are a little weird
		&JSONLogBytes{Log: []byte("\u2028 \u2029")}: `^{\"log\":\"\\u2028 \\u2029\",\"time\":}$`,
		&JSONLogBytes{Log: []byte{0xaF}}:            `^{\"log\":\"\\ufffd\",\"time\":}$`,
		&JSONLogBytes{Log: []byte{0x7F}}:            `^{\"log\":\"\x7f\",\"time\":}$`,
	}
	for jsonLog, expression := range logs {
		var buf bytes.Buffer
		if err := jsonLog.MarshalJSONBuf(&buf); err != nil {
			t.Fatal(err)
		}
		res := buf.String()
		t.Logf("Result of WriteLog: %q", res)
		logRe := regexp.MustCompile(expression)
		if !logRe.MatchString(res) {
			t.Fatalf("Log line not in expected format [%v]: %q", expression, res)
		}
	}
}
