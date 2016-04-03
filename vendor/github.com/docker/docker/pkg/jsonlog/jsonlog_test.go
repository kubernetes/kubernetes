package jsonlog

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/docker/docker/pkg/timeutils"
)

// Invalid json should return an error
func TestWriteLogWithInvalidJSON(t *testing.T) {
	json := strings.NewReader("Invalid json")
	w := bytes.NewBuffer(nil)
	if err := WriteLog(json, w, "json", time.Time{}); err == nil {
		t.Fatalf("Expected an error, got [%v]", w.String())
	}
}

// Any format is valid, it will just print it
func TestWriteLogWithInvalidFormat(t *testing.T) {
	testLine := "Line that thinks that it is log line from docker\n"
	var buf bytes.Buffer
	e := json.NewEncoder(&buf)
	for i := 0; i < 35; i++ {
		e.Encode(JSONLog{Log: testLine, Stream: "stdout", Created: time.Now()})
	}
	w := bytes.NewBuffer(nil)
	if err := WriteLog(&buf, w, "invalid format", time.Time{}); err != nil {
		t.Fatal(err)
	}
	res := w.String()
	t.Logf("Result of WriteLog: %q", res)
	lines := strings.Split(strings.TrimSpace(res), "\n")
	expression := "^invalid format Line that thinks that it is log line from docker$"
	logRe := regexp.MustCompile(expression)
	expectedLines := 35
	if len(lines) != expectedLines {
		t.Fatalf("Must be %v lines but got %d", expectedLines, len(lines))
	}
	for _, l := range lines {
		if !logRe.MatchString(l) {
			t.Fatalf("Log line not in expected format [%v]: %q", expression, l)
		}
	}
}

// Having multiple Log/Stream element
func TestWriteLogWithMultipleStreamLog(t *testing.T) {
	testLine := "Line that thinks that it is log line from docker\n"
	var buf bytes.Buffer
	e := json.NewEncoder(&buf)
	for i := 0; i < 35; i++ {
		e.Encode(JSONLog{Log: testLine, Stream: "stdout", Created: time.Now()})
	}
	w := bytes.NewBuffer(nil)
	if err := WriteLog(&buf, w, "invalid format", time.Time{}); err != nil {
		t.Fatal(err)
	}
	res := w.String()
	t.Logf("Result of WriteLog: %q", res)
	lines := strings.Split(strings.TrimSpace(res), "\n")
	expression := "^invalid format Line that thinks that it is log line from docker$"
	logRe := regexp.MustCompile(expression)
	expectedLines := 35
	if len(lines) != expectedLines {
		t.Fatalf("Must be %v lines but got %d", expectedLines, len(lines))
	}
	for _, l := range lines {
		if !logRe.MatchString(l) {
			t.Fatalf("Log line not in expected format [%v]: %q", expression, l)
		}
	}
}

// Write log with since after created, it won't print anything
func TestWriteLogWithDate(t *testing.T) {
	created, _ := time.Parse("YYYY-MM-dd", "2015-01-01")
	var buf bytes.Buffer
	testLine := "Line that thinks that it is log line from docker\n"
	jsonLog := JSONLog{Log: testLine, Stream: "stdout", Created: created}
	if err := json.NewEncoder(&buf).Encode(jsonLog); err != nil {
		t.Fatal(err)
	}
	w := bytes.NewBuffer(nil)
	if err := WriteLog(&buf, w, "json", time.Now()); err != nil {
		t.Fatal(err)
	}
	res := w.String()
	if res != "" {
		t.Fatalf("Expected empty log, got [%v]", res)
	}
}

// Happy path :)
func TestWriteLog(t *testing.T) {
	testLine := "Line that thinks that it is log line from docker\n"
	format := timeutils.RFC3339NanoFixed
	logs := map[string][]string{
		"":     {"35", "^Line that thinks that it is log line from docker$"},
		"json": {"1", `^{\"log\":\"Line that thinks that it is log line from docker\\n\",\"stream\":\"stdout\",\"time\":.{30,}\"}$`},
		// 30+ symbols, five more can come from system timezone
		format: {"35", `.{30,} Line that thinks that it is log line from docker`},
	}
	for givenFormat, expressionAndLines := range logs {
		expectedLines, _ := strconv.Atoi(expressionAndLines[0])
		expression := expressionAndLines[1]
		var buf bytes.Buffer
		e := json.NewEncoder(&buf)
		for i := 0; i < 35; i++ {
			e.Encode(JSONLog{Log: testLine, Stream: "stdout", Created: time.Now()})
		}
		w := bytes.NewBuffer(nil)
		if err := WriteLog(&buf, w, givenFormat, time.Time{}); err != nil {
			t.Fatal(err)
		}
		res := w.String()
		t.Logf("Result of WriteLog: %q", res)
		lines := strings.Split(strings.TrimSpace(res), "\n")
		if len(lines) != expectedLines {
			t.Fatalf("Must be %v lines but got %d", expectedLines, len(lines))
		}
		logRe := regexp.MustCompile(expression)
		for _, l := range lines {
			if !logRe.MatchString(l) {
				t.Fatalf("Log line not in expected format [%v]: %q", expression, l)
			}
		}
	}
}

func BenchmarkWriteLog(b *testing.B) {
	var buf bytes.Buffer
	e := json.NewEncoder(&buf)
	testLine := "Line that thinks that it is log line from docker\n"
	for i := 0; i < 30; i++ {
		e.Encode(JSONLog{Log: testLine, Stream: "stdout", Created: time.Now()})
	}
	r := bytes.NewReader(buf.Bytes())
	w := ioutil.Discard
	format := timeutils.RFC3339NanoFixed
	b.SetBytes(int64(r.Len()))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := WriteLog(r, w, format, time.Time{}); err != nil {
			b.Fatal(err)
		}
		b.StopTimer()
		r.Seek(0, 0)
		b.StartTimer()
	}
}
