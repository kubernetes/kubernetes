package jsonfilelog

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
	"time"

	"github.com/docker/docker/daemon/logger"
	"github.com/docker/docker/pkg/jsonlog"
)

func TestJSONFileLogger(t *testing.T) {
	cid := "a7317399f3f857173c6179d44823594f8294678dea9999662e5c625b5a1c7657"
	tmp, err := ioutil.TempDir("", "docker-logger-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)
	filename := filepath.Join(tmp, "container.log")
	l, err := New(logger.Info{
		ContainerID: cid,
		LogPath:     filename,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()

	if err := l.Log(&logger.Message{Line: []byte("line1"), Source: "src1"}); err != nil {
		t.Fatal(err)
	}
	if err := l.Log(&logger.Message{Line: []byte("line2"), Source: "src2"}); err != nil {
		t.Fatal(err)
	}
	if err := l.Log(&logger.Message{Line: []byte("line3"), Source: "src3"}); err != nil {
		t.Fatal(err)
	}
	res, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Fatal(err)
	}
	expected := `{"log":"line1\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
{"log":"line2\n","stream":"src2","time":"0001-01-01T00:00:00Z"}
{"log":"line3\n","stream":"src3","time":"0001-01-01T00:00:00Z"}
`

	if string(res) != expected {
		t.Fatalf("Wrong log content: %q, expected %q", res, expected)
	}
}

func BenchmarkJSONFileLogger(b *testing.B) {
	cid := "a7317399f3f857173c6179d44823594f8294678dea9999662e5c625b5a1c7657"
	tmp, err := ioutil.TempDir("", "docker-logger-")
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(tmp)
	filename := filepath.Join(tmp, "container.log")
	l, err := New(logger.Info{
		ContainerID: cid,
		LogPath:     filename,
	})
	if err != nil {
		b.Fatal(err)
	}
	defer l.Close()

	testLine := "Line that thinks that it is log line from docker\n"
	msg := &logger.Message{Line: []byte(testLine), Source: "stderr", Timestamp: time.Now().UTC()}
	jsonlog, err := (&jsonlog.JSONLog{Log: string(msg.Line) + "\n", Stream: msg.Source, Created: msg.Timestamp}).MarshalJSON()
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(jsonlog)+1) * 30)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 30; j++ {
			if err := l.Log(msg); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func TestJSONFileLoggerWithOpts(t *testing.T) {
	cid := "a7317399f3f857173c6179d44823594f8294678dea9999662e5c625b5a1c7657"
	tmp, err := ioutil.TempDir("", "docker-logger-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)
	filename := filepath.Join(tmp, "container.log")
	config := map[string]string{"max-file": "2", "max-size": "1k"}
	l, err := New(logger.Info{
		ContainerID: cid,
		LogPath:     filename,
		Config:      config,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	for i := 0; i < 20; i++ {
		if err := l.Log(&logger.Message{Line: []byte("line" + strconv.Itoa(i)), Source: "src1"}); err != nil {
			t.Fatal(err)
		}
	}
	res, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Fatal(err)
	}
	penUlt, err := ioutil.ReadFile(filename + ".1")
	if err != nil {
		t.Fatal(err)
	}

	expectedPenultimate := `{"log":"line0\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
{"log":"line1\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
{"log":"line2\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
{"log":"line3\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
{"log":"line4\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
{"log":"line5\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
{"log":"line6\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
{"log":"line7\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
{"log":"line8\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
{"log":"line9\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
{"log":"line10\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
{"log":"line11\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
{"log":"line12\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
{"log":"line13\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
{"log":"line14\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
{"log":"line15\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
`
	expected := `{"log":"line16\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
{"log":"line17\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
{"log":"line18\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
{"log":"line19\n","stream":"src1","time":"0001-01-01T00:00:00Z"}
`

	if string(res) != expected {
		t.Fatalf("Wrong log content: %q, expected %q", res, expected)
	}
	if string(penUlt) != expectedPenultimate {
		t.Fatalf("Wrong log content: %q, expected %q", penUlt, expectedPenultimate)
	}

}

func TestJSONFileLoggerWithLabelsEnv(t *testing.T) {
	cid := "a7317399f3f857173c6179d44823594f8294678dea9999662e5c625b5a1c7657"
	tmp, err := ioutil.TempDir("", "docker-logger-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)
	filename := filepath.Join(tmp, "container.log")
	config := map[string]string{"labels": "rack,dc", "env": "environ,debug,ssl", "env-regex": "^dc"}
	l, err := New(logger.Info{
		ContainerID:     cid,
		LogPath:         filename,
		Config:          config,
		ContainerLabels: map[string]string{"rack": "101", "dc": "lhr"},
		ContainerEnv:    []string{"environ=production", "debug=false", "port=10001", "ssl=true", "dc_region=west"},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	if err := l.Log(&logger.Message{Line: []byte("line"), Source: "src1"}); err != nil {
		t.Fatal(err)
	}
	res, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Fatal(err)
	}

	var jsonLog jsonlog.JSONLogs
	if err := json.Unmarshal(res, &jsonLog); err != nil {
		t.Fatal(err)
	}
	extra := make(map[string]string)
	if err := json.Unmarshal(jsonLog.RawAttrs, &extra); err != nil {
		t.Fatal(err)
	}
	expected := map[string]string{
		"rack":      "101",
		"dc":        "lhr",
		"environ":   "production",
		"debug":     "false",
		"ssl":       "true",
		"dc_region": "west",
	}
	if !reflect.DeepEqual(extra, expected) {
		t.Fatalf("Wrong log attrs: %q, expected %q", extra, expected)
	}
}

func BenchmarkJSONFileLoggerWithReader(b *testing.B) {
	b.StopTimer()
	b.ResetTimer()
	cid := "a7317399f3f857173c6179d44823594f8294678dea9999662e5c625b5a1c7657"
	dir, err := ioutil.TempDir("", "json-logger-bench")
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(dir)

	l, err := New(logger.Info{
		ContainerID: cid,
		LogPath:     filepath.Join(dir, "container.log"),
	})
	if err != nil {
		b.Fatal(err)
	}
	defer l.Close()
	msg := &logger.Message{Line: []byte("line"), Source: "src1"}
	jsonlog, err := (&jsonlog.JSONLog{Log: string(msg.Line) + "\n", Stream: msg.Source, Created: msg.Timestamp}).MarshalJSON()
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(jsonlog)+1) * 30)

	b.StartTimer()

	go func() {
		for i := 0; i < b.N; i++ {
			for j := 0; j < 30; j++ {
				l.Log(msg)
			}
		}
		l.Close()
	}()

	lw := l.(logger.LogReader).ReadLogs(logger.ReadConfig{Follow: true})
	watchClose := lw.WatchClose()
	for {
		select {
		case <-lw.Msg:
		case <-watchClose:
			return
		}
	}
}
