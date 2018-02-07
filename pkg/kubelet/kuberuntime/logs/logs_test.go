/*
Copyright 2017 The Kubernetes Authors.

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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
)

func TestLogOptions(t *testing.T) {
	var (
		line         = int64(8)
		bytes        = int64(64)
		timestamp    = metav1.Now()
		sinceseconds = int64(10)
	)
	for c, test := range []struct {
		apiOpts *v1.PodLogOptions
		expect  *LogOptions
	}{
		{ // empty options
			apiOpts: &v1.PodLogOptions{},
			expect:  &LogOptions{tail: -1, bytes: -1},
		},
		{ // test tail lines
			apiOpts: &v1.PodLogOptions{TailLines: &line},
			expect:  &LogOptions{tail: line, bytes: -1},
		},
		{ // test limit bytes
			apiOpts: &v1.PodLogOptions{LimitBytes: &bytes},
			expect:  &LogOptions{tail: -1, bytes: bytes},
		},
		{ // test since timestamp
			apiOpts: &v1.PodLogOptions{SinceTime: &timestamp},
			expect:  &LogOptions{tail: -1, bytes: -1, since: timestamp.Time},
		},
		{ // test since seconds
			apiOpts: &v1.PodLogOptions{SinceSeconds: &sinceseconds},
			expect:  &LogOptions{tail: -1, bytes: -1, since: timestamp.Add(-10 * time.Second)},
		},
	} {
		t.Logf("TestCase #%d: %+v", c, test)
		opts := NewLogOptions(test.apiOpts, timestamp.Time)
		assert.Equal(t, test.expect, opts)
	}
}

func TestParseLog(t *testing.T) {
	timestamp, err := time.Parse(timeFormat, "2016-10-20T18:39:20.57606443Z")
	assert.NoError(t, err)
	msg := &logMessage{}
	for c, test := range []struct {
		line string
		msg  *logMessage
		err  bool
	}{
		{ // Docker log format stdout
			line: `{"log":"docker stdout test log","stream":"stdout","time":"2016-10-20T18:39:20.57606443Z"}` + "\n",
			msg: &logMessage{
				timestamp: timestamp,
				stream:    runtimeapi.Stdout,
				log:       []byte("docker stdout test log"),
			},
		},
		{ // Docker log format stderr
			line: `{"log":"docker stderr test log","stream":"stderr","time":"2016-10-20T18:39:20.57606443Z"}` + "\n",
			msg: &logMessage{
				timestamp: timestamp,
				stream:    runtimeapi.Stderr,
				log:       []byte("docker stderr test log"),
			},
		},
		{ // CRI log format stdout
			line: "2016-10-20T18:39:20.57606443Z stdout F cri stdout test log\n",
			msg: &logMessage{
				timestamp: timestamp,
				stream:    runtimeapi.Stdout,
				log:       []byte("cri stdout test log\n"),
			},
		},
		{ // CRI log format stderr
			line: "2016-10-20T18:39:20.57606443Z stderr F cri stderr test log\n",
			msg: &logMessage{
				timestamp: timestamp,
				stream:    runtimeapi.Stderr,
				log:       []byte("cri stderr test log\n"),
			},
		},
		{ // Unsupported Log format
			line: "unsupported log format test log\n",
			msg:  &logMessage{},
			err:  true,
		},
		{ // Partial CRI log line
			line: "2016-10-20T18:39:20.57606443Z stdout P cri stdout partial test log\n",
			msg: &logMessage{
				timestamp: timestamp,
				stream:    runtimeapi.Stdout,
				log:       []byte("cri stdout partial test log"),
			},
		},
		{ // Partial CRI log line with multiple log tags.
			line: "2016-10-20T18:39:20.57606443Z stdout P:TAG1:TAG2 cri stdout partial test log\n",
			msg: &logMessage{
				timestamp: timestamp,
				stream:    runtimeapi.Stdout,
				log:       []byte("cri stdout partial test log"),
			},
		},
	} {
		t.Logf("TestCase #%d: %+v", c, test)
		parse, err := getParseFunc([]byte(test.line))
		if test.err {
			assert.Error(t, err)
			continue
		}
		assert.NoError(t, err)
		err = parse([]byte(test.line), msg)
		assert.NoError(t, err)
		assert.Equal(t, test.msg, msg)
	}
}

func TestWriteLogs(t *testing.T) {
	timestamp := time.Unix(1234, 4321)
	log := "abcdefg\n"

	for c, test := range []struct {
		stream       runtimeapi.LogStreamType
		since        time.Time
		timestamp    bool
		expectStdout string
		expectStderr string
	}{
		{ // stderr log
			stream:       runtimeapi.Stderr,
			expectStderr: log,
		},
		{ // stdout log
			stream:       runtimeapi.Stdout,
			expectStdout: log,
		},
		{ // since is after timestamp
			stream: runtimeapi.Stdout,
			since:  timestamp.Add(1 * time.Second),
		},
		{ // timestamp enabled
			stream:       runtimeapi.Stderr,
			timestamp:    true,
			expectStderr: timestamp.Format(timeFormat) + " " + log,
		},
	} {
		t.Logf("TestCase #%d: %+v", c, test)
		msg := &logMessage{
			timestamp: timestamp,
			stream:    test.stream,
			log:       []byte(log),
		}
		stdoutBuf := bytes.NewBuffer(nil)
		stderrBuf := bytes.NewBuffer(nil)
		w := newLogWriter(stdoutBuf, stderrBuf, &LogOptions{since: test.since, timestamp: test.timestamp, bytes: -1})
		err := w.write(msg)
		assert.NoError(t, err)
		assert.Equal(t, test.expectStdout, stdoutBuf.String())
		assert.Equal(t, test.expectStderr, stderrBuf.String())
	}
}

func TestWriteLogsWithBytesLimit(t *testing.T) {
	timestamp := time.Unix(1234, 4321)
	timestampStr := timestamp.Format(timeFormat)
	log := "abcdefg\n"

	for c, test := range []struct {
		stdoutLines  int
		stderrLines  int
		bytes        int
		timestamp    bool
		expectStdout string
		expectStderr string
	}{
		{ // limit bytes less than one line
			stdoutLines:  3,
			bytes:        3,
			expectStdout: "abc",
		},
		{ // limit bytes across lines
			stdoutLines:  3,
			bytes:        len(log) + 3,
			expectStdout: "abcdefg\nabc",
		},
		{ // limit bytes more than all lines
			stdoutLines:  3,
			bytes:        3 * len(log),
			expectStdout: "abcdefg\nabcdefg\nabcdefg\n",
		},
		{ // limit bytes for stderr
			stderrLines:  3,
			bytes:        len(log) + 3,
			expectStderr: "abcdefg\nabc",
		},
		{ // limit bytes for both stdout and stderr, stdout first.
			stdoutLines:  1,
			stderrLines:  2,
			bytes:        len(log) + 3,
			expectStdout: "abcdefg\n",
			expectStderr: "abc",
		},
		{ // limit bytes with timestamp
			stdoutLines:  3,
			timestamp:    true,
			bytes:        len(timestampStr) + 1 + len(log) + 2,
			expectStdout: timestampStr + " " + log + timestampStr[:2],
		},
	} {
		t.Logf("TestCase #%d: %+v", c, test)
		msg := &logMessage{
			timestamp: timestamp,
			log:       []byte(log),
		}
		stdoutBuf := bytes.NewBuffer(nil)
		stderrBuf := bytes.NewBuffer(nil)
		w := newLogWriter(stdoutBuf, stderrBuf, &LogOptions{timestamp: test.timestamp, bytes: int64(test.bytes)})
		for i := 0; i < test.stdoutLines; i++ {
			msg.stream = runtimeapi.Stdout
			if err := w.write(msg); err != nil {
				assert.EqualError(t, err, errMaximumWrite.Error())
			}
		}
		for i := 0; i < test.stderrLines; i++ {
			msg.stream = runtimeapi.Stderr
			if err := w.write(msg); err != nil {
				assert.EqualError(t, err, errMaximumWrite.Error())
			}
		}
		assert.Equal(t, test.expectStdout, stdoutBuf.String())
		assert.Equal(t, test.expectStderr, stderrBuf.String())
	}
}
