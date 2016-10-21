/*
Copyright 2016 The Kubernetes Authors.

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

package kuberuntime

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

func TestLogOptions(t *testing.T) {
	var (
		line         = int64(8)
		timestamp    = unversioned.Now()
		sinceseconds = int64(10)
	)
	for c, test := range []struct {
		apiOpts *api.PodLogOptions
		expect  *logOptions
	}{
		{ // empty options
			apiOpts: &api.PodLogOptions{},
			expect:  &logOptions{tail: -1},
		},
		{ // test tail lines
			apiOpts: &api.PodLogOptions{TailLines: &line},
			expect:  &logOptions{tail: line},
		},
		{ // test since timestamp
			apiOpts: &api.PodLogOptions{SinceTime: &timestamp},
			expect:  &logOptions{tail: -1, since: timestamp.Time},
		},
		{ // test since seconds
			apiOpts: &api.PodLogOptions{SinceSeconds: &sinceseconds},
			expect:  &logOptions{tail: -1, since: timestamp.Add(-10 * time.Second)},
		},
	} {
		t.Logf("TestCase #%d: %+v", c, test)
		opts := newLogOptions(test.apiOpts, timestamp.Time)
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
				stream:    stdoutType,
				log:       []byte("docker stdout test log"),
			},
		},
		{ // Docker log format stderr
			line: `{"log":"docker stderr test log","stream":"stderr","time":"2016-10-20T18:39:20.57606443Z"}` + "\n",
			msg: &logMessage{
				timestamp: timestamp,
				stream:    stderrType,
				log:       []byte("docker stderr test log"),
			},
		},
		{ // CRI log format stdout
			line: "2016-10-20T18:39:20.57606443Z stdout cri stdout test log\n",
			msg: &logMessage{
				timestamp: timestamp,
				stream:    stdoutType,
				log:       []byte("cri stdout test log\n"),
			},
		},
		{ // CRI log format stderr
			line: "2016-10-20T18:39:20.57606443Z stderr cri stderr test log\n",
			msg: &logMessage{
				timestamp: timestamp,
				stream:    stderrType,
				log:       []byte("cri stderr test log\n"),
			},
		},
		{ // Unsupported Log format
			line: "unsupported log format test log\n",
			msg:  &logMessage{},
			err:  true,
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
		stream       streamType
		since        time.Time
		timestamp    bool
		expectStdout string
		expectStderr string
	}{
		{ // stderr log
			stream:       stderrType,
			expectStderr: log,
		},
		{ // stdout log
			stream:       stdoutType,
			expectStdout: log,
		},
		{ // since is after timestamp
			stream: stdoutType,
			since:  timestamp.Add(1 * time.Second),
		},
		{ // timestamp enabled
			stream:       stderrType,
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
		err := writeLogs(msg, stdoutBuf, stderrBuf, &logOptions{since: test.since, timestamp: test.timestamp})
		assert.NoError(t, err)
		assert.Equal(t, test.expectStdout, stdoutBuf.String())
		assert.Equal(t, test.expectStderr, stderrBuf.String())
	}
}

func TestTail(t *testing.T) {
	line := strings.Repeat("a", blockSize)
	testBytes := []byte(line + "\n" +
		line + "\n" +
		line + "\n" +
		line + "\n" +
		line[blockSize/2:]) // incomplete line

	for c, test := range []struct {
		n     int64
		start int64
	}{
		{n: -1, start: 0},
		{n: 0, start: int64(len(line)+1) * 4},
		{n: 1, start: int64(len(line)+1) * 3},
		{n: 9999, start: 0},
	} {
		t.Logf("TestCase #%d: %+v", c, test)
		r := bytes.NewReader(testBytes)
		s, err := tail(r, test.n)
		assert.NoError(t, err)
		assert.Equal(t, s, test.start)
	}
}
