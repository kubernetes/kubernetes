// Copyright © 2016 Steve Francia <spf@spf13.com>.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package jwalterweatherman

import (
	"bytes"
	"github.com/stretchr/testify/assert"
	"sync"
	"testing"
)

func TestLevels(t *testing.T) {
	SetStdoutThreshold(LevelError)
	assert.Equal(t, StdoutThreshold(), LevelError)
	SetLogThreshold(LevelCritical)
	assert.Equal(t, LogThreshold(), LevelCritical)
	assert.NotEqual(t, StdoutThreshold(), LevelCritical)
	SetStdoutThreshold(LevelWarn)
	assert.Equal(t, StdoutThreshold(), LevelWarn)
}

func TestDefaultLogging(t *testing.T) {
	outputBuf := new(bytes.Buffer)
	logBuf := new(bytes.Buffer)
	LogHandle = logBuf
	OutHandle = outputBuf

	SetLogThreshold(LevelWarn)
	SetStdoutThreshold(LevelError)

	FATAL.Println("fatal err")
	CRITICAL.Println("critical err")
	ERROR.Println("an error")
	WARN.Println("a warning")
	INFO.Println("information")
	DEBUG.Println("debugging info")
	TRACE.Println("trace")

	assert.Contains(t, logBuf.String(), "fatal err")
	assert.Contains(t, logBuf.String(), "critical err")
	assert.Contains(t, logBuf.String(), "an error")
	assert.Contains(t, logBuf.String(), "a warning")
	assert.NotContains(t, logBuf.String(), "information")
	assert.NotContains(t, logBuf.String(), "debugging info")
	assert.NotContains(t, logBuf.String(), "trace")

	assert.Contains(t, outputBuf.String(), "fatal err")
	assert.Contains(t, outputBuf.String(), "critical err")
	assert.Contains(t, outputBuf.String(), "an error")
	assert.NotContains(t, outputBuf.String(), "a warning")
	assert.NotContains(t, outputBuf.String(), "information")
	assert.NotContains(t, outputBuf.String(), "debugging info")
	assert.NotContains(t, outputBuf.String(), "trace")
}

func TestLogCounter(t *testing.T) {
	ResetLogCounters()

	FATAL.Println("fatal err")
	CRITICAL.Println("critical err")
	WARN.Println("a warning")
	WARN.Println("another warning")
	INFO.Println("information")
	DEBUG.Println("debugging info")
	TRACE.Println("trace")

	wg := &sync.WaitGroup{}

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				ERROR.Println("error", j)
				// check for data races
				assert.True(t, LogCountForLevel(LevelError) > uint64(j))
				assert.True(t, LogCountForLevelsGreaterThanorEqualTo(LevelError) > uint64(j))
			}
		}()

	}

	wg.Wait()

	assert.Equal(t, uint64(1), LogCountForLevel(LevelFatal))
	assert.Equal(t, uint64(1), LogCountForLevel(LevelCritical))
	assert.Equal(t, uint64(2), LogCountForLevel(LevelWarn))
	assert.Equal(t, uint64(1), LogCountForLevel(LevelInfo))
	assert.Equal(t, uint64(1), LogCountForLevel(LevelDebug))
	assert.Equal(t, uint64(1), LogCountForLevel(LevelTrace))
	assert.Equal(t, uint64(100), LogCountForLevel(LevelError))
	assert.Equal(t, uint64(102), LogCountForLevelsGreaterThanorEqualTo(LevelError))
}
