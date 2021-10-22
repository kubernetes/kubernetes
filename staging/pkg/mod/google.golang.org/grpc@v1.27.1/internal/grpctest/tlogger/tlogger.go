/*
 *
 * Copyright 2020 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package tlogger initializes the testing logger on import which logs to the
// testing package's T struct.
package tlogger

import (
	"errors"
	"fmt"
	"os"
	"runtime/debug"
	"strconv"
	"strings"
	"testing"

	"google.golang.org/grpc/grpclog"
)

var logger = tLogger{v: 0}

const callingFrame = 4

type tLogger struct {
	v int
	t *testing.T
}

func init() {
	vLevel := os.Getenv("GRPC_GO_LOG_VERBOSITY_LEVEL")
	if vl, err := strconv.Atoi(vLevel); err == nil {
		logger.v = vl
	}
	grpclog.SetLoggerV2(&logger)
}

func getStackFrame(stack []byte, frame int) (string, error) {
	s := strings.Split(string(stack), "\n")
	if frame >= (len(s)-1)/2 {
		return "", errors.New("frame request out-of-bounds")
	}
	split := strings.Split(strings.Fields(s[(frame*2)+2][1:])[0], "/")
	return fmt.Sprintf("%v:", split[len(split)-1]), nil
}

func log(t *testing.T, format string, fatal bool, args ...interface{}) {
	s := debug.Stack()
	prefix, err := getStackFrame(s, callingFrame)
	args = append([]interface{}{prefix}, args...)
	if err != nil {
		t.Error(err)
		return
	}
	if format == "" {
		if fatal {
			panic(fmt.Sprint(args...))
		} else {
			t.Log(args...)
		}
	} else {
		if fatal {
			panic(fmt.Sprintf("%v "+format, args...))
		} else {
			t.Logf("%v "+format, args...)
		}
	}
}

// Update updates the testing.T that the testing logger logs to. Should be done
// before every test.
func Update(t *testing.T) {
	logger.t = t
}

func (g *tLogger) Info(args ...interface{}) {
	log(g.t, "", false, args...)
}

func (g *tLogger) Infoln(args ...interface{}) {
	log(g.t, "", false, args...)
}

func (g *tLogger) Infof(format string, args ...interface{}) {
	log(g.t, format, false, args...)
}

func (g *tLogger) Warning(args ...interface{}) {
	log(g.t, "", false, args...)
}

func (g *tLogger) Warningln(args ...interface{}) {
	log(g.t, "", false, args...)
}

func (g *tLogger) Warningf(format string, args ...interface{}) {
	log(g.t, format, false, args...)
}

func (g *tLogger) Error(args ...interface{}) {
	log(g.t, "", false, args...)
}

func (g *tLogger) Errorln(args ...interface{}) {
	log(g.t, "", false, args...)
}

func (g *tLogger) Errorf(format string, args ...interface{}) {
	log(g.t, format, false, args...)
}

func (g *tLogger) Fatal(args ...interface{}) {
	log(g.t, "", true, args...)
}

func (g *tLogger) Fatalln(args ...interface{}) {
	log(g.t, "", true, args...)
}

func (g *tLogger) Fatalf(format string, args ...interface{}) {
	log(g.t, format, true, args...)
}

func (g *tLogger) V(l int) bool {
	return l <= g.v
}
