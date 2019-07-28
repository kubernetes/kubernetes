// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package analysis

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
)

var (
	// Debug is true when the SWAGGER_DEBUG env var is not empty.
	// It enables a more verbose logging of the spec analyzer.
	Debug = os.Getenv("SWAGGER_DEBUG") != ""
	// analysisLogger is a debug logger for this package
	analysisLogger *log.Logger
)

func init() {
	debugOptions()
}

func debugOptions() {
	analysisLogger = log.New(os.Stdout, "analysis:", log.LstdFlags)
}

func debugLog(msg string, args ...interface{}) {
	// A private, trivial trace logger, based on go-openapi/spec/expander.go:debugLog()
	if Debug {
		_, file1, pos1, _ := runtime.Caller(1)
		analysisLogger.Printf("%s:%d: %s", filepath.Base(file1), pos1, fmt.Sprintf(msg, args...))
	}
}
