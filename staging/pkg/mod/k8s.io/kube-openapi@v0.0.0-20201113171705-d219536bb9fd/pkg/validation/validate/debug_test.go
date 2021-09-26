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

package validate

import (
	"io/ioutil"
	"os"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
)

var (
	logMutex = &sync.Mutex{}
)

func TestDebug(t *testing.T) {
	tmpFile, _ := ioutil.TempFile("", "debug-test")
	tmpName := tmpFile.Name()
	defer func() {
		Debug = false
		// mutex for -race
		logMutex.Unlock()
		os.Remove(tmpName)
	}()

	// mutex for -race
	logMutex.Lock()
	Debug = true
	debugOptions()
	defer func() {
		validateLogger.SetOutput(os.Stdout)
	}()

	validateLogger.SetOutput(tmpFile)

	debugLog("A debug")
	Debug = false
	tmpFile.Close()

	flushed, _ := os.Open(tmpName)
	buf := make([]byte, 500)
	_, _ = flushed.Read(buf)
	validateLogger.SetOutput(os.Stdout)
	assert.Contains(t, string(buf), "A debug")
}
