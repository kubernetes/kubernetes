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
// limitations untader the License.

package validate

import (
	re "regexp"
	"sync"
)

// Cache for compiled regular expressions
var (
	cacheMutex = &sync.Mutex{}
	reDict     = map[string]*re.Regexp{}
)

func compileRegexp(pattern string) (*re.Regexp, error) {
	// Save repeated regexp compilation
	if reDict[pattern] != nil {
		return reDict[pattern], nil
	}
	var err error
	cacheMutex.Lock()
	reDict[pattern], err = re.Compile(pattern)
	cacheMutex.Unlock()
	return reDict[pattern], err
}

func mustCompileRegexp(pattern string) *re.Regexp {
	// Save repeated regexp compilation, with panic on error
	if reDict[pattern] != nil {
		return reDict[pattern]
	}
	defer cacheMutex.Unlock()
	cacheMutex.Lock()
	reDict[pattern] = re.MustCompile(pattern)
	return reDict[pattern]
}
