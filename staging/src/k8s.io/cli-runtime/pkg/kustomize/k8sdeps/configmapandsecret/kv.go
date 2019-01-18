/*
Copyright 2018 The Kubernetes Authors.

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

package configmapandsecret

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"strings"
	"unicode"
	"unicode/utf8"

	"k8s.io/apimachinery/pkg/util/validation"
)

// kvPair represents a key value pair.
type kvPair struct {
	key   string
	value string
}

var utf8bom = []byte{0xEF, 0xBB, 0xBF}

// keyValuesFromLines parses given content in to a list of key-value pairs.
func keyValuesFromLines(content []byte) ([]kvPair, error) {
	var kvs []kvPair

	scanner := bufio.NewScanner(bytes.NewReader(content))
	currentLine := 0
	for scanner.Scan() {
		// Process the current line, retrieving a key/value pair if
		// possible.
		scannedBytes := scanner.Bytes()
		kv, err := kvFromLine(scannedBytes, currentLine)
		if err != nil {
			return nil, err
		}
		currentLine++

		if len(kv.key) == 0 {
			// no key means line was empty or a comment
			continue
		}

		kvs = append(kvs, kv)
	}
	return kvs, nil
}

// kvFromLine returns a kv with blank key if the line is empty or a comment.
// The value will be retrieved from the environment if necessary.
func kvFromLine(line []byte, currentLine int) (kvPair, error) {
	kv := kvPair{}

	if !utf8.Valid(line) {
		return kv, fmt.Errorf("line %d has invalid utf8 bytes : %v", line, string(line))
	}

	// We trim UTF8 BOM from the first line of the file but no others
	if currentLine == 0 {
		line = bytes.TrimPrefix(line, utf8bom)
	}

	// trim the line from all leading whitespace first
	line = bytes.TrimLeftFunc(line, unicode.IsSpace)

	// If the line is empty or a comment, we return a blank key/value pair.
	if len(line) == 0 || line[0] == '#' {
		return kv, nil
	}

	data := strings.SplitN(string(line), "=", 2)
	key := data[0]
	if errs := validation.IsEnvVarName(key); len(errs) != 0 {
		return kv, fmt.Errorf("%q is not a valid key name: %s", key, strings.Join(errs, ";"))
	}

	if len(data) == 2 {
		kv.value = data[1]
	} else {
		// No value (no `=` in the line) is a signal to obtain the value
		// from the environment.
		kv.value = os.Getenv(key)
	}
	kv.key = key
	return kv, nil
}
