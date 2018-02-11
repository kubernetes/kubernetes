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

package kubectl

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

var utf8bom = []byte{0xEF, 0xBB, 0xBF}

// proccessEnvFileLine returns a blank key if the line is empty or a comment.
// The value will be retrieved from the environment if necessary.
func proccessEnvFileLine(line []byte, filePath string,
	currentLine int) (key, value string, err error) {

	if !utf8.Valid(line) {
		return ``, ``, fmt.Errorf("env file %s contains invalid utf8 bytes at line %d: %v",
			filePath, currentLine+1, line)
	}

	// We trim UTF8 BOM from the first line of the file but no others
	if currentLine == 0 {
		line = bytes.TrimPrefix(line, utf8bom)
	}

	// trim the line from all leading whitespace first
	line = bytes.TrimLeftFunc(line, unicode.IsSpace)

	// If the line is empty or a comment, we return a blank key/value pair.
	if len(line) == 0 || line[0] == '#' {
		return ``, ``, nil
	}

	data := strings.SplitN(string(line), "=", 2)
	key = data[0]
	if errs := validation.IsEnvVarName(key); len(errs) != 0 {
		return ``, ``, fmt.Errorf("%q is not a valid key name: %s", key, strings.Join(errs, ";"))
	}

	if len(data) == 2 {
		value = data[1]
	} else {
		// No value (no `=` in the line) is a signal to obtain the value
		// from the environment.
		value = os.Getenv(key)
	}
	return
}

// addFromEnvFile processes an env file allows a generic addTo to handle the
// collection of key value pairs or returns an error.
func addFromEnvFile(filePath string, addTo func(key, value string) error) error {
	f, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	currentLine := 0
	for scanner.Scan() {
		// Process the current line, retrieving a key/value pair if
		// possible.
		scannedBytes := scanner.Bytes()
		key, value, err := proccessEnvFileLine(scannedBytes, filePath, currentLine)
		if err != nil {
			return err
		}
		currentLine++

		if len(key) == 0 {
			// no key means line was empty or a comment
			continue
		}

		if err = addTo(key, value); err != nil {
			return err
		}
	}
	return nil
}
