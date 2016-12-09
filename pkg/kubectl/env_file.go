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
	utf8bom := []byte{0xEF, 0xBB, 0xBF}
	for scanner.Scan() {
		scannedBytes := scanner.Bytes()
		if !utf8.Valid(scannedBytes) {
			return fmt.Errorf("env file %s contains invalid utf8 bytes at line %d: %v", filePath, currentLine+1, scannedBytes)
		}
		// We trim UTF8 BOM
		if currentLine == 0 {
			scannedBytes = bytes.TrimPrefix(scannedBytes, utf8bom)
		}
		// trim the line from all leading whitespace first
		line := strings.TrimLeftFunc(string(scannedBytes), unicode.IsSpace)
		currentLine++
		// line is not empty, and not starting with '#'
		if len(line) > 0 && !strings.HasPrefix(line, "#") {
			data := strings.SplitN(line, "=", 2)
			key := data[0]
			if errs := validation.IsCIdentifier(key); len(errs) != 0 {
				return fmt.Errorf("%q is not a valid key name: %s", key, strings.Join(errs, ";"))
			}

			value := ""
			if len(data) > 1 {
				// pass the value through, no trimming
				value = data[1]
			} else {
				// a pass-through variable is given
				value = os.Getenv(key)
			}
			if err = addTo(key, value); err != nil {
				return err
			}
		}
	}
	return nil
}
