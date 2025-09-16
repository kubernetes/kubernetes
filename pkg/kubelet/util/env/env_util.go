/*
Copyright 2025 The Kubernetes Authors.

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

package env

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// ParseEnv implements a strict parser for .env environment files,
// adhering to the format defined in the RFC documentation at https://smartmob-rfc.readthedocs.io/en/latest/2-dotenv.html.
//
// This function implements a strict parser for environment files similar to the requirements in the OCI and Docker env file RFCs:
//   - Leading whitespace is ignored for all lines.
//   - Blank lines (including those with only whitespace) are ignored.
//   - Lines starting with '#' are treated as comments and ignored.
//   - Each variable must be declared as VAR=VAL. Whitespace around '=' and at the end of the line is ignored.
//   - A backslash ('\') at the end of a variable declaration line indicates the value continues on the next line. The lines are joined with a single space, and the backslash is not included.
//   - If a continuation line is interrupted by a blank line or comment, it is considered invalid and an error is returned.
func ParseEnv(envFilePath, key string) (string, error) {
	file, err := os.Open(envFilePath)
	if err != nil {
		return "", fmt.Errorf("failed to open environment variable file %q: %w", envFilePath, err)
	}
	defer func() { _ = file.Close() }()

	scanner := bufio.NewScanner(file)
	var (
		currentLine    string
		inContinuation bool
		lineNum        int
	)

	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		line = strings.TrimLeft(line, " \t")

		if line == "" {
			if inContinuation {
				return "", fmt.Errorf("invalid environment variable format at line %d: blank line in continuation", lineNum)
			}
			continue
		}
		if strings.HasPrefix(line, "#") {
			if inContinuation {
				return "", fmt.Errorf("invalid environment variable format at line %d: comment in continuation", lineNum)
			}
			continue
		}

		if inContinuation {
			trimmed := strings.TrimRight(line, " \t")
			if strings.HasSuffix(trimmed, "\\") {
				currentLine += " " + strings.TrimRight(trimmed[:len(trimmed)-1], " \t")
				continue
			} else {
				currentLine += " " + trimmed
				line = currentLine
				inContinuation = false
				currentLine = ""
			}
		} else {
			trimmed := strings.TrimRight(line, " \t")
			if strings.HasSuffix(trimmed, "\\") {
				currentLine = strings.TrimRight(trimmed[:len(trimmed)-1], " \t")
				inContinuation = true
				continue
			}
		}

		eqIdx := strings.Index(line, "=")
		if eqIdx == -1 {
			return "", fmt.Errorf("invalid environment variable format at line %d", lineNum)
		}
		varName := strings.TrimSpace(line[:eqIdx])
		varValue := strings.TrimRight(strings.TrimSpace(line[eqIdx+1:]), " \t")

		if varName == "" {
			return "", fmt.Errorf("invalid environment variable format at line %d", lineNum)
		}
		if varName == key {
			return varValue, nil
		}
	}

	if inContinuation {
		return "", fmt.Errorf("unexpected end of file: unfinished line continuation")
	}

	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("error reading environment variable file %q: %w", envFilePath, err)
	}

	return "", nil
}
