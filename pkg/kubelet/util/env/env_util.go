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

// ParseEnv implements a strict parser for environment files using a subset of POSIX shell syntax.
// The parser enforces that all variable values must be enclosed in single quotes.
//
// Supported format:
//   - VAR='value' - Values must be enclosed in single quotes
//   - Content within single quotes is preserved literally (no escape sequences or expansions)
//   - Multi-line values are supported (newlines within single quotes are preserved)
//   - Inline comments after the closing quote are supported (e.g., VAR='value' # comment)
//   - Leading whitespace before the variable name is ignored
//   - Blank lines (including those with only whitespace) are ignored when not within quotes
//   - Lines starting with '#' are treated as comments and ignored
//   - Whitespace before '=' is invalid (e.g., VAR = 'value' is rejected)
//   - Whitespace after '=' but before the quote results in empty assignment (e.g., VAR= 'value' assigns empty string)
func ParseEnv(envFilePath, key string) (string, error) {
	file, err := os.Open(envFilePath)
	if err != nil {
		return "", fmt.Errorf("failed to open environment variable file %q: %w", envFilePath, err)
	}
	defer func() { _ = file.Close() }()

	scanner := bufio.NewScanner(file)
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		line = strings.TrimLeft(line, " \t")

		// Skip blank lines
		if line == "" {
			continue
		}

		// Skip comments
		if strings.HasPrefix(line, "#") {
			continue
		}

		eqIdx := strings.Index(line, "=")
		if eqIdx == -1 {
			return "", fmt.Errorf("invalid environment variable format at line %d: missing '='", lineNum)
		}

		// Variable name must not contain whitespace or trailing whitespace before '='
		varNamePart := line[:eqIdx]
		varName := strings.TrimRight(varNamePart, " \t")
		if varName == "" {
			return "", fmt.Errorf("invalid environment variable format at line %d: empty variable name", lineNum)
		}

		// If trimming removed whitespace, it means there was whitespace before '='
		if varNamePart != varName {
			return "", fmt.Errorf("invalid environment variable format at line %d: whitespace before '=' is not allowed", lineNum)
		}
		valuePart := line[eqIdx+1:]

		// Check if there's whitespace before any non-whitespace character
		trimmedValue := strings.TrimLeft(valuePart, " \t")
		if valuePart != trimmedValue {
			// There is whitespace between '=' and the value
			// This matches bash behavior: KEY= 'val1' results in KEY being empty
			if varName == key {
				return "", nil
			}
			continue
		}

		// Value must start with single quote
		if !strings.HasPrefix(trimmedValue, "'") {
			return "", fmt.Errorf("invalid environment variable format at line %d: value must be enclosed in single quotes", lineNum)
		}

		// Find the closing single quote (may span multiple lines)
		var valueBuilder strings.Builder
		rest := trimmedValue[1:]
		startLineNum := lineNum

		for {
			closingIdx := strings.Index(rest, "'")
			if closingIdx != -1 {
				valueBuilder.WriteString(rest[:closingIdx])
				afterQuote := strings.TrimLeft(rest[closingIdx+1:], " \t")
				if afterQuote != "" && !strings.HasPrefix(afterQuote, "#") {
					return "", fmt.Errorf("invalid environment variable format at line %d: unexpected content after closing quote", lineNum)
				}

				if varName == key {
					return valueBuilder.String(), nil
				}
				break
			}

			valueBuilder.WriteString(rest)
			valueBuilder.WriteString("\n")
			if !scanner.Scan() {
				return "", fmt.Errorf("invalid environment variable format starting at line %d: unclosed single quote", startLineNum)
			}
			lineNum++
			rest = scanner.Text()
		}
	}

	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("error reading environment variable file %q: %w", envFilePath, err)
	}

	return "", nil
}
