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

// ParseEnv searches for a specific key in a given environment variable file
// and returns its corresponding value.
func ParseEnv(envFilePath, key string) (string, error) {
	// Read and parse the environment variable file
	file, err := os.Open(envFilePath)
	if err != nil {
		return "", fmt.Errorf("failed to open environment variable file %q: %w", envFilePath, err)
	}
	defer func() {
		_ = file.Close()
	}()

	// Parse the environment variable file to find the specified key
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		envSpec := scanner.Text()
		if pos := strings.Index(envSpec, "#"); pos != -1 {
			envSpec = envSpec[:pos]
		}

		if strings.Contains(envSpec, "=") {
			parts := strings.SplitN(envSpec, "=", 2)
			if len(parts) != 2 {
				return "", fmt.Errorf("invalid environment variable format: %s", envSpec)
			}
			fileKey := parts[0]
			fileValue := parts[1]
			if fileKey == key {
				return fileValue, nil
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("error reading environment variable file %q: %w", envFilePath, err)
	}

	return "", nil
}
