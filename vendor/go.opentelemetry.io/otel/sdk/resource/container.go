// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package resource // import "go.opentelemetry.io/otel/sdk/resource"

import (
	"bufio"
	"context"
	"errors"
	"io"
	"os"
	"regexp"

	semconv "go.opentelemetry.io/otel/semconv/v1.12.0"
)

type containerIDProvider func() (string, error)

var (
	containerID         containerIDProvider = getContainerIDFromCGroup
	cgroupContainerIDRe                     = regexp.MustCompile(`^.*/(?:.*-)?([0-9a-f]+)(?:\.|\s*$)`)
)

type cgroupContainerIDDetector struct{}

const cgroupPath = "/proc/self/cgroup"

// Detect returns a *Resource that describes the id of the container.
// If no container id found, an empty resource will be returned.
func (cgroupContainerIDDetector) Detect(ctx context.Context) (*Resource, error) {
	containerID, err := containerID()
	if err != nil {
		return nil, err
	}

	if containerID == "" {
		return Empty(), nil
	}
	return NewWithAttributes(semconv.SchemaURL, semconv.ContainerIDKey.String(containerID)), nil
}

var (
	defaultOSStat = os.Stat
	osStat        = defaultOSStat

	defaultOSOpen = func(name string) (io.ReadCloser, error) {
		return os.Open(name)
	}
	osOpen = defaultOSOpen
)

// getContainerIDFromCGroup returns the id of the container from the cgroup file.
// If no container id found, an empty string will be returned.
func getContainerIDFromCGroup() (string, error) {
	if _, err := osStat(cgroupPath); errors.Is(err, os.ErrNotExist) {
		// File does not exist, skip
		return "", nil
	}

	file, err := osOpen(cgroupPath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	return getContainerIDFromReader(file), nil
}

// getContainerIDFromReader returns the id of the container from reader.
func getContainerIDFromReader(reader io.Reader) string {
	scanner := bufio.NewScanner(reader)
	for scanner.Scan() {
		line := scanner.Text()

		if id := getContainerIDFromLine(line); id != "" {
			return id
		}
	}
	return ""
}

// getContainerIDFromLine returns the id of the container from one string line.
func getContainerIDFromLine(line string) string {
	matches := cgroupContainerIDRe.FindStringSubmatch(line)
	if len(matches) <= 1 {
		return ""
	}
	return matches[1]
}
