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

package kuberuntime

import (
	"fmt"
	"math/rand"
	"path"
	"testing"

	"github.com/stretchr/testify/assert"
)

const letterBytes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

func randStringBytes(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = letterBytes[rand.Intn(len(letterBytes))]
	}
	return string(b)
}

func TestLogSymLink(t *testing.T) {
	as := assert.New(t)
	containerLogsDir := "/foo/bar"
	podFullName := randStringBytes(128)
	containerName := randStringBytes(70)
	dockerId := randStringBytes(80)
	// The file name cannot exceed 255 characters. Since .log suffix is required, the prefix cannot exceed 251 characters.
	expectedPath := path.Join(containerLogsDir, fmt.Sprintf("%s_%s-%s", podFullName, containerName, dockerId)[:251]+".log")
	as.Equal(expectedPath, logSymlink(containerLogsDir, podFullName, containerName, dockerId))
}
