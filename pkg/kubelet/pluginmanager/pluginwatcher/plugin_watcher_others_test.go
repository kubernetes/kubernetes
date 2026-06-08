//go:build !windows

/*
Copyright 2023 The Kubernetes Authors.

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

package pluginwatcher

import (
	"os"
	"testing"

	"github.com/fsnotify/fsnotify"
	"github.com/stretchr/testify/assert"
)

func TestGetStat(t *testing.T) {
	event := fsnotify.Event{Name: "name", Op: fsnotify.Create}
	fi, err := getStat(event)
	fiExpected, errExpected := os.Stat(event.Name)

	assert.Equal(t, fiExpected, fi)
	assert.Equal(t, errExpected, err)
}

func TestGetSocketPath(t *testing.T) {
	socketPath := "/tmp/foo/lish.sock"
	assert.Equal(t, socketPath, getSocketPath(socketPath))
}
