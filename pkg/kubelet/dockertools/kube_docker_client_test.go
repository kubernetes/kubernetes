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

package dockertools

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIsContainerNotFoundError(t *testing.T) {
	// Expected error message from docker.
	containerNotFoundError := fmt.Errorf("Error response from daemon: No such container: 96e914f31579e44fe49b239266385330a9b2125abeb9254badd9fca74580c95a")
	otherError := fmt.Errorf("Error response from daemon: Other errors")

	assert.True(t, IsContainerNotFoundError(containerNotFoundError))
	assert.False(t, IsContainerNotFoundError(otherError))
}
