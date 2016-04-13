// Copyright 2015 Google Inc. All Rights Reserved.
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

package util

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestFailedInitializer(t *testing.T) {
	init := func() error {
		return fmt.Errorf("failed setup")
	}
	ping := func() error {
		return fmt.Errorf("failed ping")
	}
	cii := NewClientInitializer("test", init, ping, time.Millisecond)
	assert.False(t, cii.Done())
}

func TestSuccessfulInitializer(t *testing.T) {
	init := func() error {
		return nil
	}
	cii := NewClientInitializer("test", init, func() error { return nil }, time.Millisecond)
	assert.True(t, cii.Done())
}

func TestFailedPing(t *testing.T) {
	init := func() error {
		return nil
	}
	ping := func() error {
		return fmt.Errorf("failed ping")
	}
	cii := &clientInitializerImpl{
		name:        "test",
		initializer: init,
		ping:        ping,
	}
	cii.clientConfigured.Store(false)
	cii.setup()
	assert.True(t, cii.Done())
	cii.setup()
	assert.True(t, cii.Done())
	cii.initializer = func() error { return fmt.Errorf("setup failed") }
	cii.setup()
	assert.False(t, cii.Done())
}
