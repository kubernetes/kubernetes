/*
Copyright 2024 The Kubernetes Authors.

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

package rest

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDefaultWarningHandler(t *testing.T) {
	t.Run("default", func(t *testing.T) {
		assert.IsType(t, WarningHandlerWithContext(WarningLogger{}), getDefaultWarningHandler())
	})

	deferRestore := func(t *testing.T) {
		handler := getDefaultWarningHandler()
		t.Cleanup(func() {
			SetDefaultWarningHandlerWithContext(handler)
		})
	}

	t.Run("no-context", func(t *testing.T) {
		deferRestore(t)
		handler := &fakeWarningHandlerWithLogging{}
		//nolint:logcheck
		SetDefaultWarningHandler(handler)
		getDefaultWarningHandler().HandleWarningHeaderWithContext(context.Background(), 0, "", "message")
		assert.Equal(t, []string{"message"}, handler.messages)
		SetDefaultWarningHandler(nil)
		assert.Nil(t, getDefaultWarningHandler())
	})

	t.Run("with-context", func(t *testing.T) {
		deferRestore(t)
		handler := &fakeWarningHandlerWithContext{}
		SetDefaultWarningHandlerWithContext(handler)
		assert.Equal(t, handler, getDefaultWarningHandler())
		SetDefaultWarningHandlerWithContext(nil)
		assert.Nil(t, getDefaultWarningHandler())
	})
}
