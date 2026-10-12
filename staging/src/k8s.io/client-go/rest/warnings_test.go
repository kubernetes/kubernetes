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

	"github.com/onsi/gomega"

	"k8s.io/ktesting"
)

func TestDefaultWarningHandler(t *testing.T) {
	tCtx := ktesting.Init(t)
	tCtx.Run("default", func(tCtx ktesting.TContext) {
		tCtx.Assert(getDefaultWarningHandler()).Should(gomega.BeAssignableToTypeOf(WarningHandlerWithContext(WarningLogger{})))
	})

	deferRestore := func(tCtx ktesting.TContext) {
		handler := getDefaultWarningHandler()
		tCtx.Cleanup(func() {
			SetDefaultWarningHandlerWithContext(handler)
		})
	}

	tCtx.Run("no-context", func(tCtx ktesting.TContext) {
		deferRestore(tCtx)
		handler := &fakeWarningHandlerWithLogging{}
		//nolint:logcheck
		SetDefaultWarningHandler(handler)
		getDefaultWarningHandler().HandleWarningHeaderWithContext(context.Background(), 0, "", "message")
		tCtx.Assert(handler.messages).Should(gomega.Equal([]string{"message"}))
		SetDefaultWarningHandler(nil)
		tCtx.Assert(getDefaultWarningHandler()).Should(gomega.BeNil())
	})

	tCtx.Run("with-context", func(tCtx ktesting.TContext) {
		deferRestore(tCtx)
		handler := &fakeWarningHandlerWithContext{}
		SetDefaultWarningHandlerWithContext(handler)
		tCtx.Assert(getDefaultWarningHandler()).Should(gomega.Equal(handler))
		SetDefaultWarningHandlerWithContext(nil)
		tCtx.Assert(getDefaultWarningHandler()).Should(gomega.BeNil())
	})
}
