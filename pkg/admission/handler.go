/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package admission

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// Handler is a base for admission control handlers that
// support a predefined set of operations
type Handler struct {
	operations util.StringSet
}

// Handles returns true for methods that this handler supports
func (h *Handler) Handles(operation Operation) bool {
	return h.operations.Has(string(operation))
}

// NewHandler creates a new base handler that handles the passed
// in operations
func NewHandler(ops ...Operation) *Handler {
	operations := util.NewStringSet()
	for _, op := range ops {
		operations.Insert(string(op))
	}
	return &Handler{
		operations: operations,
	}
}
