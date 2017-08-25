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

package executor

import (
	"context"
	"io"
)

// Unit represents a single task.
type Unit struct {
	// Name is the name of the task.
	Name string
	// Deps is a slice of the names of tasks that this task depends on
	Deps []string

	// Action is that action to preform.
	Action func(ctx context.Context, out io.Writer) error
}
