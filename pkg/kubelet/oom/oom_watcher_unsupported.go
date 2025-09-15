//go:build !linux
// +build !linux

/*
Copyright 2019 The Kubernetes Authors.

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

package oom

import (
	"context"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
)

type oomWatcherUnsupported struct{}

var _ Watcher = new(oomWatcherUnsupported)

// NewWatcher creates a fake one here
func NewWatcher(_ record.EventRecorder) (Watcher, error) {
	return &oomWatcherUnsupported{}, nil
}

func (ow *oomWatcherUnsupported) Start(_ context.Context, _ *v1.ObjectReference) error {
	return nil
}
