// Copyright 2019 Google Inc. All Rights Reserved.
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

//go:build linux

// The install package registers crio.NewPlugin() as the "crio" container provider when imported
package install

import (
	"k8s.io/klog/v2"

	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/crio"
)

func init() {
	err := container.RegisterPlugin("crio", crio.NewPlugin())
	if err != nil {
		klog.Fatalf("Failed to register crio plugin: %v", err)
	}
}
