//go:build !windows
// +build !windows

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

package fake

import (
	"fmt"
	"k8s.io/apimachinery/pkg/util/rand"
)

const (
	defaultUnixEndpoint = "unix:///tmp/kubelet_remote_%v.sock"
)

// GenerateEndpoint generates a new unix socket server of grpc server.
func GenerateEndpoint() (string, error) {
	// use random int be a part fo file name
	return fmt.Sprintf(defaultUnixEndpoint, rand.Int()), nil
}
