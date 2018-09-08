/*
Copyright 2018 The Kubernetes Authors.

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

// Package connrotation implements a connection dialer that tracks and can close
// all created connections.
//
// This is used for credential rotation of long-lived connections, when there's
// no way to re-authenticate on a live connection.
package connrotation // import "k8s.io/client-go/util/connrotation"
