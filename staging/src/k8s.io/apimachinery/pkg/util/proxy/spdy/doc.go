/*
Copyright 2023 The Kubernetes Authors.

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

// This temporary package replicates SPDY client functionality for use in the
// StreamTranslator proxy in the parent "proxy" package. This isolated
// functionality is necessary to transition Kubernetes bi-directional streaming
// protocol from SPDY to the more modern WebSockets. This effort is described
// in the following KEP:
//
//	https://github.com/kubernetes/enhancements/issues/4006
//
// This package should not be imported anywhere except the parent "proxy" package.
package spdy
