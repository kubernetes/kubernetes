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

// Package consumer provides helper functions for consuming EndpointSlices.
//
// The EndpointSliceConsumer helps applications transition from Endpoints to
// EndpointSlices by providing a unified view of all endpoints for a service
// across multiple EndpointSlice objects. It handles the complexity of tracking,
// merging, and deduplicating endpoints from multiple slices.
package consumer
