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

package common

// TODO
// Get the base image type and (optionally) install Nvidia drivers for COS base image
// Verify that GPUs are available across nodes in a reasonable amount of time.
// Start as many pods as there are GPUs that runs gcr.io/google-containers/cuda-vector-add:v0.1 container to verify GPU integration.
// Ensure that the pods complete successfully
