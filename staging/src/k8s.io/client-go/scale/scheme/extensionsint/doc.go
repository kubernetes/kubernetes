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

// Package extensionsint contains the necessary scaffolding of the
// internal version of extensions as required by conversion logic.
// It doesn't have any of its own types -- it's just necessary to
// get the expected behavoir out of runtime.Scheme.ConvertToVersion
// and associated methods.
package extensionsint
