// Copyright 2019, OpenCensus Authors
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

// Package runmetrics contains support for runtime metrics.
//
// To enable collecting runtime metrics, just call Enable():
//
//     _ := runmetrics.Enable(runmetrics.RunMetricOptions{
//         EnableCPU: true,
//         EnableMemory: true,
//     })
package runmetrics // import "go.opencensus.io/plugin/runmetrics"
