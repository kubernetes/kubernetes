// Copyright 2022 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package prometheus

type v2 struct{}

// V2 is a struct that can be referenced to access experimental API that might
// be present in v2 of client golang someday. It offers extended functionality
// of v1 with slightly changed API. It is acceptable to use some pieces from v1
// and e.g `prometheus.NewGauge` and some from v2 e.g. `prometheus.V2.NewDesc`
// in the same codebase.
var V2 = v2{}
