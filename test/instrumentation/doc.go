/*
Copyright 2022 The Kubernetes Authors.

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

/*
This stand-alone package is utilized for dynamically generating/maintaining a list of
metrics; this list is determined by the stability class of the metric. We statically analyze
all files in the Kubernetes code base to:
  - Determine whether the metric falls into a stability class with stability guarantees.
  - Determine the metric's metadata, i.e. the name, labels, type of metric.
  - Output (based on the above) the metrics which meet our criteria into a yaml file.

Due to the dynamic nature of how metrics can be written, we only support the subset of metrics
which can actually be parsed. If a metric cannot be parsed, it must be delegated to the stability
class `Internal`, which will exempt the metric from static analysis.
*/
package main
