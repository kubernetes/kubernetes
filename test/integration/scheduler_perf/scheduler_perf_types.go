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

package benchmark

// High Level Configuration for all predicates and priorities.
type schedulerPerfConfig struct {
	NodeCount    int // The number of nodes which will be seeded with metadata to match predicates and have non-trivial priority rankings.
	PodCount     int // The number of pods which will be seeded with metadata to match predicates and have non-trivial priority rankings.
	NodeAffinity *nodeAffinity
	// TODO: Other predicates and priorities to be added here.
}

// nodeAffinity priority configuration details.
type nodeAffinity struct {
	nodeAffinityKey string // Node Selection Key.
	LabelCount      int    // number of labels to be added to each node or pod.
}
