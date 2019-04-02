/*
Copyright 2019 The Kubernetes Authors.

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

package util

import (
	"sort"

	apps "k8s.io/api/apps/v1"
)

// HistoriesByRevision conforms sort.Interface and sorts controller revisions
// in ascending order.
type HistoriesByRevision []*apps.ControllerRevision

var _ sort.Interface = &HistoriesByRevision{}

func (h HistoriesByRevision) Len() int      { return len(h) }
func (h HistoriesByRevision) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h HistoriesByRevision) Less(i, j int) bool {
	return h[i].Revision < h[j].Revision
}
