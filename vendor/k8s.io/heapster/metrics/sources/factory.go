// Copyright 2015 Google Inc. All Rights Reserved.
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

package sources

import (
	"fmt"

	"k8s.io/heapster/common/flags"
	"k8s.io/heapster/metrics/core"
	"k8s.io/heapster/metrics/sources/kubelet"
	"k8s.io/heapster/metrics/sources/summary"
)

type SourceFactory struct {
}

func (this *SourceFactory) Build(uri flags.Uri) (core.MetricsSourceProvider, error) {
	switch uri.Key {
	case "kubernetes":
		provider, err := kubelet.NewKubeletProvider(&uri.Val)
		return provider, err
	case "kubernetes.summary_api":
		provider, err := summary.NewSummaryProvider(&uri.Val)
		return provider, err
	default:
		return nil, fmt.Errorf("Source not recognized: %s", uri.Key)
	}
}

func (this *SourceFactory) BuildAll(uris flags.Uris) (core.MetricsSourceProvider, error) {
	if len(uris) != 1 {
		return nil, fmt.Errorf("Only one source is supported")
	}
	return this.Build(uris[0])
}

func NewSourceFactory() *SourceFactory {
	return &SourceFactory{}
}
