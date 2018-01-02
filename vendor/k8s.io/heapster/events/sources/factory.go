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

	"github.com/golang/glog"

	"k8s.io/heapster/common/flags"
	"k8s.io/heapster/events/core"
	kube "k8s.io/heapster/events/sources/kubernetes"
)

type SourceFactory struct {
}

func (this *SourceFactory) Build(uri flags.Uri) (core.EventSource, error) {
	switch uri.Key {
	case "kubernetes":
		src, err := kube.NewKubernetesSource(&uri.Val)
		return src, err
	default:
		return nil, fmt.Errorf("Source not recognized: %s", uri.Key)
	}
}

func (this *SourceFactory) BuildAll(uris flags.Uris) ([]core.EventSource, error) {
	if len(uris) != 1 {
		return nil, fmt.Errorf("Only one source is supported")
	}
	result := []core.EventSource{}
	for _, uri := range uris {
		source, err := this.Build(uri)
		if err != nil {
			glog.Errorf("Failed to create %s: %v", uri.Key, err)
		} else {
			result = append(result, source)
		}
	}
	return result, nil
}

func NewSourceFactory() *SourceFactory {
	return &SourceFactory{}
}
