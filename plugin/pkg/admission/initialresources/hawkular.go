/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package initialresources

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
)

type hawkularSource struct{}

func newHawkularSource() (dataSource, error) {
	return nil, fmt.Errorf("hawkular source not implemented")
}

func (s *hawkularSource) GetUsagePercentile(kind api.ResourceName, perc int64, image, namespace string, exactMatch bool, start, end time.Time) (int64, int64, error) {
	return 0, 0, fmt.Errorf("gcm source not implemented")
}
