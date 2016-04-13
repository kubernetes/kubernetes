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

package manager

import (
	"fmt"
	"strings"
	"time"

	"k8s.io/heapster/extpoints"
	sink_api "k8s.io/heapster/sinks/api"
)

func newSinks(sinkUris Uris, statsRes time.Duration) ([]sink_api.ExternalSink, error) {
	var sinks []sink_api.ExternalSink
	var errors []string
	for _, u := range sinkUris {
		factory := extpoints.SinkFactories.Lookup(u.Key)
		if factory == nil {
			return nil, fmt.Errorf("Unknown sink: %s", u.Key)
		}
		createdSinks, err := factory(&u.Val, extpoints.HeapsterConf{StatsResolution: statsRes})
		if err != nil {
			errors = append(errors, err.Error())
		}
		sinks = append(sinks, createdSinks...)
	}
	var err error
	if len(errors) > 0 {
		err = fmt.Errorf("encountered following errors while setting up sinks - %v", strings.Join(errors, "; "))
	}
	return sinks, err
}
