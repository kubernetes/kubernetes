/*
Copyright 2018 The Kubernetes Authors.

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

package policy

import (
	"fmt"

	"k8s.io/apiserver/pkg/apis/audit"
)

// EnforcePolicy drops any part of the event that doesn't conform to a policy level
// or omitStages and sets the event level accordingly
func EnforcePolicy(event *audit.Event, level audit.Level, omitStages []audit.Stage) (*audit.Event, error) {
	for _, stage := range omitStages {
		if event.Stage == stage {
			return nil, nil
		}
	}
	return enforceLevel(event, level)
}

func enforceLevel(event *audit.Event, level audit.Level) (*audit.Event, error) {
	switch level {
	case audit.LevelMetadata:
		event.Level = audit.LevelMetadata
		event.ResponseObject = nil
		event.RequestObject = nil
	case audit.LevelRequest:
		event.Level = audit.LevelRequest
		event.ResponseObject = nil
	case audit.LevelRequestResponse:
		event.Level = audit.LevelRequestResponse
	case audit.LevelNone:
		return nil, nil
	default:
		return nil, fmt.Errorf("level unknown: %s", level)
	}
	return event, nil
}
