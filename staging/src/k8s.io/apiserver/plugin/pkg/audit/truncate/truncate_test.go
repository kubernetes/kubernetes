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

package truncate

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/runtime"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	auditv1 "k8s.io/apiserver/pkg/apis/audit/v1"
	"k8s.io/apiserver/plugin/pkg/audit/fake"
	// Importing just for the schema definitions.
	_ "k8s.io/apiserver/plugin/pkg/audit/webhook"
)

var (
	defaultConfig = Config{
		MaxBatchSize: 4 * 1024 * 1024,
		MaxEventSize: 100 * 1024,
	}
)

func TestTruncatingEvents(t *testing.T) {
	testCases := []struct {
		desc          string
		event         *auditinternal.Event
		wantDropped   bool
		wantTruncated bool
	}{
		{
			desc:  "Empty event should not be truncated",
			event: &auditinternal.Event{},
		},
		{
			desc: "Event with too large body should be truncated",
			event: &auditinternal.Event{
				Level: auditinternal.LevelRequest,
				RequestObject: &runtime.Unknown{
					Raw: []byte("\"" + strings.Repeat("A", int(defaultConfig.MaxEventSize)) + "\""),
				},
			},
			wantTruncated: true,
		},
		{
			desc: "Event with too large metadata should be dropped",
			event: &auditinternal.Event{
				Annotations: map[string]string{
					"key": strings.Repeat("A", int(defaultConfig.MaxEventSize)),
				},
			},
			wantDropped: true,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.desc, func(t *testing.T) {
			t.Parallel()

			var event *auditinternal.Event

			fb := &fake.Backend{
				OnRequest: func(events []*auditinternal.Event) {
					require.Len(t, events, 1, "Expected single event in batch")
					event = events[0]
				},
			}
			b := NewBackend(fb, defaultConfig, auditv1.SchemeGroupVersion)
			b.ProcessEvents(tc.event)

			require.Equal(t, !tc.wantDropped, event != nil, "Incorrect event presence")
			if tc.wantTruncated {
				require.Equal(t, annotationValue, event.Annotations[annotationKey], "Annotation should be present")
				require.Nil(t, event.RequestObject, "After truncation request should be nil")
				require.Nil(t, event.ResponseObject, "After truncation response should be nil")
			}
		})
	}
}

func TestSplittingBatches(t *testing.T) {
	testCases := []struct {
		desc           string
		config         Config
		events         []*auditinternal.Event
		wantBatchCount int
	}{
		{
			desc:           "Events fitting in one batch should not be split",
			config:         defaultConfig,
			events:         []*auditinternal.Event{{}},
			wantBatchCount: 1,
		},
		{
			desc: "Events not fitting in one batch should be split",
			config: Config{
				MaxEventSize: defaultConfig.MaxEventSize,
				MaxBatchSize: 1,
			},
			events: []*auditinternal.Event{
				{Annotations: map[string]string{"key": strings.Repeat("A", int(50))}},
				{Annotations: map[string]string{"key": strings.Repeat("A", int(50))}},
			},
			wantBatchCount: 2,
		},
	}
	for _, tc := range testCases {
		tc := tc
		t.Run(tc.desc, func(t *testing.T) {
			t.Parallel()

			var gotBatchCount int
			fb := &fake.Backend{
				OnRequest: func(events []*auditinternal.Event) {
					gotBatchCount++
				},
			}
			b := NewBackend(fb, tc.config, auditv1.SchemeGroupVersion)
			b.ProcessEvents(tc.events...)

			require.Equal(t, tc.wantBatchCount, gotBatchCount)
		})
	}
}
