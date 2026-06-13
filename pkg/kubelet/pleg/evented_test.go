/*
Copyright 2015 The Kubernetes Authors.

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

package pleg

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/types"
	v1 "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
)

var _ podLifecycleEventGeneratorHandler = (*fakeGenericPLEG)(nil)

type fakeGenericPLEG struct {
	watchCh           chan *PodLifecycleEvent
	relistRequests    []types.UID
	reinspectRequests []types.UID
	relistCount       int
}

func newFakeGenericPLEG() *fakeGenericPLEG {
	return &fakeGenericPLEG{
		watchCh: make(chan *PodLifecycleEvent, 10),
	}
}

func (f *fakeGenericPLEG) Start(ctx context.Context) {}

func (f *fakeGenericPLEG) Watch() chan *PodLifecycleEvent {
	return f.watchCh
}

func (f *fakeGenericPLEG) Healthy() (bool, error) {
	return true, nil
}

func (f *fakeGenericPLEG) RequestReinspect(podUID types.UID) {
	f.reinspectRequests = append(f.reinspectRequests, podUID)
}

func (f *fakeGenericPLEG) RequestRelist(logger klog.Logger, podUID types.UID) {
	f.relistRequests = append(f.relistRequests, podUID)
}

func (f *fakeGenericPLEG) Stop() {}

func (f *fakeGenericPLEG) Update(*RelistDuration) {}

func (f *fakeGenericPLEG) Relist(ctx context.Context) {
	f.relistCount++
}

func newTestEventedPLEG(t *testing.T, genericPleg *fakeGenericPLEG) *EventedPLEG {
	t.Helper()
	return &EventedPLEG{
		genericPleg:                 genericPleg,
		eventedPlegMaxStreamRetries: 5,
	}
}

func TestHealthyEventedPLEG(t *testing.T) {
	pleg := newTestEventedPLEG(t, newFakeGenericPLEG())

	isHealthy, err := pleg.Healthy()
	require.NoError(t, err)
	assert.True(t, isHealthy)
}

func TestProcessCRIEventsRequestsRelistOnlyForUnexpectedTermination(t *testing.T) {
	tests := []struct {
		name       string
		event      *v1.ContainerEventResponse
		wantRelist []types.UID
	}{
		{
			name: "stopped non-zero exit requests relist",
			event: newContainerEvent("pod1", "container1", v1.ContainerEventType_CONTAINER_STOPPED_EVENT, &v1.ContainerStatus{
				Id:       "container1",
				State:    v1.ContainerState_CONTAINER_EXITED,
				ExitCode: 2,
			}),
			wantRelist: []types.UID{"pod1"},
		},
		{
			name: "stopped OOMKilled requests relist",
			event: newContainerEvent("pod1", "container1", v1.ContainerEventType_CONTAINER_STOPPED_EVENT, &v1.ContainerStatus{
				Id:       "container1",
				State:    v1.ContainerState_CONTAINER_EXITED,
				ExitCode: 0,
				Reason:   "OOMKilled",
			}),
			wantRelist: []types.UID{"pod1"},
		},
		{
			name: "stopped clean exit waits for generic relist",
			event: newContainerEvent("pod1", "container1", v1.ContainerEventType_CONTAINER_STOPPED_EVENT, &v1.ContainerStatus{
				Id:       "container1",
				State:    v1.ContainerState_CONTAINER_EXITED,
				ExitCode: 0,
			}),
		},
		{
			name: "started event waits for generic relist",
			event: newContainerEvent("pod1", "container1", v1.ContainerEventType_CONTAINER_STARTED_EVENT, &v1.ContainerStatus{
				Id:       "container1",
				State:    v1.ContainerState_CONTAINER_RUNNING,
				ExitCode: 2,
			}),
		},
		{
			name: "deleted event waits for generic relist",
			event: newContainerEvent("pod1", "container1", v1.ContainerEventType_CONTAINER_DELETED_EVENT, &v1.ContainerStatus{
				Id:       "container1",
				State:    v1.ContainerState_CONTAINER_EXITED,
				ExitCode: 2,
			}),
		},
		{
			name: "missing matching container status waits for generic relist",
			event: newContainerEvent("pod1", "container1", v1.ContainerEventType_CONTAINER_STOPPED_EVENT, &v1.ContainerStatus{
				Id:       "container2",
				State:    v1.ContainerState_CONTAINER_EXITED,
				ExitCode: 2,
			}),
		},
		{
			name: "same container name unexpected non-event instance waits for generic relist",
			event: newContainerEvent("pod1", "container2", v1.ContainerEventType_CONTAINER_STOPPED_EVENT,
				&v1.ContainerStatus{
					Id:       "container1",
					Metadata: &v1.ContainerMetadata{Name: "c1"},
					State:    v1.ContainerState_CONTAINER_EXITED,
					ExitCode: 2,
				},
				&v1.ContainerStatus{
					Id:       "container2",
					Metadata: &v1.ContainerMetadata{Name: "c1"},
					State:    v1.ContainerState_CONTAINER_EXITED,
					ExitCode: 0,
				},
			),
		},
		{
			name: "same container name unexpected event instance requests relist",
			event: newContainerEvent("pod1", "container1", v1.ContainerEventType_CONTAINER_STOPPED_EVENT,
				&v1.ContainerStatus{
					Id:       "container1",
					Metadata: &v1.ContainerMetadata{Name: "c1"},
					State:    v1.ContainerState_CONTAINER_EXITED,
					ExitCode: 2,
				},
				&v1.ContainerStatus{
					Id:       "container2",
					Metadata: &v1.ContainerMetadata{Name: "c1"},
					State:    v1.ContainerState_CONTAINER_RUNNING,
				},
			),
			wantRelist: []types.UID{"pod1"},
		},
		{
			name: "missing sandbox metadata waits for generic relist",
			event: &v1.ContainerEventResponse{
				ContainerId:        "container1",
				ContainerEventType: v1.ContainerEventType_CONTAINER_STOPPED_EVENT,
				CreatedAt:          time.Now().UnixNano(),
				ContainersStatuses: []*v1.ContainerStatus{{
					Id:       "container1",
					State:    v1.ContainerState_CONTAINER_EXITED,
					ExitCode: 2,
				}},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			genericPleg := newFakeGenericPLEG()
			eventedPleg := newTestEventedPLEG(t, genericPleg)
			logger := ktesting.NewLogger(t, ktesting.DefaultConfig)
			eventsCh := make(chan *v1.ContainerEventResponse, 1)
			eventsCh <- test.event
			close(eventsCh)

			eventedPleg.processCRIEvents(logger, eventsCh)

			assert.Equal(t, test.wantRelist, genericPleg.relistRequests)
			assert.Empty(t, genericPleg.reinspectRequests)
			assert.Zero(t, genericPleg.relistCount)
			select {
			case event := <-genericPleg.watchCh:
				t.Fatalf("EventedPLEG sent a pod lifecycle event directly: %#v", event)
			default:
			}
		})
	}
}

func TestEventedPLEGDelegatesToGenericPLEG(t *testing.T) {
	genericPleg := newFakeGenericPLEG()
	eventedPleg := newTestEventedPLEG(t, genericPleg)
	logger := ktesting.NewLogger(t, ktesting.DefaultConfig)

	assert.Equal(t, genericPleg.watchCh, eventedPleg.Watch())

	eventedPleg.Relist(context.Background())
	assert.Equal(t, 1, genericPleg.relistCount)

	eventedPleg.RequestRelist(logger, "pod1")
	assert.Equal(t, []types.UID{"pod1"}, genericPleg.relistRequests)

	eventedPleg.RequestReinspect("pod2")
	assert.Equal(t, []types.UID{"pod2"}, genericPleg.reinspectRequests)
}

func newContainerEvent(podUID types.UID, containerID string, eventType v1.ContainerEventType, statuses ...*v1.ContainerStatus) *v1.ContainerEventResponse {
	return &v1.ContainerEventResponse{
		ContainerId:        containerID,
		ContainerEventType: eventType,
		CreatedAt:          time.Now().UnixNano(),
		PodSandboxStatus: &v1.PodSandboxStatus{
			Metadata: &v1.PodSandboxMetadata{
				Uid: string(podUID),
			},
		},
		ContainersStatuses: statuses,
	}
}
