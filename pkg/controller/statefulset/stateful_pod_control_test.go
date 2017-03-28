/*
Copyright 2016 The Kubernetes Authors.

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

package statefulset

import (
	"errors"
	"strings"
	"testing"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"

	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/record"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
)

func TestStatefulPodControlCreatesPods(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	fakeClient.AddReactor("get", "persistentvolumeclaims", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewNotFound(action.GetResource().GroupResource(), action.GetResource().Resource)
	})
	fakeClient.AddReactor("create", "persistentvolumeclaims", func(action core.Action) (bool, runtime.Object, error) {
		create := action.(core.CreateAction)
		return true, create.GetObject(), nil
	})
	fakeClient.AddReactor("create", "pods", func(action core.Action) (bool, runtime.Object, error) {
		create := action.(core.CreateAction)
		return true, create.GetObject(), nil
	})
	if err := control.CreateStatefulPod(set, pod); err != nil {
		t.Errorf("StatefulPodControl failed to create Pod error: %s", err)
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 2 {
		t.Errorf("Expected 2 events for successful create found %d", eventCount)
	}
	for i := range events {
		if !strings.Contains(events[i], v1.EventTypeNormal) {
			t.Errorf("Expected normal events found %s", events[i])
		}
	}
}

func TestStatefulPodControlCreatePodExists(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	pvcs := getPersistentVolumeClaims(set, pod)
	fakeClient.AddReactor("get", "persistentvolumeclaims", func(action core.Action) (bool, runtime.Object, error) {
		claim := pvcs[action.GetResource().GroupResource().Resource]
		return true, &claim, nil
	})
	fakeClient.AddReactor("create", "persistentvolumeclaims", func(action core.Action) (bool, runtime.Object, error) {
		create := action.(core.CreateAction)
		return true, create.GetObject(), nil
	})
	fakeClient.AddReactor("create", "pods", func(action core.Action) (bool, runtime.Object, error) {
		return true, pod, apierrors.NewAlreadyExists(action.GetResource().GroupResource(), pod.Name)
	})
	control = NewRealStatefulPodControl(fakeClient, recorder)
	if err := control.CreateStatefulPod(set, pod); !apierrors.IsAlreadyExists(err) {
		t.Errorf("Failed to create Pod error: %s", err)
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 0 {
		t.Errorf("Expected 0 events when Pod and PVC exist found %d", eventCount)
		for i := range events {
			t.Log(events[i])
		}
	}
}

func TestStatefulPodControlCreatePodPvcCreateFailure(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	fakeClient.AddReactor("get", "persistentvolumeclaims", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewNotFound(action.GetResource().GroupResource(), action.GetResource().Resource)
	})
	fakeClient.AddReactor("create", "persistentvolumeclaims", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewInternalError(errors.New("API server down"))
	})
	fakeClient.AddReactor("create", "pods", func(action core.Action) (bool, runtime.Object, error) {
		create := action.(core.CreateAction)
		return true, create.GetObject(), nil
	})
	control = NewRealStatefulPodControl(fakeClient, recorder)
	if err := control.CreateStatefulPod(set, pod); err == nil {
		t.Error("Failed to produce error on PVC creation failure")
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 2 {
		t.Errorf("Expected 2 events for PVC create failure found %d", eventCount)
	}
	for i := range events {
		if !strings.Contains(events[i], v1.EventTypeWarning) {
			t.Errorf("Expected normal events found %s", events[i])
		}
	}
}

func TestStatefulPodControlCreatePodPvcGetFailure(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	fakeClient.AddReactor("get", "persistentvolumeclaims", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewInternalError(errors.New("API server down"))
	})
	fakeClient.AddReactor("create", "persistentvolumeclaims", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewInternalError(errors.New("API server down"))
	})
	fakeClient.AddReactor("create", "pods", func(action core.Action) (bool, runtime.Object, error) {
		create := action.(core.CreateAction)
		return true, create.GetObject(), nil
	})
	control = NewRealStatefulPodControl(fakeClient, recorder)
	if err := control.CreateStatefulPod(set, pod); err == nil {
		t.Error("Failed to produce error on PVC creation failure")
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 2 {
		t.Errorf("Expected 2 events for PVC create failure found %d", eventCount)
	}
	for i := range events {
		if !strings.Contains(events[i], v1.EventTypeWarning) {
			t.Errorf("Expected normal events found %s", events[i])
		}
	}
}

func TestStatefulPodControlCreatePodFailed(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	fakeClient.AddReactor("get", "persistentvolumeclaims", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewNotFound(action.GetResource().GroupResource(), action.GetResource().Resource)
	})
	fakeClient.AddReactor("create", "persistentvolumeclaims", func(action core.Action) (bool, runtime.Object, error) {
		create := action.(core.CreateAction)
		return true, create.GetObject(), nil
	})
	fakeClient.AddReactor("create", "pods", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewInternalError(errors.New("API server down"))
	})
	control = NewRealStatefulPodControl(fakeClient, recorder)
	if err := control.CreateStatefulPod(set, pod); err == nil {
		t.Error("Failed to produce error on Pod creation failure")
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 2 {
		t.Errorf("Expected 2 events for failed Pod create found %d", eventCount)
	} else if !strings.Contains(events[0], v1.EventTypeNormal) {
		t.Errorf("Expected normal event found %s", events[0])

	} else if !strings.Contains(events[1], v1.EventTypeWarning) {
		t.Errorf("Expected warning event found %s", events[1])

	}
}

func TestStatefulPodControlNoOpUpdate(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	fakeClient.AddReactor("*", "*", func(action core.Action) (bool, runtime.Object, error) {
		t.Error("no-op update should not make any client invocation")
		return true, nil, apierrors.NewInternalError(errors.New("If we are here we have a problem"))
	})
	if err := control.UpdateStatefulPod(set, pod); err != nil {
		t.Errorf("Error returned on no-op update error: %s", err)
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 0 {
		t.Errorf("Expected 0 events for no-op update found %d", eventCount)
	}
}

func TestStatefulPodControlUpdatesIdentity(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	fakeClient.AddReactor("update", "pods", func(action core.Action) (bool, runtime.Object, error) {
		update := action.(core.UpdateAction)
		return true, update.GetObject(), nil
	})
	pod.Name = "goo-0"
	control = NewRealStatefulPodControl(fakeClient, recorder)
	if err := control.UpdateStatefulPod(set, pod); err != nil {
		t.Errorf("Successful update returned an error: %s", err)
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 1 {
		t.Errorf("Expected 1 event for successful Pod update found %d", eventCount)
	} else if !strings.Contains(events[0], v1.EventTypeNormal) {
		t.Errorf("Expected normal event found %s", events[0])
	}
	if !identityMatches(set, pod) {
		t.Error("Name update failed identity does not match")
	}
}

func TestStatefulPodControlUpdateIdentityFailure(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	fakeClient.AddReactor("update", "pods", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewInternalError(errors.New("API server down"))
	})
	pod.Name = "goo-0"
	control = NewRealStatefulPodControl(fakeClient, recorder)
	if err := control.UpdateStatefulPod(set, pod); err == nil {
		t.Error("Falied update does not generate an error")
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 1 {
		t.Errorf("Expected 1 event for failed Pod update found %d", eventCount)
	} else if !strings.Contains(events[0], v1.EventTypeWarning) {
		t.Errorf("Expected warning event found %s", events[0])
	}
	if identityMatches(set, pod) {
		t.Error("Failed update mutated Pod identity")
	}
}

func TestStatefulPodControlUpdatesPodStorage(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	pvcs := getPersistentVolumeClaims(set, pod)
	volumes := make([]v1.Volume, len(pod.Spec.Volumes))
	for i := range pod.Spec.Volumes {
		if _, contains := pvcs[pod.Spec.Volumes[i].Name]; !contains {
			volumes = append(volumes, pod.Spec.Volumes[i])
		}
	}
	pod.Spec.Volumes = volumes
	fakeClient.AddReactor("update", "pods", func(action core.Action) (bool, runtime.Object, error) {
		update := action.(core.UpdateAction)
		return true, update.GetObject(), nil
	})
	fakeClient.AddReactor("get", "persistentvolumeclaims", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewNotFound(action.GetResource().GroupResource(), action.GetResource().Resource)
	})
	fakeClient.AddReactor("create", "persistentvolumeclaims", func(action core.Action) (bool, runtime.Object, error) {
		update := action.(core.UpdateAction)
		return true, update.GetObject(), nil
	})
	control = NewRealStatefulPodControl(fakeClient, recorder)
	if err := control.UpdateStatefulPod(set, pod); err != nil {
		t.Errorf("Successful update returned an error: %s", err)
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 2 {
		t.Errorf("Expected 2 event for successful Pod storage update found %d", eventCount)
	}
	for i := range events {
		if !strings.Contains(events[i], v1.EventTypeNormal) {
			t.Errorf("Expected normal event found %s", events[i])
		}
	}
	if !storageMatches(set, pod) {
		t.Error("Name update failed identity does not match")
	}
}

func TestStatefulPodControlUpdatePodStorageFailure(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	pvcs := getPersistentVolumeClaims(set, pod)
	volumes := make([]v1.Volume, len(pod.Spec.Volumes))
	for i := range pod.Spec.Volumes {
		if _, contains := pvcs[pod.Spec.Volumes[i].Name]; !contains {
			volumes = append(volumes, pod.Spec.Volumes[i])
		}
	}
	pod.Spec.Volumes = volumes
	fakeClient.AddReactor("update", "pods", func(action core.Action) (bool, runtime.Object, error) {
		update := action.(core.UpdateAction)
		return true, update.GetObject(), nil
	})
	fakeClient.AddReactor("get", "persistentvolumeclaims", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewNotFound(action.GetResource().GroupResource(), action.GetResource().Resource)
	})
	fakeClient.AddReactor("create", "persistentvolumeclaims", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewInternalError(errors.New("API server down"))
	})
	control = NewRealStatefulPodControl(fakeClient, recorder)
	if err := control.UpdateStatefulPod(set, pod); err == nil {
		t.Error("Failed Pod storage update did not return an error")
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 2 {
		t.Errorf("Expected 2 event for failed Pod storage update found %d", eventCount)
	}
	for i := range events {
		if !strings.Contains(events[i], v1.EventTypeWarning) {
			t.Errorf("Expected normal event found %s", events[i])
		}
	}
	if storageMatches(set, pod) {
		t.Error("Storag matches on failed update")
	}
}

func TestStatefulPodControlUpdatePodConflictSuccess(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	attempts := 0
	fakeClient.AddReactor("update", "pods", func(action core.Action) (bool, runtime.Object, error) {
		update := action.(core.UpdateAction)
		if attempts < maxUpdateRetries/2 {
			attempts++
			return true, update.GetObject(), apierrors.NewConflict(action.GetResource().GroupResource(), pod.Name, errors.New("conflict"))
		} else {
			return true, update.GetObject(), nil
		}
	})
	fakeClient.AddReactor("get", "pods", func(action core.Action) (bool, runtime.Object, error) {
		pod.Name = "goo-0"
		return true, pod, nil

	})
	pod.Name = "goo-0"
	control = NewRealStatefulPodControl(fakeClient, recorder)
	if err := control.UpdateStatefulPod(set, pod); err != nil {
		t.Errorf("Successful update returned an error: %s", err)
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 1 {
		t.Errorf("Expected 1 event for successful Pod update found %d", eventCount)
	} else if !strings.Contains(events[0], v1.EventTypeNormal) {
		t.Errorf("Expected normal event found %s", events[0])
	}
	if !identityMatches(set, pod) {
		t.Error("Name update failed identity does not match")
	}
}

func TestStatefulPodControlUpdatePodConflictFailure(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	fakeClient.AddReactor("update", "pods", func(action core.Action) (bool, runtime.Object, error) {
		update := action.(core.UpdateAction)
		return true, update.GetObject(), apierrors.NewConflict(action.GetResource().GroupResource(), pod.Name, errors.New("conflict"))

	})
	fakeClient.AddReactor("get", "pods", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewInternalError(errors.New("API server down"))

	})
	pod.Name = "goo-0"
	control = NewRealStatefulPodControl(fakeClient, recorder)
	if err := control.UpdateStatefulPod(set, pod); err == nil {
		t.Error("Falied update did not reaturn an error")
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 1 {
		t.Errorf("Expected 1 event for failed Pod update found %d", eventCount)
	} else if !strings.Contains(events[0], v1.EventTypeWarning) {
		t.Errorf("Expected normal event found %s", events[0])
	}
	if identityMatches(set, pod) {
		t.Error("Identity matches on failed update")
	}
}

func TestStatefulPodControlUpdatePodConflictMaxRetries(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	fakeClient.AddReactor("update", "pods", func(action core.Action) (bool, runtime.Object, error) {
		update := action.(core.UpdateAction)
		return true, update.GetObject(), apierrors.NewConflict(action.GetResource().GroupResource(), pod.Name, errors.New("conflict"))

	})
	fakeClient.AddReactor("get", "pods", func(action core.Action) (bool, runtime.Object, error) {
		pod.Name = "goo-0"
		return true, pod, nil

	})
	pod.Name = "goo-0"
	control = NewRealStatefulPodControl(fakeClient, recorder)
	if err := control.UpdateStatefulPod(set, pod); err == nil {
		t.Error("Falied update did not reaturn an error")
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 1 {
		t.Errorf("Expected 1 event for failed Pod update found %d", eventCount)
	} else if !strings.Contains(events[0], v1.EventTypeWarning) {
		t.Errorf("Expected normal event found %s", events[0])
	}
	if identityMatches(set, pod) {
		t.Error("Identity matches on failed update")
	}
}

func TestStatefulPodControlDeletesStatefulPod(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	fakeClient.AddReactor("delete", "pods", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, nil
	})
	if err := control.DeleteStatefulPod(set, pod); err != nil {
		t.Errorf("Error returned on successful delete: %s", err)
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 1 {
		t.Errorf("Expected 1 events for successful delete found %d", eventCount)
	} else if !strings.Contains(events[0], v1.EventTypeNormal) {
		t.Errorf("Expected normal event found %s", events[0])
	}
}

func TestStatefulPodControlDeleteFailure(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	fakeClient.AddReactor("delete", "pods", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewInternalError(errors.New("API server down"))
	})
	if err := control.DeleteStatefulPod(set, pod); err == nil {
		t.Error("Fialed to return error on failed delete")
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 1 {
		t.Errorf("Expected 1 events for failed delete found %d", eventCount)
	} else if !strings.Contains(events[0], v1.EventTypeWarning) {
		t.Errorf("Expected warning event found %s", events[0])
	}
}

func TestStatefulPodControlUpdatesSetStatus(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	fakeClient.AddReactor("update", "statefulsets", func(action core.Action) (bool, runtime.Object, error) {
		update := action.(core.UpdateAction)
		return true, update.GetObject(), nil
	})
	if err := control.UpdateStatefulSetReplicas(set, 2); err != nil {
		t.Errorf("Error returned on successful status update: %s", err)
	}
	if set.Status.Replicas != 2 {
		t.Errorf("UpdateStatefulSetStatus mutated the sets replicas %d", set.Status.Replicas)
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 0 {
		t.Errorf("Expected 0 events for successful status update %d", eventCount)
	}
}

func TestStatefulPodControlUpdateReplicasFailure(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	replicas := set.Status.Replicas
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	fakeClient.AddReactor("update", "statefulsets", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewInternalError(errors.New("API server down"))
	})
	if err := control.UpdateStatefulSetReplicas(set, 2); err == nil {
		t.Error("Failed update did not return error")
	}
	if set.Status.Replicas != replicas {
		t.Errorf("UpdateStatefulSetStatus mutated the sets replicas %d", replicas)
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 0 {
		t.Errorf("Expected 0 events for successful status update %d", eventCount)
	}
}

func TestStatefulPodControlUpdateReplicasConflict(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	attempts := 0
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	fakeClient.AddReactor("update", "statefulsets", func(action core.Action) (bool, runtime.Object, error) {

		update := action.(core.UpdateAction)
		if attempts < maxUpdateRetries/2 {
			attempts++
			return true, update.GetObject(), apierrors.NewConflict(action.GetResource().GroupResource(), set.Name, errors.New("Object already exists"))
		} else {
			return true, update.GetObject(), nil
		}
	})
	fakeClient.AddReactor("get", "statefulsets", func(action core.Action) (bool, runtime.Object, error) {
		return true, set, nil
	})
	if err := control.UpdateStatefulSetReplicas(set, 2); err != nil {
		t.Errorf("UpdateStatefulSetStatus returned an error: %s", err)
	}
	if set.Status.Replicas != 2 {
		t.Errorf("UpdateStatefulSetStatus mutated the sets replicas %d", set.Status.Replicas)
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 0 {
		t.Errorf("Expected 0 events for successful status update %d", eventCount)
	}
}

func TestStatefulPodControlUpdateReplicasConflictFailure(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	replicas := set.Status.Replicas
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	fakeClient.AddReactor("update", "statefulsets", func(action core.Action) (bool, runtime.Object, error) {
		update := action.(core.UpdateAction)
		return true, update.GetObject(), apierrors.NewConflict(action.GetResource().GroupResource(), set.Name, errors.New("Object already exists"))
	})
	fakeClient.AddReactor("get", "statefulsets", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewInternalError(errors.New("API server down"))
	})
	if err := control.UpdateStatefulSetReplicas(set, 2); err == nil {
		t.Error("UpdateStatefulSetStatus failed to return an error on get failure")
	}
	if set.Status.Replicas != replicas {
		t.Errorf("UpdateStatefulSetStatus mutated the sets replicas %d", set.Status.Replicas)
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 0 {
		t.Errorf("Expected 0 events for successful status update %d", eventCount)
	}
}

func TestStatefulPodControlUpdateReplicasConflictMaxRetries(t *testing.T) {
	recorder := record.NewFakeRecorder(10)
	set := newStatefulSet(3)
	replicas := set.Status.Replicas
	fakeClient := &fake.Clientset{}
	control := NewRealStatefulPodControl(fakeClient, recorder)
	fakeClient.AddReactor("update", "statefulsets", func(action core.Action) (bool, runtime.Object, error) {
		return true, newStatefulSet(3), apierrors.NewConflict(action.GetResource().GroupResource(), set.Name, errors.New("Object already exists"))
	})
	fakeClient.AddReactor("get", "statefulsets", func(action core.Action) (bool, runtime.Object, error) {
		return true, newStatefulSet(3), nil
	})
	if err := control.UpdateStatefulSetReplicas(set, 2); err == nil {
		t.Error("UpdateStatefulSetStatus failure did not return an error ")
	}
	if set.Status.Replicas != replicas {
		t.Errorf("UpdateStatefulSetStatus mutated the sets replicas %d", set.Status.Replicas)
	}
	events := collectEvents(recorder.Events)
	if eventCount := len(events); eventCount != 0 {
		t.Errorf("Expected 0 events for successful status update %d", eventCount)
	}
}

func collectEvents(source <-chan string) []string {
	done := false
	events := make([]string, 0)
	for !done {
		select {
		case event := <-source:
			events = append(events, event)
		default:
			done = true
		}
	}
	return events
}
