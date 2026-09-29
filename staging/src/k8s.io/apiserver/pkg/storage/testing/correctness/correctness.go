/*
Copyright The Kubernetes Authors.

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

package correctness

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
)

type testStep struct {
	Name             string
	Request          Request
	CorrectResponse  Response
	ExpectedEvent    *watch.Event
	InvalidResponses []Response
}

func correctnessTestSteps() []testStep {
	pod1UID := types.UID("uid-1")
	pod2UID := types.UID("uid-2")
	pod3UID := types.UID("uid-3")
	pod4UID := types.UID("uid-4")
	pod5UID := types.UID("uid-5")
	wrongUID := types.UID("wrong-uid")
	pod1 := newTestPod("pod1", "ns1", pod1UID, "")
	pod2 := newTestPod("pod2", "ns1", pod2UID, "")
	pod3 := newTestPod("pod3", "ns1", pod3UID, "")
	pod4 := newTestPod("pod4", "ns1", pod4UID, "")
	// ns10 shares the string prefix of ns1, but listing ns1 must not return it.
	pod5 := newTestPod("pod5", "ns10", pod5UID, "")
	pod1Key := mustGetKey(pod1)
	pod2Key := mustGetKey(pod2)
	pod3Key := mustGetKey(pod3)
	pod4Key := mustGetKey(pod4)
	pod5Key := mustGetKey(pod5)
	wrongRV := "99"
	pod2RV := "3"
	pod3RV8 := "8"
	pod3RV9 := "9"
	pod3RV10 := "10"
	errCustom := fmt.Errorf("user rejected update")
	listRecursive := ListRequest{Options: storage.ListOptions{Recursive: true, Predicate: storage.Everything}}
	listNonRecursive := ListRequest{Options: storage.ListOptions{Predicate: storage.Everything}}

	pod3v1 := pod3.DeepCopy()
	pod3v1.Labels = map[string]string{"version": "v1"}
	pod3v2 := pod3.DeepCopy()
	pod3v2.Labels = map[string]string{"version": "v2"}
	pod3v3 := pod3.DeepCopy()
	pod3v3.Labels = map[string]string{"version": "v3"}
	pod3v4 := pod3.DeepCopy()
	pod3v4.Labels = map[string]string{"version": "v4"}
	pod3v5 := pod3.DeepCopy()
	pod3v5.Labels = map[string]string{"version": "v5"}
	pod3v6 := pod3.DeepCopy()
	pod3v6.Labels = map[string]string{"version": "v6"}
	pod3v7 := pod3.DeepCopy()
	pod3v7.Labels = map[string]string{"version": "v7"}

	return []testStep{
		{
			Name: "1. Create pod1 returns success RV=2",
			Request: Request{
				Op:  OpCreate,
				Key: pod1Key,
				Create: CreateRequest{
					Object: pod1,
				},
			},
			CorrectResponse: Response{
				Object: withRV(pod1, "2"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Added,
				Object: withRV(pod1, "2"),
			},
			InvalidResponses: []Response{
				{Object: &example.Pod{}, Err: storage.NewKeyExistsError(pod1Key, 0)},
				{Object: &example.Pod{}, Err: storage.NewKeyNotFoundError(pod1Key, 0)},
				{Object: withRV(pod1, "1")},
				{Object: withRV(pod1, "3")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "2. Create pod1 duplicate returns key exists error",
			Request: Request{
				Op:  OpCreate,
				Key: pod1Key,
				Create: CreateRequest{
					Object: pod1,
				},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    storage.NewKeyExistsError(pod1Key, 0),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod1, "2")},
				{Object: withRV(pod1, "3")},
				{Object: &example.Pod{}, Err: storage.NewKeyNotFoundError(pod1Key, 0)},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "3. Get pod1 returns success RV=2",
			Request: Request{
				Op:  OpGet,
				Key: pod1Key,
			},
			CorrectResponse: Response{
				Object: withRV(pod1, "2"),
			},
			InvalidResponses: []Response{
				{Object: &example.Pod{}, Err: storage.NewKeyNotFoundError(pod1Key, 0)},
				{Object: withRV(pod1, "1")},
				{Object: withRV(pod1, "3")},
				{Object: withRV(pod1, "uid-2")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "4. Create pod2 returns success RV=3",
			Request: Request{
				Op:  OpCreate,
				Key: pod2Key,
				Create: CreateRequest{
					Object: pod2,
				},
			},
			CorrectResponse: Response{
				Object: withRV(pod2, "3"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Added,
				Object: withRV(pod2, "3"),
			},
			InvalidResponses: []Response{
				{Object: &example.Pod{}, Err: storage.NewKeyExistsError(pod2Key, 0)},
				{Object: withRV(pod2, "2")},
				{Object: withRV(pod2, "4")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "5. Get pod2 returns success RV=3",
			Request: Request{
				Op:  OpGet,
				Key: pod2Key,
			},
			CorrectResponse: Response{
				Object: withRV(pod2, "3"),
			},
			InvalidResponses: []Response{
				{Object: &example.Pod{}, Err: storage.NewKeyNotFoundError(pod2Key, 0)},
				{Object: withRV(pod2, "2")},
				{Object: withRV(pod2, "4")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "6. Delete pod2 with mismatched UID precondition returns invalid obj error",
			Request: Request{
				Op:     OpDelete,
				Key:    pod2Key,
				Delete: DeleteRequest{Preconditions: &storage.Preconditions{UID: &wrongUID}},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    (&storage.Preconditions{UID: &wrongUID}).Check(pod2Key, withRV(pod2, "3")),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod2, "4")},
				{Object: nil, Err: nil},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod2Key, 0)},
			},
		},
		{
			Name: "7. Delete pod2 with mismatched ResourceVersion precondition returns invalid obj error",
			Request: Request{
				Op:     OpDelete,
				Key:    pod2Key,
				Delete: DeleteRequest{Preconditions: &storage.Preconditions{ResourceVersion: &wrongRV}},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    (&storage.Preconditions{ResourceVersion: &wrongRV}).Check(pod2Key, withRV(pod2, "3")),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod2, "4")},
				{Object: nil, Err: nil},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod2Key, 0)},
			},
		},
		{
			Name: "8. Delete pod1 with matching UID precondition returns success RV=4",
			Request: Request{
				Op:     OpDelete,
				Key:    pod1Key,
				Delete: DeleteRequest{Preconditions: &storage.Preconditions{UID: &pod1UID}},
			},
			CorrectResponse: Response{
				Object: withRV(pod1, "4"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Deleted,
				Object: withRV(pod1, "4"),
			},
			InvalidResponses: []Response{
				{Object: &example.Pod{}, Err: storage.NewKeyNotFoundError(pod1Key, 0)},
				{Object: withRV(pod1, "2")},
				{Object: withRV(pod1, "3")},
				{Object: withRV(pod1, "5")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "9. Delete pod1 duplicate returns NotFound",
			Request: Request{
				Op:  OpDelete,
				Key: pod1Key,
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    storage.NewKeyNotFoundError(pod1Key, 4),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod1, "4")},
				{Object: withRV(pod1, "5")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "10. Get pod1 after delete returns NotFound",
			Request: Request{
				Op:  OpGet,
				Key: pod1Key,
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    storage.NewKeyNotFoundError(pod1Key, 0),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod1, "2")},
				{Object: withRV(pod1, "4")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "11. Delete pod2 with matching UID and RV preconditions returns success RV=5",
			Request: Request{
				Op:     OpDelete,
				Key:    pod2Key,
				Delete: DeleteRequest{Preconditions: &storage.Preconditions{UID: &pod2UID, ResourceVersion: &pod2RV}},
			},
			CorrectResponse: Response{
				Object: withRV(pod2, "5"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Deleted,
				Object: withRV(pod2, "5"),
			},
			InvalidResponses: []Response{
				{Object: &example.Pod{}, Err: storage.NewKeyNotFoundError(pod2Key, 0)},
				{Object: withRV(pod2, "3")},
				{Object: withRV(pod2, "4")},
				{Object: withRV(pod2, "6")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "12. Get pod2 after delete returns NotFound",
			Request: Request{
				Op:  OpGet,
				Key: pod2Key,
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    storage.NewKeyNotFoundError(pod2Key, 0),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod2, "3")},
				{Object: withRV(pod2, "5")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "13. Update non-existing pod3 with ignoreNotFound=false returns NotFound",
			Request: Request{
				Op:  OpUpdate,
				Key: pod3Key,
				Update: UpdateRequest{
					IgnoreNotFound: false,
					UpdateFunc: storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
						pod := obj.(*example.Pod).DeepCopy()
						pod.Labels = map[string]string{"version": "v1"}
						return pod, nil
					}),
				},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    storage.NewKeyNotFoundError(pod3Key, 5),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v1, "5")},
				{Object: withRV(pod3v1, "6")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "14. Update non-existing pod3 with ignoreNotFound=true and mismatched UID precondition returns invalid obj error",
			Request: Request{
				Op:  OpUpdate,
				Key: pod3Key,
				Update: UpdateRequest{
					IgnoreNotFound: true,
					Preconditions:  &storage.Preconditions{UID: &pod3UID},
					UpdateFunc: storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
						return pod3v1.DeepCopy(), nil
					}),
				},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    (&storage.Preconditions{UID: &pod3UID}).Check(pod3Key, &example.Pod{}),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v1, "6")},
				{Object: nil, Err: nil},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod3Key, 5)},
			},
		},
		{
			Name: "15. Update non-existing pod3 with ignoreNotFound=true and failing UpdateFunc returns user error",
			Request: Request{
				Op:  OpUpdate,
				Key: pod3Key,
				Update: UpdateRequest{
					IgnoreNotFound: true,
					UpdateFunc: func(input runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
						return nil, nil, errCustom
					},
				},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    errCustom,
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v1, "6")},
				{Object: nil, Err: nil},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod3Key, 5)},
			},
		},
		{
			Name: "16. Update non-existing pod3 with ignoreNotFound=true creates pod3 returns success RV=6",
			Request: Request{
				Op:  OpUpdate,
				Key: pod3Key,
				Update: UpdateRequest{
					IgnoreNotFound: true,
					UpdateFunc: storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
						pod := obj.(*example.Pod).DeepCopy()
						pod.Name = pod3.Name
						pod.Namespace = pod3.Namespace
						pod.UID = pod3.UID
						pod.Labels = map[string]string{"version": "v1"}
						return pod, nil
					}),
				},
			},
			CorrectResponse: Response{
				Object: withRV(pod3v1, "6"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Added,
				Object: withRV(pod3v1, "6"),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v1, "5")},
				{Object: withRV(pod3v1, "7")},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod3Key, 5)},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "17. Get pod3 returns success RV=6",
			Request: Request{
				Op:  OpGet,
				Key: pod3Key,
				Get: GetRequest{},
			},
			CorrectResponse: Response{
				Object: withRV(pod3v1, "6"),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v1, "5")},
				{Object: withRV(pod3v1, "7")},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod3Key, 0)},
			},
		},
		{
			Name: "18. Update pod3 with identical data returns existing pod3 RV=6",
			Request: Request{
				Op:  OpUpdate,
				Key: pod3Key,
				Update: UpdateRequest{
					IgnoreNotFound: false,
					UpdateFunc: storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
						return obj.(*example.Pod).DeepCopy(), nil
					}),
				},
			},
			CorrectResponse: Response{
				Object: withRV(pod3v1, "6"),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v1, "7")},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod3Key, 6)},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "19. Update pod3 modifying data returns success RV=7",
			Request: Request{
				Op:  OpUpdate,
				Key: pod3Key,
				Update: UpdateRequest{
					IgnoreNotFound: false,
					UpdateFunc: storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
						pod := obj.(*example.Pod).DeepCopy()
						pod.Labels = map[string]string{"version": "v2"}
						return pod, nil
					}),
				},
			},
			CorrectResponse: Response{
				Object: withRV(pod3v2, "7"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Modified,
				Object: withRV(pod3v2, "7"),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v2, "6")},
				{Object: withRV(pod3v2, "8")},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod3Key, 6)},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "20. Update pod3 with mismatched UID precondition returns invalid obj error",
			Request: Request{
				Op:  OpUpdate,
				Key: pod3Key,
				Update: UpdateRequest{
					Preconditions: &storage.Preconditions{UID: &wrongUID},
					UpdateFunc: storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
						pod := obj.(*example.Pod).DeepCopy()
						pod.Labels = map[string]string{"version": "v3"}
						return pod, nil
					}),
				},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    (&storage.Preconditions{UID: &wrongUID}).Check(pod3Key, withRV(pod3v2, "7")),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v2, "7")},
				{Object: withRV(pod3v2, "8")},
				{Object: nil, Err: nil},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod3Key, 7)},
			},
		},
		{
			Name: "21. Update pod3 with mismatched ResourceVersion precondition returns invalid obj error",
			Request: Request{
				Op:  OpUpdate,
				Key: pod3Key,
				Update: UpdateRequest{
					Preconditions: &storage.Preconditions{ResourceVersion: &wrongRV},
					UpdateFunc: storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
						pod := obj.(*example.Pod).DeepCopy()
						pod.Labels = map[string]string{"version": "v3"}
						return pod, nil
					}),
				},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    (&storage.Preconditions{ResourceVersion: &wrongRV}).Check(pod3Key, withRV(pod3v2, "7")),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v2, "7")},
				{Object: withRV(pod3v2, "8")},
				{Object: nil, Err: nil},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod3Key, 7)},
			},
		},
		{
			Name: "22. Update pod3 with matching UID precondition returns success RV=8",
			Request: Request{
				Op:  OpUpdate,
				Key: pod3Key,
				Update: UpdateRequest{
					Preconditions: &storage.Preconditions{UID: &pod3UID},
					UpdateFunc: storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
						pod := obj.(*example.Pod).DeepCopy()
						pod.Labels = map[string]string{"version": "v3"}
						return pod, nil
					}),
				},
			},
			CorrectResponse: Response{
				Object: withRV(pod3v3, "8"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Modified,
				Object: withRV(pod3v3, "8"),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v3, "7")},
				{Object: withRV(pod3v3, "9")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "23. Update pod3 with matching ResourceVersion precondition returns success RV=9",
			Request: Request{
				Op:  OpUpdate,
				Key: pod3Key,
				Update: UpdateRequest{
					Preconditions: &storage.Preconditions{ResourceVersion: &pod3RV8},
					UpdateFunc: storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
						pod := obj.(*example.Pod).DeepCopy()
						pod.Labels = map[string]string{"version": "v4"}
						return pod, nil
					}),
				},
			},
			CorrectResponse: Response{
				Object: withRV(pod3v4, "9"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Modified,
				Object: withRV(pod3v4, "9"),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v4, "8")},
				{Object: withRV(pod3v4, "10")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "24. Update pod3 with matching UID and ResourceVersion preconditions returns success RV=10",
			Request: Request{
				Op:  OpUpdate,
				Key: pod3Key,
				Update: UpdateRequest{
					Preconditions: &storage.Preconditions{UID: &pod3UID, ResourceVersion: &pod3RV9},
					UpdateFunc: storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
						pod := obj.(*example.Pod).DeepCopy()
						pod.Labels = map[string]string{"version": "v5"}
						return pod, nil
					}),
				},
			},
			CorrectResponse: Response{
				Object: withRV(pod3v5, "10"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Modified,
				Object: withRV(pod3v5, "10"),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v5, "9")},
				{Object: withRV(pod3v5, "11")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "25. Update pod3 with identical data and matching preconditions returns existing pod3 RV=10",
			Request: Request{
				Op:  OpUpdate,
				Key: pod3Key,
				Update: UpdateRequest{
					Preconditions: &storage.Preconditions{UID: &pod3UID, ResourceVersion: &pod3RV10},
					UpdateFunc: storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
						return obj.(*example.Pod).DeepCopy(), nil
					}),
				},
			},
			CorrectResponse: Response{
				Object: withRV(pod3v5, "10"),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v5, "11")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "26. Update pod3 where UpdateFunc returns user error returns error",
			Request: Request{
				Op:  OpUpdate,
				Key: pod3Key,
				Update: UpdateRequest{
					UpdateFunc: func(input runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
						return nil, nil, errCustom
					},
				},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    errCustom,
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v5, "10")},
				{Object: withRV(pod3v5, "11")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "27. Update pod3 with CachedExistingObject returns success RV=11",
			Request: Request{
				Op:  OpUpdate,
				Key: pod3Key,
				Update: UpdateRequest{
					CachedExistingObject: withRV(pod3v5, "10"),
					UpdateFunc: storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
						pod := obj.(*example.Pod).DeepCopy()
						pod.Labels = map[string]string{"version": "v6"}
						return pod, nil
					}),
				},
			},
			CorrectResponse: Response{
				Object: withRV(pod3v6, "11"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Modified,
				Object: withRV(pod3v6, "11"),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v6, "10")},
				{Object: withRV(pod3v6, "12")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "28. Update pod3 validating ResponseMeta passed to UpdateFunc returns success RV=12",
			Request: Request{
				Op:  OpUpdate,
				Key: pod3Key,
				Update: UpdateRequest{
					UpdateFunc: func(input runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
						if res.ResourceVersion != 11 {
							return nil, nil, fmt.Errorf("expected ResourceVersion 11, got %d", res.ResourceVersion)
						}
						pod := input.(*example.Pod).DeepCopy()
						pod.Labels = map[string]string{"version": "v7"}
						return pod, nil, nil
					},
				},
			},
			CorrectResponse: Response{
				Object: withRV(pod3v7, "12"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Modified,
				Object: withRV(pod3v7, "12"),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v7, "11")},
				{Object: withRV(pod3v7, "13")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "29. Delete pod3 returns success RV=13",
			Request: Request{
				Op:     OpDelete,
				Key:    pod3Key,
				Delete: DeleteRequest{},
			},
			CorrectResponse: Response{
				Object: withRV(pod3v7, "13"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Deleted,
				Object: withRV(pod3v7, "13"),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v7, "12")},
				{Object: withRV(pod3v7, "14")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "30. Get pod3 after delete returns NotFound",
			Request: Request{
				Op:  OpGet,
				Key: pod3Key,
				Get: GetRequest{},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    storage.NewKeyNotFoundError(pod3Key, 0),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v7, "12")},
				{Object: withRV(pod3v7, "13")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "31. List pods when none exist returns empty list RV=13",
			Request: Request{
				Op:   OpList,
				Key:  "/pods/",
				List: listRecursive,
			},
			CorrectResponse: Response{
				Object: newTestPodList("13"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("12")},
				{Object: newTestPodList("13", withRV(pod3v7, "12"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "32. Create pod4 returns success RV=14",
			Request: Request{
				Op:  OpCreate,
				Key: pod4Key,
				Create: CreateRequest{
					Object: pod4,
				},
			},
			CorrectResponse: Response{
				Object: withRV(pod4, "14"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Added,
				Object: withRV(pod4, "14"),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod4, "13")},
				{Object: withRV(pod4, "15")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "33. Create pod5 in ns10 returns success RV=15",
			Request: Request{
				Op:  OpCreate,
				Key: pod5Key,
				Create: CreateRequest{
					Object: pod5,
				},
			},
			CorrectResponse: Response{
				Object: withRV(pod5, "15"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Added,
				Object: withRV(pod5, "15"),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod5, "14")},
				{Object: withRV(pod5, "16")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "34. List pods returns pod4 and pod5 RV=15",
			Request: Request{
				Op:   OpList,
				Key:  "/pods/",
				List: listRecursive,
			},
			CorrectResponse: Response{
				Object: newTestPodList("15", withRV(pod4, "14"), withRV(pod5, "15")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("14", withRV(pod4, "14"))},
				{Object: newTestPodList("15", withRV(pod4, "14"))},
				{Object: newTestPodList("15", withRV(pod5, "15"), withRV(pod4, "14"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "35. List ns1 returns pod4 without pod5 from ns10 RV=15",
			Request: Request{
				Op:   OpList,
				Key:  "/pods/ns1",
				List: listRecursive,
			},
			CorrectResponse: Response{
				Object: newTestPodList("15", withRV(pod4, "14")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("15", withRV(pod4, "14"), withRV(pod5, "15"))},
				{Object: newTestPodList("14", withRV(pod4, "14"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "36. List pod4 non-recursively returns pod4 RV=15",
			Request: Request{
				Op:   OpList,
				Key:  pod4Key,
				List: listNonRecursive,
			},
			CorrectResponse: Response{
				Object: newTestPodList("15", withRV(pod4, "14")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("14", withRV(pod4, "14"))},
				{Object: newTestPodList("15")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "37. List deleted pod3 non-recursively returns empty list RV=15",
			Request: Request{
				Op:   OpList,
				Key:  pod3Key,
				List: listNonRecursive,
			},
			CorrectResponse: Response{
				Object: newTestPodList("15"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("15", withRV(pod3v7, "12"))},
				{Object: newTestPodList("13")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "38. List ns1 non-recursively returns empty list RV=15",
			Request: Request{
				Op:   OpList,
				Key:  "/pods/ns1",
				List: listNonRecursive,
			},
			CorrectResponse: Response{
				Object: newTestPodList("15"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("15", withRV(pod4, "14"))},
				{Object: newTestPodList("15", withRV(pod4, "14"), withRV(pod5, "15"))},
				{Object: nil, Err: nil},
			},
		},
	}
}

// RunTestCorrectness executes the operations from the sequential storage model against real storage
// and validates that every transition matches the StorageModel specification.
func RunTestCorrectness(ctx context.Context, t *testing.T, store storage.Interface, storagePrefix string, keyFunc func(obj runtime.Object) (string, error)) {
	versioner := store.Versioner()
	model := NewEmptyModel(storagePrefix, func() runtime.Object { return &example.Pod{} }, func() runtime.Object { return &example.PodList{} }, versioner)

	watchRequest := WatchRequest{ResourceVersion: "1"}
	watcher, err := store.Watch(ctx, "/pods/", storage.ListOptions{ResourceVersion: watchRequest.ResourceVersion, Predicate: storage.Everything, Recursive: true})
	require.NoError(t, err)
	defer watcher.Stop()

	var history []Change
	for _, step := range correctnessTestSteps() {
		var out runtime.Object = &example.Pod{}
		var err error
		switch step.Request.Op {
		case OpCreate:
			err = store.Create(ctx, step.Request.Key, step.Request.Create.Object, out, 0)
		case OpGet:
			err = store.Get(ctx, step.Request.Key, step.Request.Get.Options, out)
		case OpList:
			out = &example.PodList{}
			err = store.GetList(ctx, step.Request.Key, step.Request.List.Options, out)
		case OpDelete:
			err = store.Delete(ctx, step.Request.Key, out, step.Request.Delete.Preconditions, storage.ValidateAllObjectFunc, nil, storage.DeleteOptions{})
		case OpUpdate:
			err = store.GuaranteedUpdate(ctx, step.Request.Key, out, step.Request.Update.IgnoreNotFound, step.Request.Update.Preconditions, step.Request.Update.UpdateFunc, step.Request.Update.CachedExistingObject)
		default:
			t.Fatalf("unknown operation: %v", step.Request.Op)
		}
		var respObj runtime.Object
		if err == nil {
			respObj = out
		}
		resp := Response{Object: respObj, Err: err}
		ok, next, change := model.Step(step.Request, resp)
		if respObj != nil {
			acc, _ := meta.CommonAccessor(respObj)
			t.Logf("Step: %s, State RV before: %d, Response RV: %s, Obj: %+v, err: %v", step.Name, model.ResourceVersion, acc.GetResourceVersion(), respObj, err)
		} else {
			t.Logf("Step: %s, State RV before: %d, Response Err: %v", step.Name, model.ResourceVersion, err)
		}
		require.True(t, ok, "step %s failed to match model state transition: req=%+v resp=%+v", step.Name, step.Request, resp)
		model = next
		if change != nil {
			history = append(history, *change)
		}
	}

	gotEvents := collectEventsTillRV(t, watcher, versioner, model.ResourceVersion)
	validator := NewWatchValidator(versioner, keyFunc, history)
	require.NoError(t, validator.ValidateWatch(watchRequest, WatchResponse{Events: gotEvents}))
}

func collectEventsTillRV(t *testing.T, watcher watch.Interface, versioner storage.Versioner, targetRV uint64) []watch.Event {
	events := []watch.Event{}
	for e := range watcher.ResultChan() {
		if cacheable, ok := e.Object.(runtime.CacheableObject); ok {
			e.Object = cacheable.GetObject()
		}
		events = append(events, e)
		if e.Type == watch.Error {
			_, open := <-watcher.ResultChan()
			require.False(t, open, "watch channel should be closed after watch.Error")
			break
		}
		accessor, err := meta.Accessor(e.Object)
		require.NoError(t, err)
		rv, err := versioner.ParseResourceVersion(accessor.GetResourceVersion())
		require.NoError(t, err)
		if rv >= targetRV {
			break
		}
	}
	return events
}

func newTestPod(name, namespace string, uid types.UID, rv string) *example.Pod {
	pod := &example.Pod{}
	pod.Name = name
	pod.Namespace = namespace
	pod.UID = uid
	pod.ResourceVersion = rv
	return pod
}

func withRV(pod *example.Pod, rv string) *example.Pod {
	new := pod.DeepCopy()
	new.ResourceVersion = rv
	return new
}

func withLabel(pod *example.Pod, rv string, label string, value string) *example.Pod {
	new := pod.DeepCopy()
	new.Labels = map[string]string{label: value}
	new.ResourceVersion = rv
	return new
}

func dropLabel(pod *example.Pod, rv string, label string) *example.Pod {
	new := pod.DeepCopy()
	delete(new.Labels, label)
	new.ResourceVersion = rv
	return new
}

func newTestPodList(rv string, pods ...*example.Pod) *example.PodList {
	list := &example.PodList{Items: []example.Pod{}}
	list.ResourceVersion = rv
	for _, pod := range pods {
		list.Items = append(list.Items, *pod)
	}
	return list
}

func mustGetKey(obj runtime.Object) string {
	key, err := getKey(obj)
	if err != nil {
		panic(err)
	}
	return key
}

func getKey(obj runtime.Object) (string, error) {
	pod, ok := obj.(*example.Pod)
	if !ok {
		return "", fmt.Errorf("object is not a pod: %T", obj)
	}
	return "/pods/" + pod.Namespace + "/" + pod.Name, nil
}
