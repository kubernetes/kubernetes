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
	"time"

	"github.com/stretchr/testify/require"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
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
	pod5RV18 := "18"
	errCustom := fmt.Errorf("user rejected update")

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
			Name: "3. Create pod2 returns success RV=3",
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
			Name: "4. Delete pod2 with mismatched UID precondition returns invalid obj error",
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
			Name: "5. Delete pod2 with mismatched ResourceVersion precondition returns invalid obj error",
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
			Name: "6. Delete pod1 with matching UID precondition returns success RV=4",
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
			Name: "7. Delete pod1 duplicate returns NotFound",
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
			Name: "8. Delete pod2 with matching UID and RV preconditions returns success RV=5",
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
			Name: "9. Update non-existing pod3 with ignoreNotFound=false returns NotFound",
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
			Name: "10. Update non-existing pod3 with ignoreNotFound=true and mismatched UID precondition returns invalid obj error",
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
			Name: "11. Update non-existing pod3 with ignoreNotFound=true and failing UpdateFunc returns user error",
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
			Name: "12. Update non-existing pod3 with ignoreNotFound=true creates pod3 returns success RV=6",
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
			Name: "13. Update pod3 with identical data returns existing pod3 RV=6",
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
			Name: "14. Update pod3 modifying data returns success RV=7",
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
			Name: "15. Update pod3 with mismatched UID precondition returns invalid obj error",
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
			Name: "16. Update pod3 with mismatched ResourceVersion precondition returns invalid obj error",
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
			Name: "17. Update pod3 with matching UID precondition returns success RV=8",
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
			Name: "18. Update pod3 with matching ResourceVersion precondition returns success RV=9",
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
			Name: "19. Update pod3 with matching UID and ResourceVersion preconditions returns success RV=10",
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
			Name: "20. Update pod3 with identical data and matching preconditions returns existing pod3 RV=10",
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
			Name: "21. Update pod3 where UpdateFunc returns user error returns error",
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
			Name: "22. Update pod3 with CachedExistingObject returns success RV=11",
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
			Name: "23. Update pod3 validating ResponseMeta passed to UpdateFunc returns success RV=12",
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
			Name: "24. Delete pod3 where ValidateDeletion returns user error returns error",
			Request: Request{
				Op:  OpDelete,
				Key: pod3Key,
				Delete: DeleteRequest{
					ValidateDeletion: func(ctx context.Context, obj runtime.Object) error {
						return errCustom
					},
				},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    errCustom,
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v7, "12")},
				{Object: withRV(pod3v7, "13")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "25. Delete pod3 with CachedExistingObject and failing ValidateDeletion returns user error",
			Request: Request{
				Op:  OpDelete,
				Key: pod3Key,
				Delete: DeleteRequest{
					CachedExistingObject: withRV(pod3v6, "11"),
					ValidateDeletion: func(ctx context.Context, obj runtime.Object) error {
						return errCustom
					},
				},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    errCustom,
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v7, "12")},
				{Object: withRV(pod3v7, "13")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "26. Delete pod3 with stale CachedExistingObject where only suggestion passes Preconditions returns invalid obj error",
			Request: Request{
				Op:  OpDelete,
				Key: pod3Key,
				Delete: DeleteRequest{
					Preconditions:        &storage.Preconditions{ResourceVersion: &pod3RV10},
					CachedExistingObject: withRV(pod3v5, "10"),
				},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    (&storage.Preconditions{ResourceVersion: &pod3RV10}).Check(pod3Key, withRV(pod3v7, "12")),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v7, "12")},
				{Object: withRV(pod3v7, "13")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "27. Compact at ResourceVersion=11",
			Request: Request{
				Op:      OpCompact,
				Compact: CompactRequest{ResourceVersion: "11"},
			},
			CorrectResponse: Response{},
			InvalidResponses: []Response{
				{Err: storage.NewTooLargeResourceVersionError(11, 12, 0)},
			},
		},
		{
			Name: "28. Delete pod3 with stale CachedExistingObject where ValidateDeletion fails on stale RV retries and returns success RV=14",
			Request: Request{
				Op:  OpDelete,
				Key: pod3Key,
				Delete: DeleteRequest{
					CachedExistingObject: withRV(pod3v5, "10"),
					ValidateDeletion: func(ctx context.Context, obj runtime.Object) error {
						pod := obj.(*example.Pod)
						if pod.ResourceVersion != "12" {
							return fmt.Errorf("expected ResourceVersion 12, got %s", pod.ResourceVersion)
						}
						return nil
					},
				},
			},
			CorrectResponse: Response{
				Object: withRV(pod3v7, "14"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Deleted,
				Object: withRV(pod3v7, "14"),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod3v7, "12")},
				{Object: withRV(pod3v7, "13")},
				{Object: withRV(pod3v7, "15")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "29. Compact between writes at ResourceVersion=13",
			Request: Request{
				Op:      OpCompact,
				Compact: CompactRequest{ResourceVersion: "13"},
			},
			CorrectResponse: Response{},
			InvalidResponses: []Response{
				{Err: storage.NewTooLargeResourceVersionError(13, 14, 0)},
			},
		},
		{
			Name: "30. Compact past last write at ResourceVersion=15",
			Request: Request{
				Op:      OpCompact,
				Compact: CompactRequest{ResourceVersion: "15"},
			},
			CorrectResponse: Response{},
			InvalidResponses: []Response{
				{Err: storage.NewTooLargeResourceVersionError(15, 15, 0)},
			},
		},
		{
			Name: "31. Create pod4 after compact past last write returns success RV=17",
			Request: Request{
				Op:  OpCreate,
				Key: pod4Key,
				Create: CreateRequest{
					Object: pod4,
				},
			},
			CorrectResponse: Response{
				Object: withRV(pod4, "17"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Added,
				Object: withRV(pod4, "17"),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod4, "16")},
				{Object: withRV(pod4, "18")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "32. Create pod5 in ns10 returns success RV=18",
			Request: Request{
				Op:  OpCreate,
				Key: pod5Key,
				Create: CreateRequest{
					Object: pod5,
				},
			},
			CorrectResponse: Response{
				Object: withRV(pod5, "18"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Added,
				Object: withRV(pod5, "18"),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod5, "17")},
				{Object: withRV(pod5, "19")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "33. Update existing pod4 with ignoreNotFound=true validating ResponseMeta has pod4 RV, not store RV, returns success RV=19",
			Request: Request{
				Op:  OpUpdate,
				Key: pod4Key,
				Update: UpdateRequest{
					IgnoreNotFound: true,
					UpdateFunc: func(input runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
						// Creating pod5 moved the store to RV 18.
						if res.ResourceVersion != 17 {
							return nil, nil, fmt.Errorf("expected ResourceVersion 17, got %d", res.ResourceVersion)
						}
						pod := input.(*example.Pod).DeepCopy()
						pod.Labels = map[string]string{"version": "v1"}
						return pod, nil, nil
					},
				},
			},
			CorrectResponse: Response{
				Object: withLabel(pod4, "19", "version", "v1"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Modified,
				Object: withLabel(pod4, "19", "version", "v1"),
			},
			InvalidResponses: []Response{
				{Object: withLabel(pod4, "18", "version", "v1")},
				{Object: withLabel(pod4, "20", "version", "v1")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "34. Update pod4 with CachedExistingObject and failing UpdateFunc returns user error",
			Request: Request{
				Op:  OpUpdate,
				Key: pod4Key,
				Update: UpdateRequest{
					CachedExistingObject: withRV(pod4, "17"),
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
				{Object: withLabel(pod4, "19", "version", "v1")},
				{Object: withLabel(pod4, "20", "version", "v1")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "35. Update pod4 with stale CachedExistingObject where UpdateFunc fails on stale RV retries and returns success RV=20",
			Request: Request{
				Op:  OpUpdate,
				Key: pod4Key,
				Update: UpdateRequest{
					CachedExistingObject: withRV(pod4, "17"),
					UpdateFunc: func(input runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
						if res.ResourceVersion != 19 {
							return nil, nil, fmt.Errorf("expected ResourceVersion 19, got %d", res.ResourceVersion)
						}
						pod := input.(*example.Pod).DeepCopy()
						pod.Labels = map[string]string{"version": "v2"}
						return pod, nil, nil
					},
				},
			},
			CorrectResponse: Response{
				Object: withLabel(pod4, "20", "version", "v2"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Modified,
				Object: withLabel(pod4, "20", "version", "v2"),
			},
			InvalidResponses: []Response{
				{Object: withLabel(pod4, "19", "version", "v2")},
				{Object: withLabel(pod4, "21", "version", "v2")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "36. Delete pod5 with CachedExistingObject and matching ResourceVersion precondition returns success RV=21",
			Request: Request{
				Op:  OpDelete,
				Key: pod5Key,
				Delete: DeleteRequest{
					Preconditions:        &storage.Preconditions{ResourceVersion: &pod5RV18},
					CachedExistingObject: withRV(pod5, "18"),
				},
			},
			CorrectResponse: Response{
				Object: withRV(pod5, "21"),
			},
			ExpectedEvent: &watch.Event{
				Type:   watch.Deleted,
				Object: withRV(pod5, "21"),
			},
			InvalidResponses: []Response{
				{Object: &example.Pod{}, Err: storage.NewKeyNotFoundError(pod5Key, 0)},
				{Object: withRV(pod5, "18")},
				{Object: withRV(pod5, "20")},
				{Object: withRV(pod5, "22")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "37. Delete deleted pod5 with stale CachedExistingObject returns NotFound",
			Request: Request{
				Op:  OpDelete,
				Key: pod5Key,
				Delete: DeleteRequest{
					CachedExistingObject: withRV(pod5, "18"),
				},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    storage.NewKeyNotFoundError(pod5Key, 21),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod5, "21")},
				{Object: withRV(pod5, "22")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "38. Create pod1 with ResourceVersion set returns ErrResourceVersionSetOnCreate",
			Request: Request{
				Op:     OpCreate,
				Key:    pod1Key,
				Create: CreateRequest{Object: withRV(pod1, "5")},
			},
			CorrectResponse: Response{Err: storage.ErrResourceVersionSetOnCreate},
			InvalidResponses: []Response{
				{Object: withRV(pod1, "22")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "39. Create with empty key returns empty key error",
			Request: Request{
				Op:     OpCreate,
				Key:    "",
				Create: CreateRequest{Object: pod1},
			},
			CorrectResponse: Response{Err: fmt.Errorf("empty key: %q", "")},
			InvalidResponses: []Response{
				{Object: withRV(pod1, "22")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "40. Update with key containing . returns invalid key error",
			Request: Request{
				Op:  OpUpdate,
				Key: "/pods/./ns1/pod4",
				Update: UpdateRequest{
					UpdateFunc: storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) { return obj, nil }),
				},
			},
			CorrectResponse: Response{Err: fmt.Errorf("invalid key: %q", "/pods/./ns1/pod4")},
			InvalidResponses: []Response{
				{Object: nil, Err: storage.NewKeyNotFoundError("/pods/./ns1/pod4", 21)},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "41. Delete with key / returns empty key error",
			Request: Request{
				Op:  OpDelete,
				Key: "/",
			},
			CorrectResponse: Response{Err: fmt.Errorf("empty key: %q", "/")},
			InvalidResponses: []Response{
				{Object: nil, Err: storage.NewKeyNotFoundError("/", 21)},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "42. Compact at ResourceVersion=16",
			Request: Request{
				Op:      OpCompact,
				Compact: CompactRequest{ResourceVersion: "16"},
			},
			CorrectResponse: Response{},
			InvalidResponses: []Response{
				{Err: storage.NewTooLargeResourceVersionError(16, 21, 0)},
			},
		},
	}
}

func readTestCases() []testStep {
	pod1 := newTestPod("pod1", "ns1", types.UID("uid-1"), "")
	pod3 := newTestPod("pod3", "ns1", types.UID("uid-3"), "")
	pod4 := newTestPod("pod4", "ns1", types.UID("uid-4"), "")
	pod5 := newTestPod("pod5", "ns10", types.UID("uid-5"), "")
	pod1Key := mustGetKey(pod1)
	pod3Key := mustGetKey(pod3)
	pod4Key := mustGetKey(pod4)
	pod5Key := mustGetKey(pod5)
	_, invalidRVErr := storage.APIObjectVersioner{}.ParseResourceVersion("abc")
	listRecursive := ListRequest{Options: storage.ListOptions{Recursive: true, Predicate: storage.Everything}}
	listNonRecursive := ListRequest{Options: storage.ListOptions{Predicate: storage.Everything}}

	return []testStep{
		{
			Name:    "Get existing pod",
			Request: Request{Op: OpGet, Key: pod4Key},
			CorrectResponse: Response{
				Object: withLabel(pod4, "20", "version", "v2"),
			},
			InvalidResponses: []Response{
				{Object: &example.Pod{}, Err: storage.NewKeyNotFoundError(pod4Key, 0)},
				{Object: withLabel(pod4, "19", "version", "v2")},
				{Object: withLabel(pod4, "21", "version", "v2")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name:    "Get non-existing pod",
			Request: Request{Op: OpGet, Key: pod1Key},
			CorrectResponse: Response{
				Object: nil,
				Err:    storage.NewKeyNotFoundError(pod1Key, 22),
			},
			InvalidResponses: []Response{
				{Object: nil, Err: storage.NewKeyNotFoundError(pod1Key, 0)},
				{Object: withRV(pod1, "2")},
				{Object: withRV(pod1, "4")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name:    "Get existing pod with ignoreNotFound=true",
			Request: Request{Op: OpGet, Key: pod4Key, Get: GetRequest{Options: storage.GetOptions{IgnoreNotFound: true}}},
			CorrectResponse: Response{
				Object: withLabel(pod4, "20", "version", "v2"),
			},
			InvalidResponses: []Response{
				{Object: &example.Pod{}},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod4Key, 0)},
				{Object: nil, Err: nil},
			},
		},
		{
			Name:    "Get non-existing pod with ignoreNotFound=true",
			Request: Request{Op: OpGet, Key: pod5Key, Get: GetRequest{Options: storage.GetOptions{IgnoreNotFound: true}}},
			CorrectResponse: Response{
				Object: &example.Pod{},
			},
			InvalidResponses: []Response{
				{Object: withRV(pod5, "18")},
				{Object: withRV(pod5, "21")},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod5Key, 0)},
				{Object: nil, Err: nil},
			},
		},
		{
			Name:            "Get with empty key returns empty key error",
			Request:         Request{Op: OpGet, Key: ""},
			CorrectResponse: Response{Err: fmt.Errorf("empty key: %q", "")},
			InvalidResponses: []Response{
				{Object: nil, Err: storage.NewKeyNotFoundError("", 0)},
				{Object: nil, Err: nil},
			},
		},
		{
			Name:            "Get with key escaping the prefix returns invalid key error",
			Request:         Request{Op: OpGet, Key: "/pods/../secrets/s1"},
			CorrectResponse: Response{Err: fmt.Errorf("invalid key: %q", "/pods/../secrets/s1")},
			InvalidResponses: []Response{
				{Object: nil, Err: storage.NewKeyNotFoundError("/pods/../secrets/s1", 0)},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "Get with unparsable ResourceVersion returns invalid error",
			Request: Request{
				Op:  OpGet,
				Key: pod4Key,
				Get: GetRequest{Options: storage.GetOptions{
					ResourceVersion: "abc",
				}},
			},
			CorrectResponse: Response{Err: invalidRVErr},
			InvalidResponses: []Response{
				{Object: withLabel(pod4, "20", "version", "v2")},
				{Object: nil, Err: apierrors.NewBadRequest(fmt.Sprintf("invalid resource version: %v", invalidRVErr))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "Get pod4 with ResourceVersion=19 returns pod4 at RV>=19",
			Request: Request{
				Op:  OpGet,
				Key: pod4Key,
				Get: GetRequest{Options: storage.GetOptions{ResourceVersion: "19"}},
			},
			CorrectResponse: Response{
				Object: withLabel(pod4, "19", "version", "v1"),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod4, "17")},
				{Object: withLabel(pod4, "23", "version", "v2")},
				{Object: &example.Pod{}},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod4Key, 0)},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod4Key, 22)},
				{Object: nil, Err: storage.NewTooLargeResourceVersionError(19, 22, 0)},
				{Object: nil, Err: apierrors.NewResourceExpired("too old resource version: 19 (22)")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "Get pod4 with compacted ResourceVersion=12 returns pod4 at RV>=16",
			Request: Request{
				Op:  OpGet,
				Key: pod4Key,
				Get: GetRequest{Options: storage.GetOptions{ResourceVersion: "12"}},
			},
			CorrectResponse: Response{
				Object: withRV(pod4, "17"),
			},
			InvalidResponses: []Response{
				{Object: withLabel(pod4, "23", "version", "v2")},
				{Object: &example.Pod{}},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod4Key, 0)},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod4Key, 22)},
				{Object: nil, Err: storage.NewTooLargeResourceVersionError(12, 22, 0)},
				{Object: nil, Err: apierrors.NewResourceExpired("too old resource version: 12 (16)")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "Get pod4 with ResourceVersion=0 and ignoreNotFound=true returns pod4 or empty pod",
			Request: Request{
				Op:  OpGet,
				Key: pod4Key,
				Get: GetRequest{Options: storage.GetOptions{ResourceVersion: "0", IgnoreNotFound: true}},
			},
			CorrectResponse: Response{
				Object: withRV(pod4, "17"),
			},
			InvalidResponses: []Response{
				{Object: withLabel(pod4, "23", "version", "v2")},
				{Object: withRV(pod1, "2")},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod4Key, 0)},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod4Key, 22)},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "Get deleted pod5 with ResourceVersion=21 returns KeyNotFoundError",
			Request: Request{
				Op:  OpGet,
				Key: pod5Key,
				Get: GetRequest{Options: storage.GetOptions{ResourceVersion: "21"}},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    storage.NewKeyNotFoundError(pod5Key, 21),
			},
			InvalidResponses: []Response{
				{Object: withRV(pod5, "18")},
				{Object: &example.Pod{}},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod5Key, 0)},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod5Key, 18)},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod5Key, 20)},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod5Key, 23)},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with Exact ResourceVersion=22",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "22",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            true,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("21", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: apierrors.NewResourceExpired("too old resource version: 22 (22)")},
				{Object: nil, Err: storage.NewTooLargeResourceVersionError(22, 22, 0)},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "Get pod4 with future ResourceVersion=99 returns TooLargeResourceVersionError",
			Request: Request{
				Op:  OpGet,
				Key: pod4Key,
				Get: GetRequest{Options: storage.GetOptions{ResourceVersion: "99"}},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    storage.NewTooLargeResourceVersionError(99, 22, 0),
			},
			InvalidResponses: []Response{
				{Object: withLabel(pod4, "20", "version", "v2")},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod4Key, 0)},
				{Object: nil, Err: storage.NewKeyNotFoundError(pod4Key, 22)},
				{Object: nil, Err: nil},
			},
		},
		{
			Name:    "List all pods",
			Request: Request{Op: OpList, Key: "/pods/", List: listRecursive},
			CorrectResponse: Response{
				Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("21", withLabel(pod4, "20", "version", "v2"))},
				{Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2"), withRV(pod5, "18"))},
				{Object: newTestPodList("22")},
				{Object: nil, Err: apierrors.NewResourceExpired("too old resource version")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name:    "List pods in namespace ns1",
			Request: Request{Op: OpList, Key: "/pods/ns1", List: listRecursive},
			CorrectResponse: Response{
				Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2"), withRV(pod5, "18"))},
				{Object: newTestPodList("21", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name:    "List pods in empty namespace ns10",
			Request: Request{Op: OpList, Key: "/pods/ns10", List: listRecursive},
			CorrectResponse: Response{
				Object: newTestPodList("22"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("21")},
				{Object: newTestPodList("22", withRV(pod5, "18"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name:    "List existing pod non-recursively",
			Request: Request{Op: OpList, Key: pod4Key, List: listNonRecursive},
			CorrectResponse: Response{
				Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("21", withLabel(pod4, "20", "version", "v2"))},
				{Object: newTestPodList("22")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name:    "List non-existing pod non-recursively",
			Request: Request{Op: OpList, Key: pod3Key, List: listNonRecursive},
			CorrectResponse: Response{
				Object: newTestPodList("22"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("22", withLabel(pod3, "12", "version", "v7"))},
				{Object: newTestPodList("16")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name:    "List namespace non-recursively",
			Request: Request{Op: OpList, Key: "/pods/ns1", List: listNonRecursive},
			CorrectResponse: Response{
				Object: newTestPodList("22"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name:    "List cluster root non-recursively",
			Request: Request{Op: OpList, Key: "/pods/", List: listNonRecursive},
			CorrectResponse: Response{
				Object: newTestPodList("22"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name:    "List existing pod recursively returns empty list",
			Request: Request{Op: OpList, Key: pod4Key, List: listRecursive},
			CorrectResponse: Response{
				Object: newTestPodList("22"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List existing pod non-recursively with ResourceVersion=0",
			Request: Request{
				Op:  OpList,
				Key: pod4Key,
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion: "0",
					Recursive:       false,
					Predicate:       storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("17", withRV(pod4, "17")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("0")},
				{Object: newTestPodList("17")},
				{Object: newTestPodList("23", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List non-existing pod non-recursively with ResourceVersion=0",
			Request: Request{
				Op:  OpList,
				Key: pod3Key,
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion: "0",
					Recursive:       false,
					Predicate:       storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("17"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("17", withLabel(pod3, "12", "version", "v7"))},
				{Object: newTestPodList("23")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List namespace non-recursively with ResourceVersion=0",
			Request: Request{
				Op:  OpList,
				Key: "/pods/ns1",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion: "0",
					Recursive:       false,
					Predicate:       storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("17"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("17", withRV(pod4, "17"))},
				{Object: newTestPodList("23")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List cluster root non-recursively with ResourceVersion=0",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion: "0",
					Recursive:       false,
					Predicate:       storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("17"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("17", withRV(pod4, "17"))},
				{Object: newTestPodList("23")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List existing pod non-recursively with Exact ResourceVersion=18",
			Request: Request{
				Op:  OpList,
				Key: pod5Key,
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "18",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            false,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("18", withRV(pod5, "18")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("18")},
				{Object: newTestPodList("18", withRV(pod4, "17"), withRV(pod5, "18"))},
				{Object: newTestPodList("22", withRV(pod5, "18"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List non-existing pod non-recursively with Exact ResourceVersion=16",
			Request: Request{
				Op:  OpList,
				Key: pod3Key,
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "16",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            false,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("16"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("16", withLabel(pod3, "12", "version", "v7"))},
				{Object: newTestPodList("22")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List namespace non-recursively with Exact ResourceVersion=18",
			Request: Request{
				Op:  OpList,
				Key: "/pods/ns1",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "18",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            false,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("18"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("18", withRV(pod4, "17"))},
				{Object: newTestPodList("22")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List cluster root non-recursively with Exact ResourceVersion=18",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "18",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            false,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("18"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("18", withRV(pod4, "17"), withRV(pod5, "18"))},
				{Object: newTestPodList("22")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pod non-recursively with compacted Exact ResourceVersion=12 returns ResourceExpired",
			Request: Request{
				Op:  OpList,
				Key: pod3Key,
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "12",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            false,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    apierrors.NewResourceExpired("The resourceVersion for the provided list is too old."),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("12", withLabel(pod3, "12", "version", "v7"))},
				{Object: newTestPodList("22")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List existing pod non-recursively with NotOlderThan ResourceVersion=18",
			Request: Request{
				Op:  OpList,
				Key: pod4Key,
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "18",
					ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
					Recursive:            false,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("19", withLabel(pod4, "19", "version", "v1")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("17", withRV(pod4, "17"))},
				{Object: newTestPodList("19", withLabel(pod4, "19", "version", "v1"), withRV(pod5, "18"))},
				{Object: newTestPodList("23", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List existing pod non-recursively with NotOlderThan compacted ResourceVersion=12",
			Request: Request{
				Op:  OpList,
				Key: pod4Key,
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "12",
					ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
					Recursive:            false,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("21", withLabel(pod4, "20", "version", "v2")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("12")},
				{Object: newTestPodList("23", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: apierrors.NewResourceExpired("The resourceVersion for the provided list is too old.")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List namespace non-recursively with NotOlderThan ResourceVersion=18",
			Request: Request{
				Op:  OpList,
				Key: "/pods/ns1",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "18",
					ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
					Recursive:            false,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("19"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("17")},
				{Object: newTestPodList("19", withLabel(pod4, "19", "version", "v1"))},
				{Object: newTestPodList("23")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List cluster root non-recursively with NotOlderThan ResourceVersion=18",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "18",
					ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
					Recursive:            false,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("19"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("17")},
				{Object: newTestPodList("19", withLabel(pod4, "19", "version", "v1"), withRV(pod5, "18"))},
				{Object: newTestPodList("23")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pod non-recursively with future ResourceVersion=99 returns TooLargeResourceVersionError",
			Request: Request{
				Op:  OpList,
				Key: pod4Key,
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "99",
					ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
					Recursive:            false,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    storage.NewTooLargeResourceVersionError(99, 22, 0),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List existing pod non-recursively with matching label selector",
			Request: Request{
				Op:  OpList,
				Key: pod4Key,
				List: ListRequest{Options: storage.ListOptions{
					Recursive: false,
					Predicate: storage.SelectionPredicate{
						Label:    labels.SelectorFromSet(labels.Set{"version": "v2"}),
						Field:    fields.Everything(),
						GetAttrs: storage.DefaultNamespaceScopedAttr,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("22")},
				{Object: newTestPodList("21", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List existing pod non-recursively with non-matching label selector",
			Request: Request{
				Op:  OpList,
				Key: pod4Key,
				List: ListRequest{Options: storage.ListOptions{
					Recursive: false,
					Predicate: storage.SelectionPredicate{
						Label:    labels.SelectorFromSet(labels.Set{"version": "v1"}),
						Field:    fields.Everything(),
						GetAttrs: storage.DefaultNamespaceScopedAttr,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("22"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List existing pod non-recursively with matching field selector",
			Request: Request{
				Op:  OpList,
				Key: pod4Key,
				List: ListRequest{Options: storage.ListOptions{
					Recursive: false,
					Predicate: storage.SelectionPredicate{
						Label:    labels.Everything(),
						Field:    fields.OneTermEqualSelector("metadata.name", "pod4"),
						GetAttrs: storage.DefaultNamespaceScopedAttr,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("22")},
				{Object: newTestPodList("21", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List existing pod non-recursively with non-matching field selector",
			Request: Request{
				Op:  OpList,
				Key: pod4Key,
				List: ListRequest{Options: storage.ListOptions{
					Recursive: false,
					Predicate: storage.SelectionPredicate{
						Label:    labels.Everything(),
						Field:    fields.OneTermEqualSelector("metadata.name", "pod5"),
						GetAttrs: storage.DefaultNamespaceScopedAttr,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("22"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List existing pod non-recursively with Exact ResourceVersion=19 and matching label selector",
			Request: Request{
				Op:  OpList,
				Key: pod4Key,
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "19",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            false,
					Predicate: storage.SelectionPredicate{
						Label:    labels.SelectorFromSet(labels.Set{"version": "v1"}),
						Field:    fields.Everything(),
						GetAttrs: storage.DefaultNamespaceScopedAttr,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("19", withLabel(pod4, "19", "version", "v1")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("19")},
				{Object: newTestPodList("22")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List existing pod non-recursively with Exact ResourceVersion=19 and matching field selector",
			Request: Request{
				Op:  OpList,
				Key: pod4Key,
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "19",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            false,
					Predicate: storage.SelectionPredicate{
						Label:    labels.Everything(),
						Field:    fields.OneTermEqualSelector("metadata.namespace", "ns1"),
						GetAttrs: storage.DefaultNamespaceScopedAttr,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("19", withLabel(pod4, "19", "version", "v1")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("19")},
				{Object: newTestPodList("22")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List existing pod non-recursively with Limit=1",
			Request: Request{
				Op:  OpList,
				Key: pod4Key,
				List: ListRequest{Options: storage.ListOptions{
					Recursive: false,
					Predicate: storage.SelectionPredicate{
						Label: labels.Everything(),
						Field: fields.Everything(),
						Limit: 1,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodListWithContinue("22", new(int64(0)), withLabel(pod4, "20", "version", "v2"))},
				{Object: newTestPodList("22")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List existing pod non-recursively with Exact ResourceVersion=18 and Limit=1",
			Request: Request{
				Op:  OpList,
				Key: pod4Key,
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "18",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            false,
					Predicate: storage.SelectionPredicate{
						Label: labels.Everything(),
						Field: fields.Everything(),
						Limit: 1,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("18", withRV(pod4, "17")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodListWithContinue("18", new(int64(1)), withRV(pod4, "17"))},
				{Object: newTestPodList("18")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List existing pod non-recursively with legacy ResourceVersion=18 and Limit=1",
			Request: Request{
				Op:  OpList,
				Key: pod4Key,
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion: "18",
					Recursive:       false,
					Predicate: storage.SelectionPredicate{
						Label: labels.Everything(),
						Field: fields.Everything(),
						Limit: 1,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("17", withRV(pod4, "17"))},
				{Object: newTestPodListWithContinue("18", new(int64(1)), withRV(pod4, "17"))},
				{Object: newTestPodListWithContinue("22", new(int64(0)), withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with Exact ResourceVersion=16 when none exist",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "16",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            true,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("16"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("12")},
				{Object: newTestPodList("16", withLabel(pod3, "12", "version", "v7"))},
				{Object: nil, Err: apierrors.NewResourceExpired("The resourceVersion for the provided list is too old.")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with compacted Exact ResourceVersion=15 returns ResourceExpired",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "15",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            true,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    apierrors.NewResourceExpired("The resourceVersion for the provided list is too old."),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("15")},
				{Object: nil, Err: storage.NewTooLargeResourceVersionError(15, 22, 0)},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with Exact ResourceVersion=18",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "18",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            true,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("18", withRV(pod4, "17"), withRV(pod5, "18")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("17", withRV(pod4, "17"))},
				{Object: newTestPodList("19", withLabel(pod4, "19", "version", "v1"), withRV(pod5, "18"))},
				{Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2"))},
				{Object: newTestPodList("18", withRV(pod4, "17"))},
				{Object: newTestPodList("18", withRV(pod5, "18"), withRV(pod4, "17"))},
				{Object: newTestPodList("18", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: storage.NewTooLargeResourceVersionError(18, 22, 0)},
				{Object: nil, Err: apierrors.NewResourceExpired("The resourceVersion for the provided list is too old.")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with compacted Exact ResourceVersion=12 returns ResourceExpired",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "12",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            true,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    apierrors.NewResourceExpired("The resourceVersion for the provided list is too old."),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("12", withLabel(pod3, "12", "version", "v7"))},
				{Object: newTestPodList("12", withLabel(pod3, "12", "version", "v7")), Err: apierrors.NewResourceExpired("The resourceVersion for the provided list is too old.")},
				{Object: nil, Err: storage.NewKeyNotFoundError("/pods/", 0)},
				{Object: nil, Err: storage.NewTooLargeResourceVersionError(12, 22, 0)},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with Exact ResourceVersion=21",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "21",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            true,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("21", withLabel(pod4, "20", "version", "v2")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("20", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: apierrors.NewResourceExpired("too old resource version: 21 (22)")},
				{Object: nil, Err: storage.NewTooLargeResourceVersionError(21, 22, 0)},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List ns1 with Exact ResourceVersion=18",
			Request: Request{
				Op:  OpList,
				Key: "/pods/ns1",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "18",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            true,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("18", withRV(pod4, "17")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("18", withRV(pod4, "17"), withRV(pod5, "18"))},
				{Object: newTestPodList("17", withRV(pod4, "17"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with NotOlderThan ResourceVersion=18",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "18",
					ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
					Recursive:            true,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("19", withLabel(pod4, "19", "version", "v1"), withRV(pod5, "18")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("17", withRV(pod4, "17"))},
				{Object: newTestPodList("23", withLabel(pod4, "20", "version", "v2"))},
				{Object: newTestPodList("19", withRV(pod4, "17"), withRV(pod5, "18"))},
				{Object: nil, Err: storage.NewTooLargeResourceVersionError(18, 22, 0)},
				{Object: nil, Err: apierrors.NewResourceExpired("too old resource version: 18 (22)")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with NotOlderThan compacted ResourceVersion=12",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "12",
					ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
					Recursive:            true,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("21", withLabel(pod4, "20", "version", "v2")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("12", withLabel(pod3, "12", "version", "v7"))},
				{Object: newTestPodList("23", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: storage.NewTooLargeResourceVersionError(12, 22, 0)},
				{Object: nil, Err: apierrors.NewResourceExpired("The resourceVersion for the provided list is too old.")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with ResourceVersion=0",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion: "0",
					Recursive:       true,
					Predicate:       storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("17", withRV(pod4, "17")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("0")},
				{Object: newTestPodList("23", withLabel(pod4, "20", "version", "v2"))},
				{Object: newTestPodList("18", withRV(pod4, "17"))},
				{Object: nil, Err: apierrors.NewResourceExpired("too old resource version: 0 (22)")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with future ResourceVersion=99 returns TooLargeResourceVersionError",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "99",
					ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
					Recursive:            true,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{
				Object: nil,
				Err:    storage.NewTooLargeResourceVersionError(99, 22, 0),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2"))},
				{Object: newTestPodList("99", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: storage.NewKeyNotFoundError("/pods/", 0)},
				{Object: nil, Err: nil},
			},
		},
		{
			Name:            "List with key ending with .. returns invalid key error",
			Request:         Request{Op: OpList, Key: "/pods/..", List: listRecursive},
			CorrectResponse: Response{Err: fmt.Errorf("invalid key: %q", "/pods/..")},
			InvalidResponses: []Response{
				{Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List with unparsable ResourceVersion returns bad request",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion: "abc",
					Recursive:       true,
					Predicate:       storage.Everything,
				}},
			},
			CorrectResponse: Response{Err: apierrors.NewBadRequest(fmt.Sprintf("invalid resource version: %v", invalidRVErr))},
			InvalidResponses: []Response{
				{Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List with unknown ResourceVersionMatch returns error",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "18",
					ResourceVersionMatch: "Newest",
					Recursive:            true,
					Predicate:            storage.Everything,
				}},
			},
			CorrectResponse: Response{Err: fmt.Errorf("unknown ResourceVersionMatch value: %v", "Newest")},
			InvalidResponses: []Response{
				{Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with matching label selector",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					Recursive: true,
					Predicate: storage.SelectionPredicate{
						Label:    labels.SelectorFromSet(labels.Set{"version": "v2"}),
						Field:    fields.Everything(),
						GetAttrs: storage.DefaultNamespaceScopedAttr,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("22")},
				{Object: newTestPodList("21", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with non-matching label selector",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					Recursive: true,
					Predicate: storage.SelectionPredicate{
						Label:    labels.SelectorFromSet(labels.Set{"version": "v1"}),
						Field:    fields.Everything(),
						GetAttrs: storage.DefaultNamespaceScopedAttr,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("22"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2"))},
				{Object: newTestPodList("22", withLabel(pod4, "19", "version", "v1"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with matching field selector",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					Recursive: true,
					Predicate: storage.SelectionPredicate{
						Label:    labels.Everything(),
						Field:    fields.OneTermEqualSelector("metadata.name", "pod4"),
						GetAttrs: storage.DefaultNamespaceScopedAttr,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("22")},
				{Object: newTestPodList("21", withLabel(pod4, "20", "version", "v2"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with non-matching field selector",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					Recursive: true,
					Predicate: storage.SelectionPredicate{
						Label:    labels.Everything(),
						Field:    fields.OneTermEqualSelector("metadata.name", "pod5"),
						GetAttrs: storage.DefaultNamespaceScopedAttr,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("22"),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2"))},
				{Object: newTestPodList("22", withRV(pod5, "18"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with Exact ResourceVersion=19 and label selector",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "19",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            true,
					Predicate: storage.SelectionPredicate{
						Label:    labels.SelectorFromSet(labels.Set{"version": "v1"}),
						Field:    fields.Everything(),
						GetAttrs: storage.DefaultNamespaceScopedAttr,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("19", withLabel(pod4, "19", "version", "v1")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("19", withLabel(pod4, "19", "version", "v1"), withRV(pod5, "18"))},
				{Object: newTestPodList("19")},
				{Object: newTestPodList("22")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with Exact ResourceVersion=19 and field selector",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "19",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            true,
					Predicate: storage.SelectionPredicate{
						Label:    labels.Everything(),
						Field:    fields.OneTermEqualSelector("metadata.namespace", "ns10"),
						GetAttrs: storage.DefaultNamespaceScopedAttr,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("19", withRV(pod5, "18")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("19", withLabel(pod4, "19", "version", "v1"), withRV(pod5, "18"))},
				{Object: newTestPodList("19")},
				{Object: newTestPodList("22")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with Limit=1 when one pod exists",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion: "",
					Recursive:       true,
					Predicate: storage.SelectionPredicate{
						Label: labels.Everything(),
						Field: fields.Everything(),
						Limit: 1,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("22", withLabel(pod4, "20", "version", "v2")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodListWithContinue("22", new(int64(0)), withLabel(pod4, "20", "version", "v2"))},
				{Object: newTestPodList("22")},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with Exact ResourceVersion=18 and Limit=1",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "18",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            true,
					Predicate: storage.SelectionPredicate{
						Label: labels.Everything(),
						Field: fields.Everything(),
						Limit: 1,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodListWithContinue("18", new(int64(1)), withRV(pod4, "17")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("18", withRV(pod4, "17"))},
				{Object: newTestPodListWithContinue("18", new(int64(2)), withRV(pod4, "17"))},
				{Object: newTestPodListWithContinue("18", nil, withRV(pod4, "17"))},
				{Object: newTestPodListWithContinue("18", new(int64(1)), withRV(pod4, "17"), withRV(pod5, "18"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with legacy ResourceVersion=18 and Limit=1",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion: "18",
					Recursive:       true,
					Predicate: storage.SelectionPredicate{
						Label: labels.Everything(),
						Field: fields.Everything(),
						Limit: 1,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodListWithContinue("18", new(int64(1)), withRV(pod4, "17")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("18", withRV(pod4, "17"))},
				{Object: newTestPodListWithContinue("18", new(int64(2)), withRV(pod4, "17"))},
				{Object: newTestPodListWithContinue("18", nil, withRV(pod4, "17"))},
				{Object: newTestPodListWithContinue("18", new(int64(1)), withRV(pod4, "17"), withRV(pod5, "18"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with Exact ResourceVersion=19, Limit=1, and field selector matching second pod",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "19",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            true,
					Predicate: storage.SelectionPredicate{
						Label:    labels.Everything(),
						Field:    fields.OneTermEqualSelector("metadata.namespace", "ns10"),
						GetAttrs: storage.DefaultNamespaceScopedAttr,
						Limit:    1,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("19", withRV(pod5, "18")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("19")},
				{Object: newTestPodListWithContinue("19", new(int64(1)), withRV(pod5, "18"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods without trailing slash with Exact ResourceVersion=18 and Limit=1",
			Request: Request{
				Op:  OpList,
				Key: "/pods",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "18",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            true,
					Predicate: storage.SelectionPredicate{
						Label: labels.Everything(),
						Field: fields.Everything(),
						Limit: 1,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodListWithContinue("18", new(int64(1)), withRV(pod4, "17")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("18", withRV(pod4, "17"))},
				{Object: newTestPodListWithContinue("18", nil, withRV(pod4, "17"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with NotOlderThan ResourceVersion=18, Limit=1, and field selector matching first pod",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "18",
					ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
					Recursive:            true,
					Predicate: storage.SelectionPredicate{
						Label:    labels.Everything(),
						Field:    fields.OneTermEqualSelector("metadata.namespace", "ns1"),
						GetAttrs: storage.DefaultNamespaceScopedAttr,
						Limit:    1,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodList("19", withLabel(pod4, "19", "version", "v1")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("19")},
				{Object: newTestPodListWithContinue("19", new(int64(1)), withLabel(pod4, "19", "version", "v1"))},
				{Object: newTestPodList("19", withLabel(pod4, "19", "version", "v1"), withRV(pod5, "18"))},
				{Object: nil, Err: nil},
			},
		},
		{
			Name: "List pods with Exact ResourceVersion=19, Limit=1, and field selector matching both pods",
			Request: Request{
				Op:  OpList,
				Key: "/pods/",
				List: ListRequest{Options: storage.ListOptions{
					ResourceVersion:      "19",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					Recursive:            true,
					Predicate: storage.SelectionPredicate{
						Label:    labels.Everything(),
						Field:    fields.OneTermNotEqualSelector("metadata.namespace", "ns2"),
						GetAttrs: storage.DefaultNamespaceScopedAttr,
						Limit:    1,
					},
				}},
			},
			CorrectResponse: Response{
				Object: newTestPodListWithContinue("19", nil, withLabel(pod4, "19", "version", "v1")),
			},
			InvalidResponses: []Response{
				{Object: newTestPodList("19", withLabel(pod4, "19", "version", "v1"))},
				{Object: newTestPodListWithContinue("19", new(int64(1)), withLabel(pod4, "19", "version", "v1"))},
				{Object: newTestPodListWithContinue("19", nil, withLabel(pod4, "19", "version", "v1"), withRV(pod5, "18"))},
				{Object: nil, Err: nil},
			},
		},
	}
}

type watchTestCase struct {
	Name        string
	Request     WatchRequest
	ExpectMaxRV uint64
	ExpectError error
}

func watchTestCases() []watchTestCase {
	_, invalidRVErr := storage.APIObjectVersioner{}.ParseResourceVersion("abc")
	watch := func(rv string, match metav1.ResourceVersionMatch) storage.ListOptions {
		pred := storage.Everything
		pred.AllowWatchBookmarks = false
		return storage.ListOptions{ResourceVersion: rv, ResourceVersionMatch: match, Predicate: pred, Recursive: true, SendInitialEvents: new(false)}
	}
	watchWithBookmarks := func(rv string, match metav1.ResourceVersionMatch) storage.ListOptions {
		opts := watch(rv, match)
		opts.Predicate.AllowWatchBookmarks = true
		return opts
	}
	watchNonRecursive := func(rv string, match metav1.ResourceVersionMatch) storage.ListOptions {
		opts := watch(rv, match)
		opts.Recursive = false
		return opts
	}
	watchNonRecursiveWithBookmarks := func(rv string, match metav1.ResourceVersionMatch) storage.ListOptions {
		opts := watchWithBookmarks(rv, match)
		opts.Recursive = false
		return opts
	}
	watchList := func(rv string) storage.ListOptions {
		pred := storage.Everything
		pred.AllowWatchBookmarks = true
		return storage.ListOptions{ResourceVersion: rv, ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan, Predicate: pred, Recursive: true, SendInitialEvents: new(true)}
	}
	watchListNonRecursive := func(rv string) storage.ListOptions {
		opts := watchList(rv)
		opts.Recursive = false
		return opts
	}
	return []watchTestCase{
		{
			Name:        "Watch everything",
			Request:     WatchRequest{Key: "/pods/", Options: watch("", "")},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch everything with AllowWatchBookmarks",
			Request:     WatchRequest{Key: "/pods/", Options: watchWithBookmarks("", "")},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch everything with NotOlderThan",
			Request:     WatchRequest{Key: "/pods/", Options: watch("", metav1.ResourceVersionMatchNotOlderThan)},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch everything with ResourceVersion=0",
			Request:     WatchRequest{Key: "/pods/", Options: watch("0", "")},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch everything with ResourceVersion=0 and AllowWatchBookmarks",
			Request:     WatchRequest{Key: "/pods/", Options: watchWithBookmarks("0", "")},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch everything with ResourceVersion=0 and NotOlderThan",
			Request:     WatchRequest{Key: "/pods/", Options: watch("0", metav1.ResourceVersionMatchNotOlderThan)},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch everything with ResourceVersion=1",
			Request:     WatchRequest{Key: "/pods/", Options: watch("1", "")},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch everything with ResourceVersion=1 and AllowWatchBookmarks",
			Request:     WatchRequest{Key: "/pods/", Options: watchWithBookmarks("1", "")},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch everything with ResourceVersion=1 and Exact",
			Request:     WatchRequest{Key: "/pods/", Options: watch("1", metav1.ResourceVersionMatchExact)},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch everything with ResourceVersion=1 and NotOlderThan",
			Request:     WatchRequest{Key: "/pods/", Options: watch("1", metav1.ResourceVersionMatchNotOlderThan)},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch everything with ResourceVersion=8",
			Request:     WatchRequest{Key: "/pods/", Options: watch("8", "")},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch everything with ResourceVersion=8 and AllowWatchBookmarks",
			Request:     WatchRequest{Key: "/pods/", Options: watchWithBookmarks("8", "")},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch everything with ResourceVersion=8 and Exact",
			Request:     WatchRequest{Key: "/pods/", Options: watch("8", metav1.ResourceVersionMatchExact)},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch everything with ResourceVersion=8 and NotOlderThan",
			Request:     WatchRequest{Key: "/pods/", Options: watch("8", metav1.ResourceVersionMatchNotOlderThan)},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch on namespace ns1 with ResourceVersion=1",
			Request:     WatchRequest{Key: "/pods/ns1/", Options: watch("1", "")},
			ExpectMaxRV: 20,
		},
		{
			Name:        "Watch on namespace ns1 with ResourceVersion=1 and Exact",
			Request:     WatchRequest{Key: "/pods/ns1/", Options: watch("1", metav1.ResourceVersionMatchExact)},
			ExpectMaxRV: 20,
		},
		{
			Name:        "Watch on namespace ns1 with ResourceVersion=1 and NotOlderThan",
			Request:     WatchRequest{Key: "/pods/ns1/", Options: watch("1", metav1.ResourceVersionMatchNotOlderThan)},
			ExpectMaxRV: 20,
		},
		{
			Name:        "Watch on namespace ns10 with ResourceVersion=1",
			Request:     WatchRequest{Key: "/pods/ns10/", Options: watch("1", "")},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch on namespace ns10 with ResourceVersion=1 and Exact",
			Request:     WatchRequest{Key: "/pods/ns10/", Options: watch("1", metav1.ResourceVersionMatchExact)},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch on namespace ns10 with ResourceVersion=1 and NotOlderThan",
			Request:     WatchRequest{Key: "/pods/ns10/", Options: watch("1", metav1.ResourceVersionMatchNotOlderThan)},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch list everything",
			Request:     WatchRequest{Key: "/pods/", Options: watchList("")},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch list everything with ResourceVersion=0",
			Request:     WatchRequest{Key: "/pods/", Options: watchList("0")},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch list everything with ResourceVersion=1",
			Request:     WatchRequest{Key: "/pods/", Options: watchList("1")},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch list on namespace ns1 with ResourceVersion=1",
			Request:     WatchRequest{Key: "/pods/ns1/", Options: watchList("1")},
			ExpectMaxRV: 20,
		},
		{
			Name:        "Watch list on namespace ns10 with ResourceVersion=1",
			Request:     WatchRequest{Key: "/pods/ns10/", Options: watchList("1")},
			ExpectMaxRV: 21,
		},
		{
			Name:        "Watch single pod pod1 non-recursively",
			Request:     WatchRequest{Key: "/pods/ns1/pod1", Options: watchNonRecursive("", "")},
			ExpectMaxRV: 4,
		},
		{
			Name:        "Watch single pod pod1 non-recursively with AllowWatchBookmarks",
			Request:     WatchRequest{Key: "/pods/ns1/pod1", Options: watchNonRecursiveWithBookmarks("", "")},
			ExpectMaxRV: 4,
		},
		{
			Name:        "Watch single pod pod1 non-recursively with NotOlderThan",
			Request:     WatchRequest{Key: "/pods/ns1/pod1", Options: watchNonRecursive("", metav1.ResourceVersionMatchNotOlderThan)},
			ExpectMaxRV: 4,
		},
		{
			Name:        "Watch single pod pod1 non-recursively with ResourceVersion=0",
			Request:     WatchRequest{Key: "/pods/ns1/pod1", Options: watchNonRecursive("0", "")},
			ExpectMaxRV: 4,
		},
		{
			Name:        "Watch single pod pod1 non-recursively with ResourceVersion=0 and AllowWatchBookmarks",
			Request:     WatchRequest{Key: "/pods/ns1/pod1", Options: watchNonRecursiveWithBookmarks("0", "")},
			ExpectMaxRV: 4,
		},
		{
			Name:        "Watch single pod pod1 non-recursively with ResourceVersion=0 and NotOlderThan",
			Request:     WatchRequest{Key: "/pods/ns1/pod1", Options: watchNonRecursive("0", metav1.ResourceVersionMatchNotOlderThan)},
			ExpectMaxRV: 4,
		},
		{
			Name:        "Watch single pod pod4 non-recursively with ResourceVersion=1",
			Request:     WatchRequest{Key: "/pods/ns1/pod4", Options: watchNonRecursive("1", "")},
			ExpectMaxRV: 20,
		},
		{
			Name:        "Watch single pod pod4 non-recursively with ResourceVersion=1 and AllowWatchBookmarks",
			Request:     WatchRequest{Key: "/pods/ns1/pod4", Options: watchNonRecursiveWithBookmarks("1", "")},
			ExpectMaxRV: 20,
		},
		{
			Name:        "Watch single pod pod4 non-recursively with ResourceVersion=1 and Exact",
			Request:     WatchRequest{Key: "/pods/ns1/pod4", Options: watchNonRecursive("1", metav1.ResourceVersionMatchExact)},
			ExpectMaxRV: 20,
		},
		{
			Name:        "Watch single pod pod4 non-recursively with ResourceVersion=1 and NotOlderThan",
			Request:     WatchRequest{Key: "/pods/ns1/pod4", Options: watchNonRecursive("1", metav1.ResourceVersionMatchNotOlderThan)},
			ExpectMaxRV: 20,
		},
		{
			Name:        "Watch single pod pod4 non-recursively with ResourceVersion=8",
			Request:     WatchRequest{Key: "/pods/ns1/pod4", Options: watchNonRecursive("8", "")},
			ExpectMaxRV: 20,
		},
		{
			Name:        "Watch single pod pod4 non-recursively with ResourceVersion=8 and AllowWatchBookmarks",
			Request:     WatchRequest{Key: "/pods/ns1/pod4", Options: watchNonRecursiveWithBookmarks("8", "")},
			ExpectMaxRV: 20,
		},
		{
			Name:        "Watch single pod pod4 non-recursively with ResourceVersion=8 and Exact",
			Request:     WatchRequest{Key: "/pods/ns1/pod4", Options: watchNonRecursive("8", metav1.ResourceVersionMatchExact)},
			ExpectMaxRV: 20,
		},
		{
			Name:        "Watch single pod pod4 non-recursively with ResourceVersion=8 and NotOlderThan",
			Request:     WatchRequest{Key: "/pods/ns1/pod4", Options: watchNonRecursive("8", metav1.ResourceVersionMatchNotOlderThan)},
			ExpectMaxRV: 20,
		},
		{
			Name:        "Watch list single pod pod1 non-recursively",
			Request:     WatchRequest{Key: "/pods/ns1/pod1", Options: watchListNonRecursive("")},
			ExpectMaxRV: 4,
		},
		{
			Name:        "Watch list single pod pod1 non-recursively with ResourceVersion=0",
			Request:     WatchRequest{Key: "/pods/ns1/pod1", Options: watchListNonRecursive("0")},
			ExpectMaxRV: 4,
		},
		{
			Name:        "Watch list single pod pod4 non-recursively with ResourceVersion=1",
			Request:     WatchRequest{Key: "/pods/ns1/pod4", Options: watchListNonRecursive("1")},
			ExpectMaxRV: 20,
		},
		{
			Name:        "Watch with empty key returns empty key error",
			Request:     WatchRequest{Key: "", Options: watch("1", "")},
			ExpectError: fmt.Errorf("empty key: %q", ""),
		},
		{
			Name:        "Watch with key escaping the prefix returns invalid key error",
			Request:     WatchRequest{Key: "/pods/../secrets", Options: watch("1", "")},
			ExpectError: fmt.Errorf("invalid key: %q", "/pods/../secrets"),
		},
		{
			Name:        "Watch with unparsable ResourceVersion returns invalid error",
			Request:     WatchRequest{Key: "/pods/", Options: watch("abc", "")},
			ExpectError: invalidRVErr,
		},
		{
			Name:        "Watch list with empty key returns empty key error",
			Request:     WatchRequest{Key: "", Options: watchList("")},
			ExpectError: fmt.Errorf("empty key: %q", ""),
		},
		{
			Name:        "Watch list with key escaping the prefix returns invalid key error",
			Request:     WatchRequest{Key: "/pods/../secrets", Options: watchList("")},
			ExpectError: fmt.Errorf("invalid key: %q", "/pods/../secrets"),
		},
		{
			Name:        "Watch list with unparsable ResourceVersion returns invalid error",
			Request:     WatchRequest{Key: "/pods/", Options: watchList("abc")},
			ExpectError: invalidRVErr,
		},
	}
}

// RunTestCorrectness executes the operations from the sequential storage model against real storage
// and validates that every transition matches the StorageModel specification.
func RunTestCorrectness(ctx context.Context, t *testing.T, store storage.Interface, storagePrefix string, keyFunc func(obj runtime.Object) (string, error), compact func(ctx context.Context, t *testing.T, resourceVersion string)) {
	versioner := store.Versioner()
	initialState := NewEmptyModel(storagePrefix, func() runtime.Object { return &example.Pod{} }, func() runtime.Object { return &example.PodList{} }, versioner)
	model := initialState.Clone()

	watchCases := watchTestCases()
	watches := make([]watch.Interface, len(watchCases))
	watchErrs := make([]error, len(watchCases))
	for i, tc := range watchCases {
		w, err := store.Watch(ctx, tc.Request.Key, tc.Request.Options)
		watches[i] = w
		watchErrs[i] = err
		if tc.ExpectError != nil {
			require.EqualError(t, err, tc.ExpectError.Error(), "step %s", tc.Name)
			continue
		}
		require.NoError(t, err, "step %s", tc.Name)
		t.Cleanup(w.Stop)
	}

	var operations []Operation
	for _, step := range correctnessTestSteps() {
		var out runtime.Object = &example.Pod{}
		var err error
		switch step.Request.Op {
		case OpCreate:
			err = store.Create(ctx, step.Request.Key, step.Request.Create.Object, out, 0)
		case OpDelete:
			validateDeletion := step.Request.Delete.ValidateDeletion
			if validateDeletion == nil {
				validateDeletion = storage.ValidateAllObjectFunc
			}
			err = store.Delete(ctx, step.Request.Key, out, step.Request.Delete.Preconditions, validateDeletion, step.Request.Delete.CachedExistingObject, storage.DeleteOptions{})
		case OpUpdate:
			err = store.GuaranteedUpdate(ctx, step.Request.Key, out, step.Request.Update.IgnoreNotFound, step.Request.Update.Preconditions, step.Request.Update.UpdateFunc, step.Request.Update.CachedExistingObject)
		case OpCompact:
			compact(ctx, t, step.Request.Compact.ResourceVersion)
			out = nil
		default:
			t.Fatalf("unknown mutation operation: %v", step.Request.Op)
		}
		var respObj runtime.Object
		if err == nil {
			respObj = out
		}
		resp := Response{Object: respObj, Err: err}
		ok, next, _ := model.Step(step.Request, resp)
		if respObj != nil {
			acc, _ := meta.CommonAccessor(respObj)
			t.Logf("Step: %s, State RV before: %d, Response RV: %s, Obj: %+v, err: %v", step.Name, model.ResourceVersion, acc.GetResourceVersion(), respObj, err)
		} else {
			t.Logf("Step: %s, State RV before: %d, Response Err: %v", step.Name, model.ResourceVersion, err)
		}
		require.True(t, ok, "step %s failed to match model state transition: req=%+v resp=%+v", step.Name, step.Request, resp)
		operations = append(operations, Operation{Request: step.Request, Response: resp})
		model = next
		if step.Request.Op == OpCompact {
			require.Equal(t, int64(model.CompactResourceVersion), store.CompactRevision(), "step %s", step.Name)
		}
	}

	replay, err := NewReplay(initialState, operations)
	require.NoError(t, err)

	for _, tc := range readTestCases() {
		t.Run(tc.Name, func(t *testing.T) {
			var out runtime.Object
			var err error
			switch tc.Request.Op {
			case OpGet:
				out = &example.Pod{}
				err = store.Get(ctx, tc.Request.Key, tc.Request.Get.Options, out)
			case OpList:
				out = &example.PodList{}
				err = store.GetList(ctx, tc.Request.Key, tc.Request.List.Options, out)
			default:
				t.Fatalf("unknown read operation: %v", tc.Request.Op)
			}
			var respObj runtime.Object
			if err == nil {
				respObj = out
			}
			resp := Response{Object: respObj, Err: err}
			ok, next, change := model.Step(tc.Request, resp)
			require.True(t, ok, "step %s failed to match model state transition: req=%+v resp=%+v", tc.Name, tc.Request, resp)
			require.Equal(t, model, next, "step %s", tc.Name)
			require.Nil(t, change, "step %s", tc.Name)
			require.NoError(t, replay.Validate(tc.Request, resp), "step %s", tc.Name)
		})
	}

	validator := NewWatchValidator(versioner, replay, keyFunc)
	for i, w := range watchCases {
		t.Run(w.Name, func(t *testing.T) {
			t.Parallel()
			var resp WatchResponse
			if watchErrs[i] != nil {
				resp = WatchResponse{Err: watchErrs[i]}
			} else {
				resp = WatchResponse{Events: collectEventsTillRV(t, watches[i], versioner, w.ExpectMaxRV)}
			}
			require.NoError(t, validator.ValidateWatch(w.Request, resp), "step %s", w.Name)
		})
	}
}

func collectEventsTillRV(t *testing.T, watcher watch.Interface, versioner storage.Versioner, targetRV uint64) []watch.Event {
	events := []watch.Event{}
	var maxRV uint64
	for {
		timeout := wait.ForeverTestTimeout
		if maxRV >= targetRV {
			timeout = 100 * time.Millisecond
		}
		select {
		case e, open := <-watcher.ResultChan():
			if !open {
				return events
			}
			if cacheable, ok := e.Object.(runtime.CacheableObject); ok {
				e.Object = cacheable.GetObject()
			}
			events = append(events, e)
			if e.Type == watch.Error {
				_, open := <-watcher.ResultChan()
				require.False(t, open, "watch channel should be closed after watch.Error")
				return events
			}
			accessor, err := meta.Accessor(e.Object)
			require.NoError(t, err)
			rv, err := versioner.ParseResourceVersion(accessor.GetResourceVersion())
			require.NoError(t, err)
			if rv > maxRV {
				maxRV = rv
			}
		case <-time.After(timeout):
			if maxRV < targetRV {
				t.Fatalf("timed out waiting for watch event with RV >= %d (maxRV=%d)", targetRV, maxRV)
			}
			return events
		}
	}
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

func newTestPodListWithContinue(rv string, remaining *int64, pods ...*example.Pod) *example.PodList {
	list := newTestPodList(rv, pods...)
	parsedRV, err := (storage.APIObjectVersioner{}).ParseResourceVersion(rv)
	if err != nil {
		panic(err)
	}
	lastPod := pods[len(pods)-1]
	continueToken, err := storage.EncodeContinue(mustGetKey(lastPod)+"\x00", "/pods/", int64(parsedRV))
	if err != nil {
		panic(err)
	}
	list.Continue = continueToken
	list.RemainingItemCount = remaining
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
