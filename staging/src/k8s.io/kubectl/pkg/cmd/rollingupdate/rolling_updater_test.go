/*
Copyright 2014 The Kubernetes Authors.

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

package rollingupdate

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"reflect"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	restclient "k8s.io/client-go/rest"
	manualfake "k8s.io/client-go/rest/fake"
	testcore "k8s.io/client-go/testing"
	"k8s.io/kubectl/pkg/scale"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
)

func oldRc(replicas int, original int) *corev1.ReplicationController {
	t := replicas
	replicasCopy := int32(t)
	return &corev1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceDefault,
			Name:      "foo-v1",
			UID:       "7764ae47-9092-11e4-8393-42010af018ff",
			Annotations: map[string]string{
				originalReplicasAnnotation: fmt.Sprintf("%d", original),
			},
		},
		Spec: corev1.ReplicationControllerSpec{
			Replicas: &replicasCopy,
			Selector: map[string]string{"version": "v1"},
			Template: &corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo-v1",
					Labels: map[string]string{"version": "v1"},
				},
			},
		},
		Status: corev1.ReplicationControllerStatus{
			Replicas: int32(replicas),
		},
	}
}

func newRc(replicas int, desired int) *corev1.ReplicationController {
	rc := oldRc(replicas, replicas)
	rc.Spec.Template = &corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "foo-v2",
			Labels: map[string]string{"version": "v2"},
		},
	}
	rc.Spec.Selector = map[string]string{"version": "v2"}
	rc.ObjectMeta = metav1.ObjectMeta{
		Namespace: metav1.NamespaceDefault,
		Name:      "foo-v2",
		Annotations: map[string]string{
			desiredReplicasAnnotation: fmt.Sprintf("%d", desired),
			sourceIDAnnotation:        "foo-v1:7764ae47-9092-11e4-8393-42010af018ff",
		},
	}
	return rc
}

// TestUpdate performs complex scenario testing for rolling updates. It
// provides fine grained control over the states for each update interval to
// allow the expression of as many edge cases as possible.
func TestUpdate(t *testing.T) {
	// up represents a simulated scale up event and expectation
	type up struct {
		// to is the expected replica count for a scale-up
		to int
	}
	// down represents a simulated scale down event and expectation
	type down struct {
		// oldReady is the number of oldRc replicas which will be seen
		// as ready during the scale down attempt
		oldReady int
		// newReady is the number of newRc replicas which will be seen
		// as ready during the scale up attempt
		newReady int
		// to is the expected replica count for the scale down
		to int
		// noop and to are mutually exclusive; if noop is true, that means for
		// this down event, no scaling attempt should be made (for example, if
		// by scaling down, the readiness minimum would be crossed.)
		noop bool
	}

	tests := []struct {
		name string
		// oldRc is the "from" deployment
		oldRc *corev1.ReplicationController
		// newRc is the "to" deployment
		newRc *corev1.ReplicationController
		// whether newRc existed (false means it was created)
		newRcExists bool
		maxUnavail  intstr.IntOrString
		maxSurge    intstr.IntOrString
		// expected is the sequence of up/down events that will be simulated and
		// verified
		expected []interface{}
		// output is the expected textual output written
		output string
	}{
		{
			name:        "10->10 30/0 fast readiness",
			oldRc:       oldRc(10, 10),
			newRc:       newRc(0, 10),
			newRcExists: false,
			maxUnavail:  intstr.FromString("30%"),
			maxSurge:    intstr.FromString("0%"),
			expected: []interface{}{
				down{oldReady: 10, newReady: 0, to: 7},
				up{3},
				down{oldReady: 7, newReady: 3, to: 4},
				up{6},
				down{oldReady: 4, newReady: 6, to: 1},
				up{9},
				down{oldReady: 1, newReady: 9, to: 0},
				up{10},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 10, scaling down foo-v1 from 10 to 0 (keep 7 pods available, don't exceed 10 pods)
Scaling foo-v1 down to 7
Scaling foo-v2 up to 3
Scaling foo-v1 down to 4
Scaling foo-v2 up to 6
Scaling foo-v1 down to 1
Scaling foo-v2 up to 9
Scaling foo-v1 down to 0
Scaling foo-v2 up to 10
`,
		},
		{
			name:        "10->10 30/0 delayed readiness",
			oldRc:       oldRc(10, 10),
			newRc:       newRc(0, 10),
			newRcExists: false,
			maxUnavail:  intstr.FromString("30%"),
			maxSurge:    intstr.FromString("0%"),
			expected: []interface{}{
				down{oldReady: 10, newReady: 0, to: 7},
				up{3},
				down{oldReady: 7, newReady: 0, noop: true},
				down{oldReady: 7, newReady: 1, to: 6},
				up{4},
				down{oldReady: 6, newReady: 4, to: 3},
				up{7},
				down{oldReady: 3, newReady: 7, to: 0},
				up{10},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 10, scaling down foo-v1 from 10 to 0 (keep 7 pods available, don't exceed 10 pods)
Scaling foo-v1 down to 7
Scaling foo-v2 up to 3
Scaling foo-v1 down to 6
Scaling foo-v2 up to 4
Scaling foo-v1 down to 3
Scaling foo-v2 up to 7
Scaling foo-v1 down to 0
Scaling foo-v2 up to 10
`,
		}, {
			name:        "10->10 30/0 fast readiness, continuation",
			oldRc:       oldRc(7, 10),
			newRc:       newRc(3, 10),
			newRcExists: false,
			maxUnavail:  intstr.FromString("30%"),
			maxSurge:    intstr.FromString("0%"),
			expected: []interface{}{
				down{oldReady: 7, newReady: 3, to: 4},
				up{6},
				down{oldReady: 4, newReady: 6, to: 1},
				up{9},
				down{oldReady: 1, newReady: 9, to: 0},
				up{10},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 3 to 10, scaling down foo-v1 from 7 to 0 (keep 7 pods available, don't exceed 10 pods)
Scaling foo-v1 down to 4
Scaling foo-v2 up to 6
Scaling foo-v1 down to 1
Scaling foo-v2 up to 9
Scaling foo-v1 down to 0
Scaling foo-v2 up to 10
`,
		}, {
			name:        "10->10 30/0 fast readiness, continued after restart which prevented first scale-up",
			oldRc:       oldRc(7, 10),
			newRc:       newRc(0, 10),
			newRcExists: false,
			maxUnavail:  intstr.FromString("30%"),
			maxSurge:    intstr.FromString("0%"),
			expected: []interface{}{
				down{oldReady: 7, newReady: 0, noop: true},
				up{3},
				down{oldReady: 7, newReady: 3, to: 4},
				up{6},
				down{oldReady: 4, newReady: 6, to: 1},
				up{9},
				down{oldReady: 1, newReady: 9, to: 0},
				up{10},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 10, scaling down foo-v1 from 7 to 0 (keep 7 pods available, don't exceed 10 pods)
Scaling foo-v2 up to 3
Scaling foo-v1 down to 4
Scaling foo-v2 up to 6
Scaling foo-v1 down to 1
Scaling foo-v2 up to 9
Scaling foo-v1 down to 0
Scaling foo-v2 up to 10
`,
		}, {
			name:        "10->10 0/30 fast readiness",
			oldRc:       oldRc(10, 10),
			newRc:       newRc(0, 10),
			newRcExists: false,
			maxUnavail:  intstr.FromString("0%"),
			maxSurge:    intstr.FromString("30%"),
			expected: []interface{}{
				up{3},
				down{oldReady: 10, newReady: 3, to: 7},
				up{6},
				down{oldReady: 7, newReady: 6, to: 4},
				up{9},
				down{oldReady: 4, newReady: 9, to: 1},
				up{10},
				down{oldReady: 1, newReady: 10, to: 0},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 10, scaling down foo-v1 from 10 to 0 (keep 10 pods available, don't exceed 13 pods)
Scaling foo-v2 up to 3
Scaling foo-v1 down to 7
Scaling foo-v2 up to 6
Scaling foo-v1 down to 4
Scaling foo-v2 up to 9
Scaling foo-v1 down to 1
Scaling foo-v2 up to 10
Scaling foo-v1 down to 0
`,
		}, {
			name:        "10->10 0/30 delayed readiness",
			oldRc:       oldRc(10, 10),
			newRc:       newRc(0, 10),
			newRcExists: false,
			maxUnavail:  intstr.FromString("0%"),
			maxSurge:    intstr.FromString("30%"),
			expected: []interface{}{
				up{3},
				down{oldReady: 10, newReady: 0, noop: true},
				down{oldReady: 10, newReady: 1, to: 9},
				up{4},
				down{oldReady: 9, newReady: 3, to: 7},
				up{6},
				down{oldReady: 7, newReady: 6, to: 4},
				up{9},
				down{oldReady: 4, newReady: 9, to: 1},
				up{10},
				down{oldReady: 1, newReady: 9, noop: true},
				down{oldReady: 1, newReady: 10, to: 0},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 10, scaling down foo-v1 from 10 to 0 (keep 10 pods available, don't exceed 13 pods)
Scaling foo-v2 up to 3
Scaling foo-v1 down to 9
Scaling foo-v2 up to 4
Scaling foo-v1 down to 7
Scaling foo-v2 up to 6
Scaling foo-v1 down to 4
Scaling foo-v2 up to 9
Scaling foo-v1 down to 1
Scaling foo-v2 up to 10
Scaling foo-v1 down to 0
`,
		}, {
			name:        "10->10 10/20 fast readiness",
			oldRc:       oldRc(10, 10),
			newRc:       newRc(0, 10),
			newRcExists: false,
			maxUnavail:  intstr.FromString("10%"),
			maxSurge:    intstr.FromString("20%"),
			expected: []interface{}{
				up{2},
				down{oldReady: 10, newReady: 2, to: 7},
				up{5},
				down{oldReady: 7, newReady: 5, to: 4},
				up{8},
				down{oldReady: 4, newReady: 8, to: 1},
				up{10},
				down{oldReady: 1, newReady: 10, to: 0},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 10, scaling down foo-v1 from 10 to 0 (keep 9 pods available, don't exceed 12 pods)
Scaling foo-v2 up to 2
Scaling foo-v1 down to 7
Scaling foo-v2 up to 5
Scaling foo-v1 down to 4
Scaling foo-v2 up to 8
Scaling foo-v1 down to 1
Scaling foo-v2 up to 10
Scaling foo-v1 down to 0
`,
		}, {
			name:        "10->10 10/20 delayed readiness",
			oldRc:       oldRc(10, 10),
			newRc:       newRc(0, 10),
			newRcExists: false,
			maxUnavail:  intstr.FromString("10%"),
			maxSurge:    intstr.FromString("20%"),
			expected: []interface{}{
				up{2},
				down{oldReady: 10, newReady: 2, to: 7},
				up{5},
				down{oldReady: 7, newReady: 4, to: 5},
				up{7},
				down{oldReady: 5, newReady: 4, noop: true},
				down{oldReady: 5, newReady: 7, to: 2},
				up{10},
				down{oldReady: 2, newReady: 9, to: 0},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 10, scaling down foo-v1 from 10 to 0 (keep 9 pods available, don't exceed 12 pods)
Scaling foo-v2 up to 2
Scaling foo-v1 down to 7
Scaling foo-v2 up to 5
Scaling foo-v1 down to 5
Scaling foo-v2 up to 7
Scaling foo-v1 down to 2
Scaling foo-v2 up to 10
Scaling foo-v1 down to 0
`,
		}, {
			name:        "10->10 10/20 fast readiness continued after restart which prevented first scale-down",
			oldRc:       oldRc(10, 10),
			newRc:       newRc(2, 10),
			newRcExists: false,
			maxUnavail:  intstr.FromString("10%"),
			maxSurge:    intstr.FromString("20%"),
			expected: []interface{}{
				down{oldReady: 10, newReady: 2, to: 7},
				up{5},
				down{oldReady: 7, newReady: 5, to: 4},
				up{8},
				down{oldReady: 4, newReady: 8, to: 1},
				up{10},
				down{oldReady: 1, newReady: 10, to: 0},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 2 to 10, scaling down foo-v1 from 10 to 0 (keep 9 pods available, don't exceed 12 pods)
Scaling foo-v1 down to 7
Scaling foo-v2 up to 5
Scaling foo-v1 down to 4
Scaling foo-v2 up to 8
Scaling foo-v1 down to 1
Scaling foo-v2 up to 10
Scaling foo-v1 down to 0
`,
		}, {
			name:        "10->10 0/100 fast readiness",
			oldRc:       oldRc(10, 10),
			newRc:       newRc(0, 10),
			newRcExists: false,
			maxUnavail:  intstr.FromString("0%"),
			maxSurge:    intstr.FromString("100%"),
			expected: []interface{}{
				up{10},
				down{oldReady: 10, newReady: 10, to: 0},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 10, scaling down foo-v1 from 10 to 0 (keep 10 pods available, don't exceed 20 pods)
Scaling foo-v2 up to 10
Scaling foo-v1 down to 0
`,
		}, {
			name:        "10->10 0/100 delayed readiness",
			oldRc:       oldRc(10, 10),
			newRc:       newRc(0, 10),
			newRcExists: false,
			maxUnavail:  intstr.FromString("0%"),
			maxSurge:    intstr.FromString("100%"),
			expected: []interface{}{
				up{10},
				down{oldReady: 10, newReady: 0, noop: true},
				down{oldReady: 10, newReady: 2, to: 8},
				down{oldReady: 8, newReady: 7, to: 3},
				down{oldReady: 3, newReady: 10, to: 0},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 10, scaling down foo-v1 from 10 to 0 (keep 10 pods available, don't exceed 20 pods)
Scaling foo-v2 up to 10
Scaling foo-v1 down to 8
Scaling foo-v1 down to 3
Scaling foo-v1 down to 0
`,
		}, {
			name:        "10->10 100/0 fast readiness",
			oldRc:       oldRc(10, 10),
			newRc:       newRc(0, 10),
			newRcExists: false,
			maxUnavail:  intstr.FromString("100%"),
			maxSurge:    intstr.FromString("0%"),
			expected: []interface{}{
				down{oldReady: 10, newReady: 0, to: 0},
				up{10},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 10, scaling down foo-v1 from 10 to 0 (keep 0 pods available, don't exceed 10 pods)
Scaling foo-v1 down to 0
Scaling foo-v2 up to 10
`,
		}, {
			name:        "1->1 25/25 maintain minimum availability",
			oldRc:       oldRc(1, 1),
			newRc:       newRc(0, 1),
			newRcExists: false,
			maxUnavail:  intstr.FromString("25%"),
			maxSurge:    intstr.FromString("25%"),
			expected: []interface{}{
				up{1},
				down{oldReady: 1, newReady: 0, noop: true},
				down{oldReady: 1, newReady: 1, to: 0},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 1, scaling down foo-v1 from 1 to 0 (keep 1 pods available, don't exceed 2 pods)
Scaling foo-v2 up to 1
Scaling foo-v1 down to 0
`,
		}, {
			name:        "1->1 0/10 delayed readiness",
			oldRc:       oldRc(1, 1),
			newRc:       newRc(0, 1),
			newRcExists: false,
			maxUnavail:  intstr.FromString("0%"),
			maxSurge:    intstr.FromString("10%"),
			expected: []interface{}{
				up{1},
				down{oldReady: 1, newReady: 0, noop: true},
				down{oldReady: 1, newReady: 1, to: 0},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 1, scaling down foo-v1 from 1 to 0 (keep 1 pods available, don't exceed 2 pods)
Scaling foo-v2 up to 1
Scaling foo-v1 down to 0
`,
		}, {
			name:        "1->1 10/10 delayed readiness",
			oldRc:       oldRc(1, 1),
			newRc:       newRc(0, 1),
			newRcExists: false,
			maxUnavail:  intstr.FromString("10%"),
			maxSurge:    intstr.FromString("10%"),
			expected: []interface{}{
				up{1},
				down{oldReady: 1, newReady: 0, noop: true},
				down{oldReady: 1, newReady: 1, to: 0},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 1, scaling down foo-v1 from 1 to 0 (keep 1 pods available, don't exceed 2 pods)
Scaling foo-v2 up to 1
Scaling foo-v1 down to 0
`,
		}, {
			name:        "3->3 1/1 fast readiness (absolute values)",
			oldRc:       oldRc(3, 3),
			newRc:       newRc(0, 3),
			newRcExists: false,
			maxUnavail:  intstr.FromInt(0),
			maxSurge:    intstr.FromInt(1),
			expected: []interface{}{
				up{1},
				down{oldReady: 3, newReady: 1, to: 2},
				up{2},
				down{oldReady: 2, newReady: 2, to: 1},
				up{3},
				down{oldReady: 1, newReady: 3, to: 0},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 3, scaling down foo-v1 from 3 to 0 (keep 3 pods available, don't exceed 4 pods)
Scaling foo-v2 up to 1
Scaling foo-v1 down to 2
Scaling foo-v2 up to 2
Scaling foo-v1 down to 1
Scaling foo-v2 up to 3
Scaling foo-v1 down to 0
`,
		}, {
			name:        "10->10 0/20 fast readiness, continued after restart which resulted in partial first scale-up",
			oldRc:       oldRc(6, 10),
			newRc:       newRc(5, 10),
			newRcExists: false,
			maxUnavail:  intstr.FromString("0%"),
			maxSurge:    intstr.FromString("20%"),
			expected: []interface{}{
				up{6},
				down{oldReady: 6, newReady: 6, to: 4},
				up{8},
				down{oldReady: 4, newReady: 8, to: 2},
				up{10},
				down{oldReady: 1, newReady: 10, to: 0},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 5 to 10, scaling down foo-v1 from 6 to 0 (keep 10 pods available, don't exceed 12 pods)
Scaling foo-v2 up to 6
Scaling foo-v1 down to 4
Scaling foo-v2 up to 8
Scaling foo-v1 down to 2
Scaling foo-v2 up to 10
Scaling foo-v1 down to 0
`,
		}, {
			name:        "10->20 0/300 fast readiness",
			oldRc:       oldRc(10, 10),
			newRc:       newRc(0, 20),
			newRcExists: false,
			maxUnavail:  intstr.FromString("0%"),
			maxSurge:    intstr.FromString("300%"),
			expected: []interface{}{
				up{20},
				down{oldReady: 10, newReady: 20, to: 0},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 20, scaling down foo-v1 from 10 to 0 (keep 20 pods available, don't exceed 80 pods)
Scaling foo-v2 up to 20
Scaling foo-v1 down to 0
`,
		}, {
			name:        "1->1 0/1 scale down unavailable rc to a ready rc (rollback)",
			oldRc:       oldRc(1, 1),
			newRc:       newRc(1, 1),
			newRcExists: true,
			maxUnavail:  intstr.FromInt(0),
			maxSurge:    intstr.FromInt(1),
			expected: []interface{}{
				up{1},
				down{oldReady: 0, newReady: 1, to: 0},
			},
			output: `Continuing update with existing controller foo-v2.
Scaling up foo-v2 from 1 to 1, scaling down foo-v1 from 1 to 0 (keep 1 pods available, don't exceed 2 pods)
Scaling foo-v1 down to 0
`,
		},
		{
			name:        "3->0 1/1 desired 0 (absolute values)",
			oldRc:       oldRc(3, 3),
			newRc:       newRc(0, 0),
			newRcExists: true,
			maxUnavail:  intstr.FromInt(1),
			maxSurge:    intstr.FromInt(1),
			expected: []interface{}{
				down{oldReady: 3, newReady: 0, to: 0},
			},
			output: `Continuing update with existing controller foo-v2.
Scaling up foo-v2 from 0 to 0, scaling down foo-v1 from 3 to 0 (keep 0 pods available, don't exceed 1 pods)
Scaling foo-v1 down to 0
`,
		},
		{
			name:        "3->0 10/10 desired 0 (percentages)",
			oldRc:       oldRc(3, 3),
			newRc:       newRc(0, 0),
			newRcExists: true,
			maxUnavail:  intstr.FromString("10%"),
			maxSurge:    intstr.FromString("10%"),
			expected: []interface{}{
				down{oldReady: 3, newReady: 0, to: 0},
			},
			output: `Continuing update with existing controller foo-v2.
Scaling up foo-v2 from 0 to 0, scaling down foo-v1 from 3 to 0 (keep 0 pods available, don't exceed 0 pods)
Scaling foo-v1 down to 0
`,
		},
		{
			name:        "3->0 10/10 desired 0 (create new RC)",
			oldRc:       oldRc(3, 3),
			newRc:       newRc(0, 0),
			newRcExists: false,
			maxUnavail:  intstr.FromString("10%"),
			maxSurge:    intstr.FromString("10%"),
			expected: []interface{}{
				down{oldReady: 3, newReady: 0, to: 0},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 0, scaling down foo-v1 from 3 to 0 (keep 0 pods available, don't exceed 0 pods)
Scaling foo-v1 down to 0
`,
		},
		{
			name:        "0->0 1/1 desired 0 (absolute values)",
			oldRc:       oldRc(0, 0),
			newRc:       newRc(0, 0),
			newRcExists: true,
			maxUnavail:  intstr.FromInt(1),
			maxSurge:    intstr.FromInt(1),
			expected: []interface{}{
				down{oldReady: 0, newReady: 0, to: 0},
			},
			output: `Continuing update with existing controller foo-v2.
Scaling up foo-v2 from 0 to 0, scaling down foo-v1 from 0 to 0 (keep 0 pods available, don't exceed 1 pods)
`,
		}, {
			name:        "30->2 50%/0",
			oldRc:       oldRc(30, 30),
			newRc:       newRc(0, 2),
			newRcExists: false,
			maxUnavail:  intstr.FromString("50%"),
			maxSurge:    intstr.FromInt(0),
			expected: []interface{}{
				down{oldReady: 30, newReady: 0, to: 1},
				up{1},
				down{oldReady: 1, newReady: 2, to: 0},
				up{2},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 2, scaling down foo-v1 from 30 to 0 (keep 1 pods available, don't exceed 2 pods)
Scaling foo-v1 down to 1
Scaling foo-v2 up to 1
Scaling foo-v1 down to 0
Scaling foo-v2 up to 2
`,
		},
		{
			name:        "2->2 1/0 blocked oldRc",
			oldRc:       oldRc(2, 2),
			newRc:       newRc(0, 2),
			newRcExists: false,
			maxUnavail:  intstr.FromInt(1),
			maxSurge:    intstr.FromInt(0),
			expected: []interface{}{
				down{oldReady: 1, newReady: 0, to: 1},
				up{1},
				down{oldReady: 1, newReady: 1, to: 0},
				up{2},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 2, scaling down foo-v1 from 2 to 0 (keep 1 pods available, don't exceed 2 pods)
Scaling foo-v1 down to 1
Scaling foo-v2 up to 1
Scaling foo-v1 down to 0
Scaling foo-v2 up to 2
`,
		},
		{
			name:        "1->1 1/0 allow maxUnavailability",
			oldRc:       oldRc(1, 1),
			newRc:       newRc(0, 1),
			newRcExists: false,
			maxUnavail:  intstr.FromString("1%"),
			maxSurge:    intstr.FromInt(0),
			expected: []interface{}{
				down{oldReady: 1, newReady: 0, to: 0},
				up{1},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 1, scaling down foo-v1 from 1 to 0 (keep 0 pods available, don't exceed 1 pods)
Scaling foo-v1 down to 0
Scaling foo-v2 up to 1
`,
		},
		{
			name:        "1->2 25/25 complex asymmetric deployment",
			oldRc:       oldRc(1, 1),
			newRc:       newRc(0, 2),
			newRcExists: false,
			maxUnavail:  intstr.FromString("25%"),
			maxSurge:    intstr.FromString("25%"),
			expected: []interface{}{
				up{2},
				down{oldReady: 1, newReady: 2, to: 0},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 2, scaling down foo-v1 from 1 to 0 (keep 2 pods available, don't exceed 3 pods)
Scaling foo-v2 up to 2
Scaling foo-v1 down to 0
`,
		},
		{
			name:        "2->2 25/1 maxSurge trumps maxUnavailable",
			oldRc:       oldRc(2, 2),
			newRc:       newRc(0, 2),
			newRcExists: false,
			maxUnavail:  intstr.FromString("25%"),
			maxSurge:    intstr.FromString("1%"),
			expected: []interface{}{
				up{1},
				down{oldReady: 2, newReady: 1, to: 1},
				up{2},
				down{oldReady: 1, newReady: 2, to: 0},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 2, scaling down foo-v1 from 2 to 0 (keep 2 pods available, don't exceed 3 pods)
Scaling foo-v2 up to 1
Scaling foo-v1 down to 1
Scaling foo-v2 up to 2
Scaling foo-v1 down to 0
`,
		},
		{
			name:        "2->2 25/0 maxUnavailable resolves to zero, then one",
			oldRc:       oldRc(2, 2),
			newRc:       newRc(0, 2),
			newRcExists: false,
			maxUnavail:  intstr.FromString("25%"),
			maxSurge:    intstr.FromString("0%"),
			expected: []interface{}{
				down{oldReady: 2, newReady: 0, to: 1},
				up{1},
				down{oldReady: 1, newReady: 1, to: 0},
				up{2},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 2, scaling down foo-v1 from 2 to 0 (keep 1 pods available, don't exceed 2 pods)
Scaling foo-v1 down to 1
Scaling foo-v2 up to 1
Scaling foo-v1 down to 0
Scaling foo-v2 up to 2
`,
		},
	}

	for i, tt := range tests {
		// Extract expectations into some makeshift FIFOs so they can be returned
		// in the correct order from the right places. This lets scale downs be
		// expressed a single event even though the data is used from multiple
		// interface calls.
		t.Run(tt.name, func(t *testing.T) {
			oldReady := []int{}
			newReady := []int{}
			upTo := []int{}
			downTo := []int{}
			for _, event := range tt.expected {
				switch e := event.(type) {
				case down:
					oldReady = append(oldReady, e.oldReady)
					newReady = append(newReady, e.newReady)
					if !e.noop {
						downTo = append(downTo, e.to)
					}
				case up:
					upTo = append(upTo, e.to)
				}
			}

			// Make a way to get the next item from our FIFOs. Returns -1 if the array
			// is empty.
			next := func(s *[]int) int {
				slice := *s
				v := -1
				if len(slice) > 0 {
					v = slice[0]
					if len(slice) > 1 {
						*s = slice[1:]
					} else {
						*s = []int{}
					}
				}
				return v
			}
			t.Logf("running test %d (%s) (up: %v, down: %v, oldReady: %v, newReady: %v)", i, tt.name, upTo, downTo, oldReady, newReady)
			updater := &RollingUpdater{
				ns: "default",
				scaleAndWait: func(rc *corev1.ReplicationController, retry *scale.RetryParams, wait *scale.RetryParams) (*corev1.ReplicationController, error) {
					// Return a scale up or scale down expectation depending on the rc,
					// and throw errors if there is no expectation expressed for this
					// call.
					expected := -1
					switch {
					case rc == tt.newRc:
						t.Logf("scaling up %s to %d", rc.Name, rc.Spec.Replicas)
						expected = next(&upTo)
					case rc == tt.oldRc:
						t.Logf("scaling down %s to %d", rc.Name, rc.Spec.Replicas)
						expected = next(&downTo)
					}
					if expected == -1 {
						t.Fatalf("unexpected scale of %s to %d", rc.Name, rc.Spec.Replicas)
					} else if e, a := expected, int(*rc.Spec.Replicas); e != a {
						t.Fatalf("expected scale of %s to %d, got %d", rc.Name, e, a)
					}
					// Simulate the scale.
					rc.Status.Replicas = *rc.Spec.Replicas
					return rc, nil
				},
				getOrCreateTargetController: func(controller *corev1.ReplicationController, sourceID string) (*corev1.ReplicationController, bool, error) {
					// Simulate a create vs. update of an existing controller.
					return tt.newRc, tt.newRcExists, nil
				},
				cleanup: func(oldRc, newRc *corev1.ReplicationController, config *RollingUpdaterConfig) error {
					return nil
				},
			}
			// Set up a mock readiness check which handles the test assertions.
			updater.getReadyPods = func(oldRc, newRc *corev1.ReplicationController, minReadySecondsDeadline int32) (int32, int32, error) {
				// Return simulated readiness, and throw an error if this call has no
				// expectations defined.
				oldReady := next(&oldReady)
				newReady := next(&newReady)
				if oldReady == -1 || newReady == -1 {
					t.Fatalf("unexpected getReadyPods call for:\noldRc: %#v\nnewRc: %#v", oldRc, newRc)
				}
				return int32(oldReady), int32(newReady), nil
			}
			var buffer bytes.Buffer
			config := &RollingUpdaterConfig{
				Out:            &buffer,
				OldRc:          tt.oldRc,
				NewRc:          tt.newRc,
				UpdatePeriod:   0,
				Interval:       time.Millisecond,
				Timeout:        time.Millisecond,
				CleanupPolicy:  DeleteRollingUpdateCleanupPolicy,
				MaxUnavailable: tt.maxUnavail,
				MaxSurge:       tt.maxSurge,
			}
			err := updater.Update(config)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if buffer.String() != tt.output {
				t.Errorf("Bad output. expected:\n%s\ngot:\n%s", tt.output, buffer.String())
			}
		})
	}
}

// TestUpdate_progressTimeout ensures that an update which isn't making any
// progress will eventually time out with a specified error.
func TestUpdate_progressTimeout(t *testing.T) {
	oldRc := oldRc(2, 2)
	newRc := newRc(0, 2)
	updater := &RollingUpdater{
		ns: "default",
		scaleAndWait: func(rc *corev1.ReplicationController, retry *scale.RetryParams, wait *scale.RetryParams) (*corev1.ReplicationController, error) {
			// Do nothing.
			return rc, nil
		},
		getOrCreateTargetController: func(controller *corev1.ReplicationController, sourceID string) (*corev1.ReplicationController, bool, error) {
			return newRc, false, nil
		},
		cleanup: func(oldRc, newRc *corev1.ReplicationController, config *RollingUpdaterConfig) error {
			return nil
		},
	}
	updater.getReadyPods = func(oldRc, newRc *corev1.ReplicationController, minReadySeconds int32) (int32, int32, error) {
		// Coerce a timeout by pods never becoming ready.
		return 0, 0, nil
	}
	var buffer bytes.Buffer
	config := &RollingUpdaterConfig{
		Out:            &buffer,
		OldRc:          oldRc,
		NewRc:          newRc,
		UpdatePeriod:   0,
		Interval:       time.Millisecond,
		Timeout:        time.Millisecond,
		CleanupPolicy:  DeleteRollingUpdateCleanupPolicy,
		MaxUnavailable: intstr.FromInt(0),
		MaxSurge:       intstr.FromInt(1),
	}
	err := updater.Update(config)
	if err == nil {
		t.Fatalf("expected an error")
	}
	if e, a := "timed out waiting for any update progress to be made", err.Error(); e != a {
		t.Fatalf("expected error message: %s, got: %s", e, a)
	}
}

func TestUpdate_assignOriginalAnnotation(t *testing.T) {
	oldRc := oldRc(1, 1)
	delete(oldRc.Annotations, originalReplicasAnnotation)
	newRc := newRc(1, 1)
	fake := fake.NewSimpleClientset(oldRc)
	updater := &RollingUpdater{
		rcClient:  fake.CoreV1(),
		podClient: fake.CoreV1(),
		ns:        "default",
		scaleAndWait: func(rc *corev1.ReplicationController, retry *scale.RetryParams, wait *scale.RetryParams) (*corev1.ReplicationController, error) {
			return rc, nil
		},
		getOrCreateTargetController: func(controller *corev1.ReplicationController, sourceID string) (*corev1.ReplicationController, bool, error) {
			return newRc, false, nil
		},
		cleanup: func(oldRc, newRc *corev1.ReplicationController, config *RollingUpdaterConfig) error {
			return nil
		},
		getReadyPods: func(oldRc, newRc *corev1.ReplicationController, minReadySeconds int32) (int32, int32, error) {
			return 1, 1, nil
		},
	}
	var buffer bytes.Buffer
	config := &RollingUpdaterConfig{
		Out:            &buffer,
		OldRc:          oldRc,
		NewRc:          newRc,
		UpdatePeriod:   0,
		Interval:       time.Millisecond,
		Timeout:        time.Millisecond,
		CleanupPolicy:  DeleteRollingUpdateCleanupPolicy,
		MaxUnavailable: intstr.FromString("100%"),
	}
	err := updater.Update(config)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	updateAction := fake.Actions()[1].(testcore.UpdateAction)
	if updateAction.GetResource().GroupResource() != corev1.Resource("replicationcontrollers") {
		t.Fatalf("expected rc to be updated: %#v", updateAction)
	}
	if e, a := "1", updateAction.GetObject().(*corev1.ReplicationController).Annotations[originalReplicasAnnotation]; e != a {
		t.Fatalf("expected annotation value %s, got %s", e, a)
	}
}

func TestRollingUpdater_multipleContainersInPod(t *testing.T) {
	tests := []struct {
		name          string
		oldRc         *corev1.ReplicationController
		newRc         *corev1.ReplicationController
		container     string
		image         string
		deploymentKey string
	}{
		{
			name: "test1",
			oldRc: &corev1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: metav1.NamespaceDefault,
					Name:      "foo",
				},
				Spec: corev1.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "old",
					},
					Template: &corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"dk": "old",
							},
						},
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  "container1",
									Image: "image1",
								},
								{
									Name:  "container2",
									Image: "image2",
								},
							},
						},
					},
				},
			},
			newRc: &corev1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: metav1.NamespaceDefault,
					Name:      "foo",
				},
				Spec: corev1.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "old",
					},
					Template: &corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"dk": "old",
							},
						},
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  "container1",
									Image: "newimage",
								},
								{
									Name:  "container2",
									Image: "image2",
								},
							},
						},
					},
				},
			},
			container:     "container1",
			image:         "newimage",
			deploymentKey: "dk",
		},
		{
			name: "test2",
			oldRc: &corev1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: metav1.NamespaceDefault,
					Name:      "bar",
				},
				Spec: corev1.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "old",
					},
					Template: &corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"dk": "old",
							},
						},
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  "container1",
									Image: "image1",
								},
							},
						},
					},
				},
			},
			newRc: &corev1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: metav1.NamespaceDefault,
					Name:      "bar",
				},
				Spec: corev1.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "old",
					},
					Template: &corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"dk": "old",
							},
						},
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  "container1",
									Image: "newimage",
								},
							},
						},
					},
				},
			},
			container:     "container1",
			image:         "newimage",
			deploymentKey: "dk",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fake := fake.NewSimpleClientset(tt.oldRc)

			codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

			deploymentHash, err := util.HashObject(tt.newRc, codec)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			tt.newRc.Spec.Selector[tt.deploymentKey] = deploymentHash
			tt.newRc.Spec.Template.Labels[tt.deploymentKey] = deploymentHash
			tt.newRc.Name = fmt.Sprintf("%s-%s", tt.newRc.Name, deploymentHash)

			config := &NewControllerConfig{
				Namespace:     metav1.NamespaceDefault,
				OldName:       tt.oldRc.ObjectMeta.Name,
				NewName:       tt.newRc.ObjectMeta.Name,
				Image:         tt.image,
				Container:     tt.container,
				DeploymentKey: tt.deploymentKey,
			}
			updatedRc, err := CreateNewControllerFromCurrentController(fake.CoreV1(), codec, config)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(updatedRc, tt.newRc) {
				t.Errorf("expected:\n%#v\ngot:\n%#v\n", tt.newRc, updatedRc)
			}
		})
	}
}

// TestRollingUpdater_cleanupWithClients ensures that the cleanup policy is
// correctly implemented.
func TestRollingUpdater_cleanupWithClients(t *testing.T) {
	rc := oldRc(2, 2)
	rcExisting := newRc(1, 3)

	tests := []struct {
		name      string
		policy    RollingUpdaterCleanupPolicy
		responses []runtime.Object
		expected  []string
	}{
		{
			name:      "preserve",
			policy:    PreserveRollingUpdateCleanupPolicy,
			responses: []runtime.Object{rcExisting},
			expected: []string{
				"get",
				"update",
				"get",
				"get",
			},
		},
		{
			name:      "delete",
			policy:    DeleteRollingUpdateCleanupPolicy,
			responses: []runtime.Object{rcExisting},
			expected: []string{
				"get",
				"update",
				"get",
				"get",
				"delete",
			},
		},
		//{
		// This cases is separated to a standalone
		// TestRollingUpdater_cleanupWithClients_Rename. We have to do this
		// because the unversioned fake client is unable to delete objects.
		// TODO: uncomment this case when the unversioned fake client uses
		// pkg/client/testing/core.
		//	{
		//		name:      "rename",
		//		policy:    RenameRollingUpdateCleanupPolicy,
		//		responses: []runtime.Object{rcExisting},
		//		expected: []string{
		//			"get",
		//			"update",
		//			"get",
		//			"get",
		//			"delete",
		//			"create",
		//			"delete",
		//		},
		//	},
		//},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			objs := []runtime.Object{rc}
			objs = append(objs, tt.responses...)
			fake := fake.NewSimpleClientset(objs...)
			updater := &RollingUpdater{
				ns:        "default",
				rcClient:  fake.CoreV1(),
				podClient: fake.CoreV1(),
			}
			config := &RollingUpdaterConfig{
				Out:           ioutil.Discard,
				OldRc:         rc,
				NewRc:         rcExisting,
				UpdatePeriod:  0,
				Interval:      time.Millisecond,
				Timeout:       time.Millisecond,
				CleanupPolicy: tt.policy,
			}
			err := updater.cleanupWithClients(rc, rcExisting, config)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if len(fake.Actions()) != len(tt.expected) {
				t.Fatalf("%s: unexpected actions: %v, expected %v", tt.name, fake.Actions(), tt.expected)
			}
			for j, action := range fake.Actions() {
				if e, a := tt.expected[j], action.GetVerb(); e != a {
					t.Errorf("%s: unexpected action: expected %s, got %s", tt.name, e, a)
				}
			}
		})
	}
}

// TestRollingUpdater_cleanupWithClients_Rename tests the rename cleanup policy. It's separated to
// a standalone test because the unversioned fake client is unable to delete
// objects.
// TODO: move this test back to TestRollingUpdater_cleanupWithClients
// when the fake client uses pkg/client/testing/core in the future.
func TestRollingUpdater_cleanupWithClients_Rename(t *testing.T) {
	rc := oldRc(2, 2)
	rcExisting := newRc(1, 3)
	expectedActions := []string{"delete", "get", "create"}
	fake := fake.NewSimpleClientset()
	fake.AddReactor("*", "*", func(action testcore.Action) (handled bool, ret runtime.Object, err error) {
		switch action.(type) {
		case testcore.CreateAction:
			return true, nil, nil
		case testcore.GetAction:
			return true, nil, errors.NewNotFound(schema.GroupResource{}, "")
		case testcore.DeleteAction:
			return true, nil, nil
		}
		return false, nil, nil
	})

	err := Rename(fake.CoreV1(), rcExisting, rc.Name)
	if err != nil {
		t.Fatal(err)
	}
	for j, action := range fake.Actions() {
		if e, a := expectedActions[j], action.GetVerb(); e != a {
			t.Errorf("unexpected action: expected %s, got %s", e, a)
		}
	}
}

func TestFindSourceController(t *testing.T) {
	ctrl1 := corev1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceDefault,
			Name:      "foo",
			Annotations: map[string]string{
				sourceIDAnnotation: "bar:1234",
			},
		},
	}
	ctrl2 := corev1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceDefault,
			Name:      "bar",
			Annotations: map[string]string{
				sourceIDAnnotation: "foo:12345",
			},
		},
	}
	ctrl3 := corev1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceDefault,
			Name:      "baz",
			Annotations: map[string]string{
				sourceIDAnnotation: "baz:45667",
			},
		},
	}
	tests := []struct {
		list               *corev1.ReplicationControllerList
		expectedController *corev1.ReplicationController
		name               string
		expectError        bool
	}{
		{
			list:        &corev1.ReplicationControllerList{},
			expectError: true,
		},
		{
			list: &corev1.ReplicationControllerList{
				Items: []corev1.ReplicationController{ctrl1},
			},
			name:        "foo",
			expectError: true,
		},
		{
			list: &corev1.ReplicationControllerList{
				Items: []corev1.ReplicationController{ctrl1},
			},
			name:               "bar",
			expectedController: &ctrl1,
		},
		{
			list: &corev1.ReplicationControllerList{
				Items: []corev1.ReplicationController{ctrl1, ctrl2},
			},
			name:               "bar",
			expectedController: &ctrl1,
		},
		{
			list: &corev1.ReplicationControllerList{
				Items: []corev1.ReplicationController{ctrl1, ctrl2},
			},
			name:               "foo",
			expectedController: &ctrl2,
		},
		{
			list: &corev1.ReplicationControllerList{
				Items: []corev1.ReplicationController{ctrl1, ctrl2, ctrl3},
			},
			name:               "baz",
			expectedController: &ctrl3,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fakeClient := fake.NewSimpleClientset(tt.list)
			ctrl, err := FindSourceController(fakeClient.CoreV1(), "default", tt.name)
			if tt.expectError && err == nil {
				t.Errorf("unexpected non-error")
			}
			if !tt.expectError && err != nil {
				t.Errorf("unexpected error")
			}
			if !reflect.DeepEqual(ctrl, tt.expectedController) {
				t.Errorf("expected:\n%v\ngot:\n%v\n", tt.expectedController, ctrl)
			}
		})
	}
}

func TestUpdateExistingReplicationController(t *testing.T) {
	tests := []struct {
		rc              *corev1.ReplicationController
		name            string
		deploymentKey   string
		deploymentValue string

		expectedRc *corev1.ReplicationController
		expectErr  bool
	}{
		{
			rc: &corev1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: metav1.NamespaceDefault,
					Name:      "foo",
				},
				Spec: corev1.ReplicationControllerSpec{
					Template: &corev1.PodTemplateSpec{},
				},
			},
			name:            "foo",
			deploymentKey:   "dk",
			deploymentValue: "some-hash",

			expectedRc: &corev1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: metav1.NamespaceDefault,
					Name:      "foo",
					Annotations: map[string]string{
						"kubectl.kubernetes.io/next-controller-id": "foo",
					},
				},
				Spec: corev1.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "some-hash",
					},
					Template: &corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"dk": "some-hash",
							},
						},
					},
				},
			},
		},
		{
			rc: &corev1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: metav1.NamespaceDefault,
					Name:      "foo",
				},
				Spec: corev1.ReplicationControllerSpec{
					Template: &corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"dk": "some-other-hash",
							},
						},
					},
					Selector: map[string]string{
						"dk": "some-other-hash",
					},
				},
			},
			name:            "foo",
			deploymentKey:   "dk",
			deploymentValue: "some-hash",

			expectedRc: &corev1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: metav1.NamespaceDefault,
					Name:      "foo",
					Annotations: map[string]string{
						"kubectl.kubernetes.io/next-controller-id": "foo",
					},
				},
				Spec: corev1.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "some-other-hash",
					},
					Template: &corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"dk": "some-other-hash",
							},
						},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buffer := &bytes.Buffer{}
			fakeClient := fake.NewSimpleClientset(tt.expectedRc)
			rc, err := UpdateExistingReplicationController(fakeClient.CoreV1(), fakeClient.CoreV1(), tt.rc, "default", tt.name, tt.deploymentKey, tt.deploymentValue, buffer)
			if !reflect.DeepEqual(rc, tt.expectedRc) {
				t.Errorf("expected:\n%#v\ngot:\n%#v\n", tt.expectedRc, rc)
			}
			if tt.expectErr && err == nil {
				t.Errorf("unexpected non-error")
			}
			if !tt.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestUpdateRcWithRetries(t *testing.T) {
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	one := int32(1)
	grace := int64(30)
	enableServiceLinks := corev1.DefaultEnableServiceLinks
	rc := &corev1.ReplicationController{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "ReplicationController",
		},
		ObjectMeta: metav1.ObjectMeta{Name: "rc",
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		Spec: corev1.ReplicationControllerSpec{
			Replicas: &one,
			Selector: map[string]string{
				"foo": "bar",
			},
			Template: &corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
				Spec: corev1.PodSpec{
					RestartPolicy:                 corev1.RestartPolicyAlways,
					DNSPolicy:                     corev1.DNSClusterFirst,
					TerminationGracePeriodSeconds: &grace,
					SecurityContext:               &corev1.PodSecurityContext{},
					EnableServiceLinks:            &enableServiceLinks,
				},
			},
		},
	}
	rc.Spec.Template.Spec.SchedulerName = "default-scheduler"

	// Test end to end updating of the rc with retries. Essentially make sure the update handler
	// sees the right updates, failures in update/get are handled properly, and that the updated
	// rc with new resource version is returned to the caller. Without any of these rollingupdate
	// will fail cryptically.
	newRc := *rc
	newRc.ResourceVersion = "2"
	newRc.Spec.Selector["baz"] = "foobar"
	header := http.Header{}
	header.Set("Content-Type", runtime.ContentTypeJSON)
	updates := []*http.Response{
		{StatusCode: http.StatusConflict, Header: header, Body: objBody(codec, &corev1.ReplicationController{})}, // conflict
		{StatusCode: http.StatusConflict, Header: header, Body: objBody(codec, &corev1.ReplicationController{})}, // conflict
		{StatusCode: http.StatusOK, Header: header, Body: objBody(codec, &newRc)},
	}
	gets := []*http.Response{
		{StatusCode: http.StatusInternalServerError, Header: header, Body: objBody(codec, &corev1.ReplicationController{})},
		{StatusCode: http.StatusOK, Header: header, Body: objBody(codec, rc)},
	}
	fakeClient := &manualfake.RESTClient{
		GroupVersion:         corev1.SchemeGroupVersion,
		NegotiatedSerializer: scheme.Codecs,
		Client: manualfake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api/v1/namespaces/default/replicationcontrollers/rc" && m == "PUT":
				update := updates[0]
				updates = updates[1:]
				// We should always get an update with a valid rc even when the get fails. The rc should always
				// contain the update.
				if c, ok := readOrDie(t, req, codec).(*corev1.ReplicationController); !ok || !apiequality.Semantic.DeepEqual(rc, c) {
					t.Errorf("Unexpected update body, got %+v expected %+v", c, rc)
					t.Error(diff.ObjectDiff(rc, c))
				} else if sel, ok := c.Spec.Selector["baz"]; !ok || sel != "foobar" {
					t.Errorf("Expected selector label update, got %+v", c.Spec.Selector)
				} else {
					delete(c.Spec.Selector, "baz")
				}
				return update, nil
			case p == "/api/v1/namespaces/default/replicationcontrollers/rc" && m == "GET":
				get := gets[0]
				gets = gets[1:]
				return get, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	clientConfig := &restclient.Config{
		APIPath: "/api",
		ContentConfig: restclient.ContentConfig{
			NegotiatedSerializer: scheme.Codecs,
			GroupVersion:         &corev1.SchemeGroupVersion,
		},
	}
	restClient, _ := restclient.RESTClientFor(clientConfig)
	restClient.Client = fakeClient.Client
	clientset := kubernetes.New(restClient)

	if rc, err := updateRcWithRetries(
		clientset.CoreV1(), "default", rc, func(c *corev1.ReplicationController) {
			c.Spec.Selector["baz"] = "foobar"
		}); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else if sel, ok := rc.Spec.Selector["baz"]; !ok || sel != "foobar" || rc.ResourceVersion != "2" {
		t.Errorf("Expected updated rc, got %+v", rc)
	}
	if len(updates) != 0 || len(gets) != 0 {
		t.Errorf("Remaining updates %#v gets %#v", updates, gets)
	}
}

func readOrDie(t *testing.T, req *http.Request, codec runtime.Codec) runtime.Object {
	data, err := ioutil.ReadAll(req.Body)
	if err != nil {
		t.Errorf("Error reading: %v", err)
		t.FailNow()
	}
	codec2 := scheme.Codecs.UniversalDecoder(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	obj, err := runtime.Decode(codec2, data)
	if err != nil {
		t.Log(string(data))
		t.Errorf("error decoding: %v", err)
		t.FailNow()
	}
	return obj
}

func objBody(codec runtime.Codec, obj runtime.Object) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(codec, obj))))
}

func TestAddDeploymentHash(t *testing.T) {
	buf := &bytes.Buffer{}
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	rc := &corev1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{Name: "rc"},
		Spec: corev1.ReplicationControllerSpec{
			Selector: map[string]string{
				"foo": "bar",
			},
			Template: &corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
			},
		},
	}

	podList := &corev1.PodList{
		Items: []corev1.Pod{
			{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			{ObjectMeta: metav1.ObjectMeta{Name: "bar"}},
			{ObjectMeta: metav1.ObjectMeta{Name: "baz"}},
		},
	}

	seen := sets.String{}
	updatedRc := false
	fakeClient := &manualfake.RESTClient{
		GroupVersion:         corev1.SchemeGroupVersion,
		NegotiatedSerializer: scheme.Codecs,
		Client: manualfake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			header := http.Header{}
			header.Set("Content-Type", runtime.ContentTypeJSON)
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api/v1/namespaces/default/pods" && m == "GET":
				if req.URL.RawQuery != "labelSelector=foo%3Dbar" {
					t.Errorf("Unexpected query string: %s", req.URL.RawQuery)
				}
				return &http.Response{StatusCode: http.StatusOK, Header: header, Body: objBody(codec, podList)}, nil
			case p == "/api/v1/namespaces/default/pods/foo" && m == "PUT":
				seen.Insert("foo")
				obj := readOrDie(t, req, codec)
				podList.Items[0] = *(obj.(*corev1.Pod))
				return &http.Response{StatusCode: http.StatusOK, Header: header, Body: objBody(codec, &podList.Items[0])}, nil
			case p == "/api/v1/namespaces/default/pods/bar" && m == "PUT":
				seen.Insert("bar")
				obj := readOrDie(t, req, codec)
				podList.Items[1] = *(obj.(*corev1.Pod))
				return &http.Response{StatusCode: http.StatusOK, Header: header, Body: objBody(codec, &podList.Items[1])}, nil
			case p == "/api/v1/namespaces/default/pods/baz" && m == "PUT":
				seen.Insert("baz")
				obj := readOrDie(t, req, codec)
				podList.Items[2] = *(obj.(*corev1.Pod))
				return &http.Response{StatusCode: http.StatusOK, Header: header, Body: objBody(codec, &podList.Items[2])}, nil
			case p == "/api/v1/namespaces/default/replicationcontrollers/rc" && m == "PUT":
				updatedRc = true
				return &http.Response{StatusCode: http.StatusOK, Header: header, Body: objBody(codec, rc)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	clientConfig := &restclient.Config{
		APIPath: "/api",
		ContentConfig: restclient.ContentConfig{
			NegotiatedSerializer: scheme.Codecs,
			GroupVersion:         &corev1.SchemeGroupVersion,
		},
	}
	restClient, _ := restclient.RESTClientFor(clientConfig)
	restClient.Client = fakeClient.Client
	clientset := kubernetes.New(restClient)

	if _, err := AddDeploymentKeyToReplicationController(rc, clientset.CoreV1(), clientset.CoreV1(), "dk", "hash", metav1.NamespaceDefault, buf); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	for _, pod := range podList.Items {
		if !seen.Has(pod.Name) {
			t.Errorf("Missing update for pod: %s", pod.Name)
		}
	}
	if !updatedRc {
		t.Errorf("Failed to update replication controller with new labels")
	}
}

func TestRollingUpdater_readyPods(t *testing.T) {
	count := 0
	now := metav1.Date(2016, time.April, 1, 1, 0, 0, 0, time.UTC)
	mkpod := func(owner *corev1.ReplicationController, ready bool, readyTime metav1.Time) *corev1.Pod {
		count = count + 1
		labels := map[string]string{}
		for k, v := range owner.Spec.Selector {
			labels[k] = v
		}
		status := corev1.ConditionTrue
		if !ready {
			status = corev1.ConditionFalse
		}
		return &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: metav1.NamespaceDefault,
				Name:      fmt.Sprintf("pod-%d", count),
				Labels:    labels,
			},
			Status: corev1.PodStatus{
				Conditions: []corev1.PodCondition{
					{
						Type:               corev1.PodReady,
						Status:             status,
						LastTransitionTime: readyTime,
					},
				},
			},
		}
	}

	tests := []struct {
		name  string
		oldRc *corev1.ReplicationController
		newRc *corev1.ReplicationController
		// expectated old/new ready counts
		oldReady int32
		newReady int32
		// pods owned by the rcs; indicate whether they're ready
		oldPods []bool
		newPods []bool
		// deletions - should be less then the size of the respective slice above
		// e.g. len(oldPods) > oldPodDeletions && len(newPods) > newPodDeletions
		oldPodDeletions int
		newPodDeletions int
		// specify additional time to wait for deployment to wait on top of the
		// pod ready time
		minReadySeconds int32
		podReadyTimeFn  func() metav1.Time
		nowFn           func() metav1.Time
	}{
		{
			name:     "test1",
			oldRc:    oldRc(4, 4),
			newRc:    newRc(4, 4),
			oldReady: 4,
			newReady: 2,
			oldPods: []bool{
				true,
				true,
				true,
				true,
			},
			newPods: []bool{
				true,
				false,
				true,
				false,
			},
		},
		{
			name:     "test2",
			oldRc:    oldRc(4, 4),
			newRc:    newRc(4, 4),
			oldReady: 0,
			newReady: 1,
			oldPods: []bool{
				false,
			},
			newPods: []bool{
				true,
			},
		},
		{
			name:     "test3",
			oldRc:    oldRc(4, 4),
			newRc:    newRc(4, 4),
			oldReady: 1,
			newReady: 0,
			oldPods: []bool{
				true,
			},
			newPods: []bool{
				false,
			},
		},
		{
			name:     "test4",
			oldRc:    oldRc(4, 4),
			newRc:    newRc(4, 4),
			oldReady: 0,
			newReady: 0,
			oldPods: []bool{
				true,
			},
			newPods: []bool{
				true,
			},
			minReadySeconds: 5,
			nowFn:           func() metav1.Time { return now },
		},
		{
			name:     "test5",
			oldRc:    oldRc(4, 4),
			newRc:    newRc(4, 4),
			oldReady: 1,
			newReady: 1,
			oldPods: []bool{
				true,
			},
			newPods: []bool{
				true,
			},
			minReadySeconds: 5,
			nowFn:           func() metav1.Time { return metav1.Time{Time: now.Add(time.Duration(6 * time.Second))} },
			podReadyTimeFn:  func() metav1.Time { return now },
		},
		{
			name:     "test6",
			oldRc:    oldRc(4, 4),
			newRc:    newRc(4, 4),
			oldReady: 2,
			newReady: 0,
			oldPods: []bool{
				// All old pods are ready
				true, true, true, true,
			},
			// Two of them have been marked for deletion though
			oldPodDeletions: 2,
		},
	}

	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Logf("evaluating test %d", i)
			if tt.nowFn == nil {
				tt.nowFn = func() metav1.Time { return now }
			}
			if tt.podReadyTimeFn == nil {
				tt.podReadyTimeFn = tt.nowFn
			}
			// Populate the fake client with pods associated with their owners.
			pods := []runtime.Object{}
			for _, ready := range tt.oldPods {
				pod := mkpod(tt.oldRc, ready, tt.podReadyTimeFn())
				if tt.oldPodDeletions > 0 {
					now := metav1.Now()
					pod.DeletionTimestamp = &now
					tt.oldPodDeletions--
				}
				pods = append(pods, pod)
			}
			for _, ready := range tt.newPods {
				pod := mkpod(tt.newRc, ready, tt.podReadyTimeFn())
				if tt.newPodDeletions > 0 {
					now := metav1.Now()
					pod.DeletionTimestamp = &now
					tt.newPodDeletions--
				}
				pods = append(pods, pod)
			}
			client := fake.NewSimpleClientset(pods...)

			updater := &RollingUpdater{
				ns:        "default",
				rcClient:  client.CoreV1(),
				podClient: client.CoreV1(),
				nowFn:     tt.nowFn,
			}
			oldReady, newReady, err := updater.readyPods(tt.oldRc, tt.newRc, tt.minReadySeconds)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if e, a := tt.oldReady, oldReady; e != a {
				t.Errorf("expected old ready %d, got %d", e, a)
			}
			if e, a := tt.newReady, newReady; e != a {
				t.Errorf("expected new ready %d, got %d", e, a)
			}
		})
	}
}
