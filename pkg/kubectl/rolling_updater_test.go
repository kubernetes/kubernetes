/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package kubectl

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"reflect"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/fake"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/sets"
)

func oldRc(replicas int, original int) *api.ReplicationController {
	return &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "foo-v1",
			UID:  "7764ae47-9092-11e4-8393-42010af018ff",
			Annotations: map[string]string{
				originalReplicasAnnotation: fmt.Sprintf("%d", original),
			},
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: replicas,
			Selector: map[string]string{"version": "v1"},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Name:   "foo-v1",
					Labels: map[string]string{"version": "v1"},
				},
			},
		},
		Status: api.ReplicationControllerStatus{
			Replicas: replicas,
		},
	}
}

func newRc(replicas int, desired int) *api.ReplicationController {
	rc := oldRc(replicas, replicas)
	rc.Spec.Template = &api.PodTemplateSpec{
		ObjectMeta: api.ObjectMeta{
			Name:   "foo-v2",
			Labels: map[string]string{"version": "v2"},
		},
	}
	rc.Spec.Selector = map[string]string{"version": "v2"}
	rc.ObjectMeta = api.ObjectMeta{
		Name: "foo-v2",
		Annotations: map[string]string{
			desiredReplicasAnnotation: fmt.Sprintf("%d", desired),
			sourceIdAnnotation:        "foo-v1:7764ae47-9092-11e4-8393-42010af018ff",
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
		oldRc *api.ReplicationController
		// newRc is the "to" deployment
		newRc *api.ReplicationController
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
Scaling up foo-v2 from 0 to 20, scaling down foo-v1 from 10 to 0 (keep 20 pods available, don't exceed 70 pods)
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
Scaling up foo-v2 from 0 to 0, scaling down foo-v1 from 3 to 0 (keep 0 pods available, don't exceed 4 pods)
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
Scaling up foo-v2 from 0 to 0, scaling down foo-v1 from 3 to 0 (keep 0 pods available, don't exceed 3 pods)
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
Scaling up foo-v2 from 0 to 0, scaling down foo-v1 from 3 to 0 (keep 0 pods available, don't exceed 3 pods)
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
				up{2},
				down{oldReady: 1, newReady: 2, to: 0},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 2, scaling down foo-v1 from 30 to 0 (keep 1 pods available, don't exceed 30 pods)
Scaling foo-v1 down to 1
Scaling foo-v2 up to 2
Scaling foo-v1 down to 0
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
			name:        "1->1 100%/0 allow maxUnavailability",
			oldRc:       oldRc(1, 1),
			newRc:       newRc(0, 1),
			newRcExists: false,
			maxUnavail:  intstr.FromString("100%"),
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
	}

	for i, test := range tests {
		// Extract expectations into some makeshift FIFOs so they can be returned
		// in the correct order from the right places. This lets scale downs be
		// expressed a single event even though the data is used from multiple
		// interface calls.
		oldReady := []int{}
		newReady := []int{}
		upTo := []int{}
		downTo := []int{}
		for _, event := range test.expected {
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
		t.Logf("running test %d (%s) (up: %v, down: %v, oldReady: %v, newReady: %v)", i, test.name, upTo, downTo, oldReady, newReady)
		updater := &RollingUpdater{
			ns: "default",
			scaleAndWait: func(rc *api.ReplicationController, retry *RetryParams, wait *RetryParams) (*api.ReplicationController, error) {
				// Return a scale up or scale down expectation depending on the rc,
				// and throw errors if there is no expectation expressed for this
				// call.
				expected := -1
				switch {
				case rc == test.newRc:
					t.Logf("scaling up %s to %d", rc.Name, rc.Spec.Replicas)
					expected = next(&upTo)
				case rc == test.oldRc:
					t.Logf("scaling down %s to %d", rc.Name, rc.Spec.Replicas)
					expected = next(&downTo)
				}
				if expected == -1 {
					t.Fatalf("unexpected scale of %s to %d", rc.Name, rc.Spec.Replicas)
				} else if e, a := expected, rc.Spec.Replicas; e != a {
					t.Fatalf("expected scale of %s to %d, got %d", rc.Name, e, a)
				}
				// Simulate the scale.
				rc.Status.Replicas = rc.Spec.Replicas
				return rc, nil
			},
			getOrCreateTargetController: func(controller *api.ReplicationController, sourceId string) (*api.ReplicationController, bool, error) {
				// Simulate a create vs. update of an existing controller.
				return test.newRc, test.newRcExists, nil
			},
			cleanup: func(oldRc, newRc *api.ReplicationController, config *RollingUpdaterConfig) error {
				return nil
			},
		}
		// Set up a mock readiness check which handles the test assertions.
		updater.getReadyPods = func(oldRc, newRc *api.ReplicationController) (int, int, error) {
			// Return simulated readiness, and throw an error if this call has no
			// expectations defined.
			oldReady := next(&oldReady)
			newReady := next(&newReady)
			if oldReady == -1 || newReady == -1 {
				t.Fatalf("unexpected getReadyPods call for:\noldRc: %+v\nnewRc: %+v", oldRc, newRc)
			}
			return oldReady, newReady, nil
		}
		var buffer bytes.Buffer
		config := &RollingUpdaterConfig{
			Out:            &buffer,
			OldRc:          test.oldRc,
			NewRc:          test.newRc,
			UpdatePeriod:   0,
			Interval:       time.Millisecond,
			Timeout:        time.Millisecond,
			CleanupPolicy:  DeleteRollingUpdateCleanupPolicy,
			MaxUnavailable: test.maxUnavail,
			MaxSurge:       test.maxSurge,
		}
		err := updater.Update(config)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if buffer.String() != test.output {
			t.Errorf("Bad output. expected:\n%s\ngot:\n%s", test.output, buffer.String())
		}
	}
}

// TestUpdate_progressTimeout ensures that an update which isn't making any
// progress will eventually time out with a specified error.
func TestUpdate_progressTimeout(t *testing.T) {
	oldRc := oldRc(2, 2)
	newRc := newRc(0, 2)
	updater := &RollingUpdater{
		ns: "default",
		scaleAndWait: func(rc *api.ReplicationController, retry *RetryParams, wait *RetryParams) (*api.ReplicationController, error) {
			// Do nothing.
			return rc, nil
		},
		getOrCreateTargetController: func(controller *api.ReplicationController, sourceId string) (*api.ReplicationController, bool, error) {
			return newRc, false, nil
		},
		cleanup: func(oldRc, newRc *api.ReplicationController, config *RollingUpdaterConfig) error {
			return nil
		},
	}
	updater.getReadyPods = func(oldRc, newRc *api.ReplicationController) (int, int, error) {
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
	var updatedOldRc *api.ReplicationController
	fake := &testclient.Fake{}
	fake.AddReactor("*", "*", func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
		switch a := action.(type) {
		case testclient.GetAction:
			return true, oldRc, nil
		case testclient.UpdateAction:
			updatedOldRc = a.GetObject().(*api.ReplicationController)
			return true, updatedOldRc, nil
		}
		return false, nil, nil
	})
	updater := &RollingUpdater{
		c:  fake,
		ns: "default",
		scaleAndWait: func(rc *api.ReplicationController, retry *RetryParams, wait *RetryParams) (*api.ReplicationController, error) {
			return rc, nil
		},
		getOrCreateTargetController: func(controller *api.ReplicationController, sourceId string) (*api.ReplicationController, bool, error) {
			return newRc, false, nil
		},
		cleanup: func(oldRc, newRc *api.ReplicationController, config *RollingUpdaterConfig) error {
			return nil
		},
		getReadyPods: func(oldRc, newRc *api.ReplicationController) (int, int, error) {
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
	if updatedOldRc == nil {
		t.Fatalf("expected rc to be updated")
	}
	if e, a := "1", updatedOldRc.Annotations[originalReplicasAnnotation]; e != a {
		t.Fatalf("expected annotation value %s, got %s", e, a)
	}
}

func TestRollingUpdater_multipleContainersInPod(t *testing.T) {
	tests := []struct {
		oldRc *api.ReplicationController
		newRc *api.ReplicationController

		container     string
		image         string
		deploymentKey string
	}{
		{
			oldRc: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Spec: api.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "old",
					},
					Template: &api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"dk": "old",
							},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
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
			newRc: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Spec: api.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "old",
					},
					Template: &api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"dk": "old",
							},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
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
			oldRc: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Name: "bar",
				},
				Spec: api.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "old",
					},
					Template: &api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"dk": "old",
							},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Name:  "container1",
									Image: "image1",
								},
							},
						},
					},
				},
			},
			newRc: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Name: "bar",
				},
				Spec: api.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "old",
					},
					Template: &api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"dk": "old",
							},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
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

	for _, test := range tests {
		fake := &testclient.Fake{}
		fake.AddReactor("*", "*", func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
			switch action.(type) {
			case testclient.GetAction:
				return true, test.oldRc, nil
			}
			return false, nil, nil
		})

		codec := testapi.Default.Codec()

		deploymentHash, err := api.HashObject(test.newRc, codec)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		test.newRc.Spec.Selector[test.deploymentKey] = deploymentHash
		test.newRc.Spec.Template.Labels[test.deploymentKey] = deploymentHash
		test.newRc.Name = fmt.Sprintf("%s-%s", test.newRc.Name, deploymentHash)

		updatedRc, err := CreateNewControllerFromCurrentController(fake, codec, "", test.oldRc.ObjectMeta.Name, test.newRc.ObjectMeta.Name, test.image, test.container, test.deploymentKey)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(updatedRc, test.newRc) {
			t.Errorf("expected:\n%v\ngot:\n%v\n", test.newRc, updatedRc)
		}
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
		{
			name:      "rename",
			policy:    RenameRollingUpdateCleanupPolicy,
			responses: []runtime.Object{rcExisting},
			expected: []string{
				"get",
				"update",
				"get",
				"get",
				"delete",
				"create",
				"delete",
			},
		},
	}

	for _, test := range tests {
		fake := testclient.NewSimpleFake(test.responses...)
		updater := &RollingUpdater{
			ns: "default",
			c:  fake,
		}
		config := &RollingUpdaterConfig{
			Out:           ioutil.Discard,
			OldRc:         rc,
			NewRc:         rcExisting,
			UpdatePeriod:  0,
			Interval:      time.Millisecond,
			Timeout:       time.Millisecond,
			CleanupPolicy: test.policy,
		}
		err := updater.cleanupWithClients(rc, rcExisting, config)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(fake.Actions()) != len(test.expected) {
			t.Fatalf("%s: unexpected actions: %v, expected %v", test.name, fake.Actions(), test.expected)
		}
		for j, action := range fake.Actions() {
			if e, a := test.expected[j], action.GetVerb(); e != a {
				t.Errorf("%s: unexpected action: expected %s, got %s", test.name, e, a)
			}
		}
	}
}

func TestFindSourceController(t *testing.T) {
	ctrl1 := api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
			Annotations: map[string]string{
				sourceIdAnnotation: "bar:1234",
			},
		},
	}
	ctrl2 := api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "bar",
			Annotations: map[string]string{
				sourceIdAnnotation: "foo:12345",
			},
		},
	}
	ctrl3 := api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Annotations: map[string]string{
				sourceIdAnnotation: "baz:45667",
			},
		},
	}
	tests := []struct {
		list               *api.ReplicationControllerList
		expectedController *api.ReplicationController
		err                error
		name               string
		expectError        bool
	}{
		{
			list:        &api.ReplicationControllerList{},
			expectError: true,
		},
		{
			list: &api.ReplicationControllerList{
				Items: []api.ReplicationController{ctrl1},
			},
			name:        "foo",
			expectError: true,
		},
		{
			list: &api.ReplicationControllerList{
				Items: []api.ReplicationController{ctrl1},
			},
			name:               "bar",
			expectedController: &ctrl1,
		},
		{
			list: &api.ReplicationControllerList{
				Items: []api.ReplicationController{ctrl1, ctrl2},
			},
			name:               "bar",
			expectedController: &ctrl1,
		},
		{
			list: &api.ReplicationControllerList{
				Items: []api.ReplicationController{ctrl1, ctrl2},
			},
			name:               "foo",
			expectedController: &ctrl2,
		},
		{
			list: &api.ReplicationControllerList{
				Items: []api.ReplicationController{ctrl1, ctrl2, ctrl3},
			},
			name:               "baz",
			expectedController: &ctrl3,
		},
	}
	for _, test := range tests {
		fakeClient := testclient.NewSimpleFake(test.list)
		ctrl, err := FindSourceController(fakeClient, "default", test.name)
		if test.expectError && err == nil {
			t.Errorf("unexpected non-error")
		}
		if !test.expectError && err != nil {
			t.Errorf("unexpected error")
		}
		if !reflect.DeepEqual(ctrl, test.expectedController) {
			t.Errorf("expected:\n%v\ngot:\n%v\n", test.expectedController, ctrl)
		}
	}
}

func TestUpdateExistingReplicationController(t *testing.T) {
	tests := []struct {
		rc              *api.ReplicationController
		name            string
		deploymentKey   string
		deploymentValue string

		expectedRc *api.ReplicationController
		expectErr  bool
	}{
		{
			rc: &api.ReplicationController{
				Spec: api.ReplicationControllerSpec{
					Template: &api.PodTemplateSpec{},
				},
			},
			name:            "foo",
			deploymentKey:   "dk",
			deploymentValue: "some-hash",

			expectedRc: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						"kubectl.kubernetes.io/next-controller-id": "foo",
					},
				},
				Spec: api.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "some-hash",
					},
					Template: &api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"dk": "some-hash",
							},
						},
					},
				},
			},
		},
		{
			rc: &api.ReplicationController{
				Spec: api.ReplicationControllerSpec{
					Template: &api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
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

			expectedRc: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						"kubectl.kubernetes.io/next-controller-id": "foo",
					},
				},
				Spec: api.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "some-other-hash",
					},
					Template: &api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"dk": "some-other-hash",
							},
						},
					},
				},
			},
		},
	}
	for _, test := range tests {
		buffer := &bytes.Buffer{}
		fakeClient := testclient.NewSimpleFake(test.expectedRc)
		rc, err := UpdateExistingReplicationController(fakeClient, test.rc, "default", test.name, test.deploymentKey, test.deploymentValue, buffer)
		if !reflect.DeepEqual(rc, test.expectedRc) {
			t.Errorf("expected:\n%#v\ngot:\n%#v\n", test.expectedRc, rc)
		}
		if test.expectErr && err == nil {
			t.Errorf("unexpected non-error")
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}
}

func TestUpdateWithRetries(t *testing.T) {
	codec := testapi.Default.Codec()
	rc := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: "rc",
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		Spec: api.ReplicationControllerSpec{
			Selector: map[string]string{
				"foo": "bar",
			},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
				Spec: apitesting.DeepEqualSafePodSpec(),
			},
		},
	}

	// Test end to end updating of the rc with retries. Essentially make sure the update handler
	// sees the right updates, failures in update/get are handled properly, and that the updated
	// rc with new resource version is returned to the caller. Without any of these rollingupdate
	// will fail cryptically.
	newRc := *rc
	newRc.ResourceVersion = "2"
	newRc.Spec.Selector["baz"] = "foobar"
	updates := []*http.Response{
		{StatusCode: 500, Body: objBody(codec, &api.ReplicationController{})},
		{StatusCode: 500, Body: objBody(codec, &api.ReplicationController{})},
		{StatusCode: 200, Body: objBody(codec, &newRc)},
	}
	gets := []*http.Response{
		{StatusCode: 500, Body: objBody(codec, &api.ReplicationController{})},
		{StatusCode: 200, Body: objBody(codec, rc)},
	}
	fakeClient := &fake.RESTClient{
		Codec: codec,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == testapi.Default.ResourcePath("replicationcontrollers", "default", "rc") && m == "PUT":
				update := updates[0]
				updates = updates[1:]
				// We should always get an update with a valid rc even when the get fails. The rc should always
				// contain the update.
				if c, ok := readOrDie(t, req, codec).(*api.ReplicationController); !ok || !reflect.DeepEqual(rc, c) {
					t.Errorf("Unexpected update body, got %+v expected %+v", c, rc)
				} else if sel, ok := c.Spec.Selector["baz"]; !ok || sel != "foobar" {
					t.Errorf("Expected selector label update, got %+v", c.Spec.Selector)
				} else {
					delete(c.Spec.Selector, "baz")
				}
				return update, nil
			case p == testapi.Default.ResourcePath("replicationcontrollers", "default", "rc") && m == "GET":
				get := gets[0]
				gets = gets[1:]
				return get, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	clientConfig := &client.Config{ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}}
	client := client.NewOrDie(clientConfig)
	client.Client = fakeClient.Client

	if rc, err := updateWithRetries(
		client.ReplicationControllers("default"), rc, func(c *api.ReplicationController) {
			c.Spec.Selector["baz"] = "foobar"
		}); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else if sel, ok := rc.Spec.Selector["baz"]; !ok || sel != "foobar" || rc.ResourceVersion != "2" {
		t.Errorf("Expected updated rc, got %+v", rc)
	}
	if len(updates) != 0 || len(gets) != 0 {
		t.Errorf("Remaining updates %+v gets %+v", updates, gets)
	}
}

func readOrDie(t *testing.T, req *http.Request, codec runtime.Codec) runtime.Object {
	data, err := ioutil.ReadAll(req.Body)
	if err != nil {
		t.Errorf("Error reading: %v", err)
		t.FailNow()
	}
	obj, err := runtime.Decode(codec, data)
	if err != nil {
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
	codec := testapi.Default.Codec()
	rc := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: "rc"},
		Spec: api.ReplicationControllerSpec{
			Selector: map[string]string{
				"foo": "bar",
			},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
			},
		},
	}

	podList := &api.PodList{
		Items: []api.Pod{
			{ObjectMeta: api.ObjectMeta{Name: "foo"}},
			{ObjectMeta: api.ObjectMeta{Name: "bar"}},
			{ObjectMeta: api.ObjectMeta{Name: "baz"}},
		},
	}

	seen := sets.String{}
	updatedRc := false
	fakeClient := &fake.RESTClient{
		Codec: codec,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == testapi.Default.ResourcePath("pods", "default", "") && m == "GET":
				if req.URL.RawQuery != "labelSelector=foo%3Dbar" {
					t.Errorf("Unexpected query string: %s", req.URL.RawQuery)
				}
				return &http.Response{StatusCode: 200, Body: objBody(codec, podList)}, nil
			case p == testapi.Default.ResourcePath("pods", "default", "foo") && m == "PUT":
				seen.Insert("foo")
				obj := readOrDie(t, req, codec)
				podList.Items[0] = *(obj.(*api.Pod))
				return &http.Response{StatusCode: 200, Body: objBody(codec, &podList.Items[0])}, nil
			case p == testapi.Default.ResourcePath("pods", "default", "bar") && m == "PUT":
				seen.Insert("bar")
				obj := readOrDie(t, req, codec)
				podList.Items[1] = *(obj.(*api.Pod))
				return &http.Response{StatusCode: 200, Body: objBody(codec, &podList.Items[1])}, nil
			case p == testapi.Default.ResourcePath("pods", "default", "baz") && m == "PUT":
				seen.Insert("baz")
				obj := readOrDie(t, req, codec)
				podList.Items[2] = *(obj.(*api.Pod))
				return &http.Response{StatusCode: 200, Body: objBody(codec, &podList.Items[2])}, nil
			case p == testapi.Default.ResourcePath("replicationcontrollers", "default", "rc") && m == "PUT":
				updatedRc = true
				return &http.Response{StatusCode: 200, Body: objBody(codec, rc)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	clientConfig := &client.Config{ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}}
	client := client.NewOrDie(clientConfig)
	client.Client = fakeClient.Client

	if _, err := AddDeploymentKeyToReplicationController(rc, client, "dk", "hash", api.NamespaceDefault, buf); err != nil {
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
	mkpod := func(owner *api.ReplicationController, ready bool) *api.Pod {
		labels := map[string]string{}
		for k, v := range owner.Spec.Selector {
			labels[k] = v
		}
		status := api.ConditionTrue
		if !ready {
			status = api.ConditionFalse
		}
		return &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   "pod",
				Labels: labels,
			},
			Status: api.PodStatus{
				Conditions: []api.PodCondition{
					{
						Type:   api.PodReady,
						Status: status,
					},
				},
			},
		}
	}

	tests := []struct {
		oldRc *api.ReplicationController
		newRc *api.ReplicationController
		// expectated old/new ready counts
		oldReady int
		newReady int
		// pods owned by the rcs; indicate whether they're ready
		oldPods []bool
		newPods []bool
	}{
		{
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
	}

	for i, test := range tests {
		t.Logf("evaluating test %d", i)
		// Populate the fake client with pods associated with their owners.
		pods := []runtime.Object{}
		for _, ready := range test.oldPods {
			pods = append(pods, mkpod(test.oldRc, ready))
		}
		for _, ready := range test.newPods {
			pods = append(pods, mkpod(test.newRc, ready))
		}
		client := testclient.NewSimpleFake(pods...)

		updater := &RollingUpdater{
			ns: "default",
			c:  client,
		}
		oldReady, newReady, err := updater.readyPods(test.oldRc, test.newRc)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if e, a := test.oldReady, oldReady; e != a {
			t.Errorf("expected old ready %d, got %d", e, a)
		}
		if e, a := test.newReady, newReady; e != a {
			t.Errorf("expected new ready %d, got %d", e, a)
		}
	}
}
