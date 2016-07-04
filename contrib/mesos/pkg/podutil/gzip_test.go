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

package podutil

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
)

func TestGzipList(t *testing.T) {
	// pod spec defaults are written during deserialization, this is what we
	// expect them to be
	period := int64(v1.DefaultTerminationGracePeriodSeconds)
	defaultSpec := api.PodSpec{
		DNSPolicy:                     api.DNSClusterFirst,
		RestartPolicy:                 api.RestartPolicyAlways,
		TerminationGracePeriodSeconds: &period,
		SecurityContext:               new(api.PodSecurityContext),
	}
	list := &api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
				},
			},
			{
				ObjectMeta: api.ObjectMeta{
					Name:      "qax",
					Namespace: "lkj",
				},
			},
		},
	}

	amap := map[string]string{
		"crazy": "horse",
	}
	annotator := Annotator(amap)
	raw, err := Gzip(annotator.Do(Stream(list, nil)))
	assert.NoError(t, err)

	list2, err := gunzipList(raw)
	assert.NoError(t, err)

	list.Items[0].Spec = defaultSpec
	list.Items[0].Annotations = amap
	list.Items[1].Spec = defaultSpec
	list.Items[1].Annotations = amap
	assert.True(t, api.Semantic.DeepEqual(*list, *list2), "expected %+v instead of %+v", *list, *list2)
}
