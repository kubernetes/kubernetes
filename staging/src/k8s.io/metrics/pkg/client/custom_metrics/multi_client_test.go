/*
Copyright 2020 The Kubernetes Authors.

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

package custom_metrics

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/discovery/fake"
	cmint "k8s.io/metrics/pkg/apis/custom_metrics"
)

type fakeDiscovery struct {
	*fake.FakeDiscovery

	groupList *metav1.APIGroupList
}

func (c *fakeDiscovery) ServerGroups() (*metav1.APIGroupList, error) {
	if c.groupList == nil {
		return nil, errors.New("doesn't exist")
	}
	return c.groupList, nil
}

func TestPeriodicallyInvalidate(t *testing.T) {
	fake := &fakeDiscovery{
		groupList: &metav1.APIGroupList{
			Groups: []metav1.APIGroup{{
				Name: cmint.GroupName,
				Versions: []metav1.GroupVersionForDiscovery{{
					GroupVersion: cmint.SchemeGroupVersion.String(),
				}},
			}},
		},
	}

	apiVersionsGetter := NewAvailableAPIsGetter(fake)
	cache := apiVersionsGetter.(*apiVersionsFromDiscovery)
	stopCh := make(chan struct{})
	defer close(stopCh)

	_, err := apiVersionsGetter.PreferredVersion()
	require.NoError(t, err)
	require.NotNil(t, cache.prefVersion)
	require.Equal(t, cache.prefVersion.Group, cmint.GroupName)

	go PeriodicallyInvalidate(
		apiVersionsGetter,
		200*time.Millisecond,
		stopCh,
	)

	// Wait for cache invalidation.
	time.Sleep(1 * time.Second)

	cache.mu.Lock()
	defer cache.mu.Unlock()
	require.Nil(t, cache.prefVersion)
}
