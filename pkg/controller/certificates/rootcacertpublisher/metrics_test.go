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

package rootcacertpublisher

import (
	"errors"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestSyncCounter(t *testing.T) {
	testCases := []struct {
		desc    string
		err     error
		metrics []string
		want    string
	}{
		{
			desc: "nil error",
			err:  nil,
			metrics: []string{
				"root_ca_cert_publisher_sync_total",
			},
			want: `
# HELP root_ca_cert_publisher_sync_total [ALPHA] Number of namespace syncs happened in root ca cert publisher.
# TYPE root_ca_cert_publisher_sync_total counter
root_ca_cert_publisher_sync_total{code="200"} 1
				`,
		},
		{
			desc: "kube api error",
			err:  apierrors.NewNotFound(corev1.Resource("configmap"), "test-configmap"),
			metrics: []string{
				"root_ca_cert_publisher_sync_total",
			},
			want: `
# HELP root_ca_cert_publisher_sync_total [ALPHA] Number of namespace syncs happened in root ca cert publisher.
# TYPE root_ca_cert_publisher_sync_total counter
root_ca_cert_publisher_sync_total{code="404"} 1
				`,
		},
		{
			desc: "kube api error without code",
			err:  &apierrors.StatusError{},
			metrics: []string{
				"root_ca_cert_publisher_sync_total",
			},
			want: `
# HELP root_ca_cert_publisher_sync_total [ALPHA] Number of namespace syncs happened in root ca cert publisher.
# TYPE root_ca_cert_publisher_sync_total counter
root_ca_cert_publisher_sync_total{code="500"} 1
				`,
		},
		{
			desc: "general error",
			err:  errors.New("test"),
			metrics: []string{
				"root_ca_cert_publisher_sync_total",
			},
			want: `
# HELP root_ca_cert_publisher_sync_total [ALPHA] Number of namespace syncs happened in root ca cert publisher.
# TYPE root_ca_cert_publisher_sync_total counter
root_ca_cert_publisher_sync_total{code="500"} 1
				`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			recordMetrics(time.Now(), tc.err)
			defer syncCounter.Reset()
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tc.want), tc.metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}
