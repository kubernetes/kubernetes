package servicecacertpublisher

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
				"service_ca_cert_publisher_sync_total",
			},
			want: `
# HELP service_ca_cert_publisher_sync_total [ALPHA] Number of namespace syncs happened in service ca cert publisher.
# TYPE service_ca_cert_publisher_sync_total counter
service_ca_cert_publisher_sync_total{code="200"} 1
				`,
		},
		{
			desc: "kube api error",
			err:  apierrors.NewNotFound(corev1.Resource("configmap"), "test-configmap"),
			metrics: []string{
				"service_ca_cert_publisher_sync_total",
			},
			want: `
# HELP service_ca_cert_publisher_sync_total [ALPHA] Number of namespace syncs happened in service ca cert publisher.
# TYPE service_ca_cert_publisher_sync_total counter
service_ca_cert_publisher_sync_total{code="404"} 1
				`,
		},
		{
			desc: "kube api error without code",
			err:  &apierrors.StatusError{},
			metrics: []string{
				"service_ca_cert_publisher_sync_total",
			},
			want: `
# HELP service_ca_cert_publisher_sync_total [ALPHA] Number of namespace syncs happened in service ca cert publisher.
# TYPE service_ca_cert_publisher_sync_total counter
service_ca_cert_publisher_sync_total{code="500"} 1
				`,
		},
		{
			desc: "general error",
			err:  errors.New("test"),
			metrics: []string{
				"service_ca_cert_publisher_sync_total",
			},
			want: `
# HELP service_ca_cert_publisher_sync_total [ALPHA] Number of namespace syncs happened in service ca cert publisher.
# TYPE service_ca_cert_publisher_sync_total counter
service_ca_cert_publisher_sync_total{code="500"} 1
				`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			recordMetrics(time.Now(), "test-ns", tc.err)
			defer syncCounter.Reset()
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tc.want), tc.metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}
