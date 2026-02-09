package apiserver

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"testing"

	"path"

	"github.com/google/uuid"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestRangeStreamList(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RangeStream, true)
	tCtx := ktesting.Init(t)

	// Use a shared prefix so we can restart the apiserver against the same data.
	sharedPrefix := path.Join("/", uuid.New().String(), "registry")

	// 1. Setup Server and Populate Data
	setupClientSet, _, setupTearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.Etcd.StorageConfig.Prefix = sharedPrefix
		},
	})

	ns := framework.CreateNamespaceOrDie(setupClientSet, "rangestream-test", t)

	count := 200
	t.Logf("Creating %d secrets", count)
	for i := 0; i < count; i++ {
		secret := &v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("secret-%d", i),
			},
			Data: map[string][]byte{
				"data": []byte("some-data"),
			},
		}
		_, err := setupClientSet.CoreV1().Secrets(ns.Name).Create(context.Background(), secret, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create secret %d: %v", i, err)
		}
	}
	setupTearDownFn()

	// 2. Restart Server and Verify Watch Cache Init
	clientSet, _, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.Etcd.StorageConfig.Prefix = sharedPrefix
			opts.Etcd.EnableWatchCache = true
			// Ensure cache size is sufficient
			opts.Etcd.WatchCacheSizes = []string{"secrets#1000"}
		},
	})
	defer tearDownFn()

	t.Logf("Listing secrets to trigger watch cache initialization")
	// List with RV=0 to force wait for watch cache sync
	list, err := clientSet.CoreV1().Secrets(ns.Name).List(context.Background(), metav1.ListOptions{ResourceVersion: "0"})
	if err != nil {
		t.Fatalf("Failed to list secrets: %v", err)
	}

	if len(list.Items) != count {
		t.Errorf("Expected %d secrets, got %d", count, len(list.Items))
	}

	// Verify RangeStream was used via metrics
	if err := verifyRangeStreamMetric(clientSet, "secrets"); err != nil {
		t.Errorf("Failed to verify RangeStream metric: %v", err)
	}
}

func verifyRangeStreamMetric(client clientset.Interface, resource string) error {
	body, err := client.CoreV1().RESTClient().Get().AbsPath("/metrics").DoRaw(context.Background())
	if err != nil {
		return err
	}

	// Look for etcd_request_duration_seconds_count{group="",operation="listStream",resource="secrets"}
	// Note: The metric output format might slightly vary in ordering of labels.
	// We scan for the line containing our expected labels.
	var validLines []string
	for _, line := range strings.Split(string(body), "\n") {
		if strings.HasPrefix(line, "etcd_request_duration_seconds_count") {
			validLines = append(validLines, line)
			// Check for presence of key labels independent of order
			if strings.Contains(line, fmt.Sprintf("operation=\"listStream\"")) &&
				strings.Contains(line, fmt.Sprintf("resource=\"%s\"", resource)) {
				// Parse value
				parts := strings.Split(line, " ")
				if len(parts) != 2 {
					return fmt.Errorf("unexpected metric format: %s", line)
				}
				val, err := strconv.ParseFloat(parts[1], 64)
				if err != nil {
					return fmt.Errorf("failed to parse metric value: %v", err)
				}
				if val > 0 {
					return nil
				}
				return fmt.Errorf("metric found but value is 0: %s", line)
			}
		}
	}
	return fmt.Errorf("metric for listStream not found in:\n%s", strings.Join(validLines, "\n"))
}
