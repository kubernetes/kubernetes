package apiserver

import (
	"context"
	"fmt"
	"path"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func BenchmarkWatchCacheInitializationRangeStream(b *testing.B) {
	runWatchCacheInitBenchmark(b, true)
}

func BenchmarkWatchCacheInitializationPaginated(b *testing.B) {
	runWatchCacheInitBenchmark(b, false)
}

func runWatchCacheInitBenchmark(b *testing.B, rangeStreamEnabled bool) {
	featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, features.RangeStream, rangeStreamEnabled)
	tCtx := ktesting.Init(b)
	sharedPrefix := path.Join("/", uuid.New().String(), "registry")

	// 1. Setup data
	setupClientSet, _, setupTearDownFn := framework.StartTestServer(tCtx, b, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.Etcd.StorageConfig.Prefix = sharedPrefix
		},
	})

	ns := framework.CreateNamespaceOrDie(setupClientSet, "benchmark-test", b)
	count := 2000
	b.Logf("Creating %d secrets", count)
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
			b.Fatalf("Failed to create secret %d: %v", i, err)
		}
	}
	setupTearDownFn()

	b.ResetTimer()

	// 2. Restart server and measure watch cache initialization
	clientSet, _, tearDownFn := framework.StartTestServer(tCtx, b, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.Etcd.StorageConfig.Prefix = sharedPrefix
			opts.Etcd.EnableWatchCache = true
			opts.Etcd.WatchCacheSizes = []string{"secrets#5000"}
		},
	})
	defer tearDownFn()

	duration, _, _, err := getWatchCacheInitDuration(tCtx, clientSet, "secrets", 0, 0)
	if err != nil {
		b.Errorf("Failed to get watch cache init duration: %v", err)
		return
	}
	b.ReportMetric(duration, "s/init")
}

// getWatchCacheInitDuration waits for the watch cache initialization metric to
// exceed prevCount and returns the incremental duration and new count/sum.
func getWatchCacheInitDuration(ctx context.Context, client clientset.Interface, resource string, prevCount int, prevSum float64) (float64, int, float64, error) {
	var duration float64
	var newCount int
	var newSum float64
	err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		count, sum, err := readWatchCacheInitMetric(ctx, client, resource)
		if err != nil {
			return false, nil
		}
		if count > prevCount {
			newCount = count
			newSum = sum
			duration = (sum - prevSum) / float64(count-prevCount)
			return true, nil
		}
		return false, nil
	})
	return duration, newCount, newSum, err
}

func readWatchCacheInitMetric(ctx context.Context, client clientset.Interface, resource string) (count int, sum float64, err error) {
	body, err := client.CoreV1().RESTClient().Get().AbsPath("/metrics").DoRaw(ctx)
	if err != nil {
		return 0, 0, err
	}
	for _, line := range strings.Split(string(body), "\n") {
		if !strings.Contains(line, fmt.Sprintf("resource=\"%s\"", resource)) {
			continue
		}
		parts := strings.SplitN(line, " ", 2)
		if len(parts) != 2 {
			continue
		}
		val, parseErr := strconv.ParseFloat(parts[1], 64)
		if parseErr != nil {
			continue
		}
		if strings.HasPrefix(line, "etcd_watch_cache_initialization_duration_seconds_count") {
			count = int(val)
		} else if strings.HasPrefix(line, "etcd_watch_cache_initialization_duration_seconds_sum") {
			sum = val
		}
	}
	return count, sum, nil
}
