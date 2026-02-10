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

func BenchmarkWatchCacheInitialization(b *testing.B) {
	// 1. Setup Data (Once)
	sharedPrefix := path.Join("/", uuid.New().String(), "registry")
	tCtx := ktesting.Init(b)

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

	// 2. Run Benchmarks
	b.ResetTimer()

	state := &benchmarkState{}

	b.Run("RangeStream=true", func(b *testing.B) {
		runWatchCacheBenchmark(b, sharedPrefix, true, state)
	})

	b.Run("RangeStream=false", func(b *testing.B) {
		runWatchCacheBenchmark(b, sharedPrefix, false, state)
	})
}

type benchmarkState struct {
	lastCount int
	lastSum   float64
}

func runWatchCacheBenchmark(b *testing.B, sharedPrefix string, rangeStreamEnabled bool, state *benchmarkState) {
	// User requested to run only once, ignoring b.N loop.
	featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, features.RangeStream, rangeStreamEnabled)
	tCtx := ktesting.Init(b)

	// Restart Server
	clientSet, _, tearDownFn := framework.StartTestServer(tCtx, b, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.Etcd.StorageConfig.Prefix = sharedPrefix
			opts.Etcd.EnableWatchCache = true
			opts.Etcd.WatchCacheSizes = []string{"secrets#5000"}
		},
	})
	defer tearDownFn()

	// Verify/Measure
	// We want to capture the initialization duration.
	// We can get it from the metric `etcd_watch_cache_initialization_duration_seconds`.

	duration, err := getWatchCacheInitDuration(tCtx, clientSet, "secrets", state)
	if err != nil {
		b.Errorf("Failed to get watch cache init duration: %v", err)
		return
	}

	b.ReportMetric(duration, "s/init")
}

func getWatchCacheInitDuration(ctx context.Context, client clientset.Interface, resource string, state *benchmarkState) (float64, error) {
	var duration float64
	err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		body, err := client.CoreV1().RESTClient().Get().AbsPath("/metrics").DoRaw(ctx)
		if err != nil {
			return false, err
		}

		// Look for etcd_watch_cache_initialization_duration_seconds_sum and count
		var sum float64
		var count int

		lines := strings.Split(string(body), "\n")
		for _, line := range lines {
			if strings.HasPrefix(line, "etcd_watch_cache_initialization_duration_seconds_sum") && strings.Contains(line, fmt.Sprintf("resource=\"%s\"", resource)) {
				parts := strings.Split(line, " ")
				if len(parts) == 2 {
					if s, err := strconv.ParseFloat(parts[1], 64); err == nil {
						sum = s
					}
				}
			}
			if strings.HasPrefix(line, "etcd_watch_cache_initialization_duration_seconds_count") && strings.Contains(line, fmt.Sprintf("resource=\"%s\"", resource)) {
				parts := strings.Split(line, " ")
				if len(parts) == 2 {
					if c, err := strconv.ParseFloat(parts[1], 64); err == nil {
						count = int(c)
					}
				}
			}
		}

		if count > state.lastCount {
			// Found new measurement(s)
			// Assuming single measurement added or we take the delta
			duration = sum - state.lastSum
			// Update state
			state.lastCount = count
			state.lastSum = sum
			return true, nil
		}
		return false, nil
	})
	return duration, err
}
