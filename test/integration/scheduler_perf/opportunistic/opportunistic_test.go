package opportunistic

import (
	"fmt"
	"os"
	"testing"

	_ "k8s.io/component-base/logs/json/register"
	perf "k8s.io/kubernetes/test/integration/scheduler_perf"
)

func TestMain(m *testing.M) {
	if err := perf.InitTests(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	m.Run()
}

func TestSchedulerPerf(t *testing.T) {
	perf.RunIntegrationPerfScheduling(t, "performance-config.yaml")
}

func BenchmarkPerfScheduling(b *testing.B) {
	perf.RunBenchmarkPerfScheduling(b, "performance-config.yaml", "opportunistic", nil)
}
