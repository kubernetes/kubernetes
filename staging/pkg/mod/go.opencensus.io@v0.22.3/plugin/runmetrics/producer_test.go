package runmetrics_test

import (
	"context"
	"github.com/stretchr/testify/assert"
	"go.opencensus.io/metric/metricdata"
	"go.opencensus.io/metric/metricexport"
	"go.opencensus.io/metric/metricproducer"
	"go.opencensus.io/plugin/runmetrics"
	"testing"
)

type testExporter struct {
	data []*metricdata.Metric
}

func (t *testExporter) ExportMetrics(ctx context.Context, data []*metricdata.Metric) error {
	t.data = append(t.data, data...)
	return nil
}

func TestEnable(t *testing.T) {
	tests := []struct {
		name                string
		options             runmetrics.RunMetricOptions
		wantMetricNames     [][]string
		dontWantMetricNames [][]string
	}{
		{
			"no stats",
			runmetrics.RunMetricOptions{
				EnableCPU:    false,
				EnableMemory: false,
			},
			[][]string{},
			[][]string{},
		},
		{
			"cpu and memory stats",
			runmetrics.RunMetricOptions{
				EnableCPU:    true,
				EnableMemory: true,
			},
			[][]string{
				{"process/memory_alloc", "process/total_memory_alloc", "process/sys_memory_alloc", "process/memory_lookups", "process/memory_malloc", "process/memory_frees"},
				{"process/heap_alloc", "process/sys_heap", "process/heap_idle", "process/heap_inuse", "process/heap_objects", "process/heap_release"},
				{"process/stack_inuse", "process/sys_stack", "process/stack_mspan_inuse", "process/sys_stack_mspan", "process/stack_mcache_inuse", "process/sys_stack_mcache"},
				{"process/cpu_goroutines", "process/cpu_cgo_calls"},
			},
			[][]string{},
		},
		{
			"only cpu stats",
			runmetrics.RunMetricOptions{
				EnableCPU:    true,
				EnableMemory: false,
			},
			[][]string{
				{"process/cpu_goroutines", "process/cpu_cgo_calls"},
			},
			[][]string{
				{"process/memory_alloc", "process/total_memory_alloc", "process/sys_memory_alloc", "process/memory_lookups", "process/memory_malloc", "process/memory_frees"},
				{"process/heap_alloc", "process/sys_heap", "process/heap_idle", "process/heap_inuse", "process/heap_objects", "process/heap_release"},
				{"process/stack_inuse", "process/sys_stack", "process/stack_mspan_inuse", "process/sys_stack_mspan", "process/stack_mcache_inuse", "process/sys_stack_mcache"},
			},
		},
		{
			"only memory stats",
			runmetrics.RunMetricOptions{
				EnableCPU:    false,
				EnableMemory: true,
			},
			[][]string{
				{"process/memory_alloc", "process/total_memory_alloc", "process/sys_memory_alloc", "process/memory_lookups", "process/memory_malloc", "process/memory_frees"},
				{"process/heap_alloc", "process/sys_heap", "process/heap_idle", "process/heap_inuse", "process/heap_objects", "process/heap_release"},
				{"process/stack_inuse", "process/sys_stack", "process/stack_mspan_inuse", "process/sys_stack_mspan", "process/stack_mcache_inuse", "process/sys_stack_mcache"},
			},
			[][]string{
				{"process/cpu_goroutines", "process/cpu_cgo_calls"},
			},
		},
		{
			"cpu and memory stats with custom prefix",
			runmetrics.RunMetricOptions{
				EnableCPU:    true,
				EnableMemory: true,
				Prefix:       "test_",
			},
			[][]string{
				{"test_process/memory_alloc", "test_process/total_memory_alloc", "test_process/sys_memory_alloc", "test_process/memory_lookups", "test_process/memory_malloc", "test_process/memory_frees"},
				{"test_process/heap_alloc", "test_process/sys_heap", "test_process/heap_idle", "test_process/heap_inuse", "test_process/heap_objects", "test_process/heap_release"},
				{"test_process/stack_inuse", "test_process/sys_stack", "test_process/stack_mspan_inuse", "test_process/sys_stack_mspan", "test_process/stack_mcache_inuse", "test_process/sys_stack_mcache"},
				{"test_process/cpu_goroutines", "test_process/cpu_cgo_calls"},
			},
			[][]string{},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {

			err := runmetrics.Enable(test.options)

			if err != nil {
				t.Errorf("want: nil, got: %v", err)
			}

			defer runmetrics.Disable()

			exporter := &testExporter{}
			reader := metricexport.NewReader()
			reader.ReadAndExport(exporter)

			for _, want := range test.wantMetricNames {
				assertNames(t, true, exporter, want)
			}

			for _, dontWant := range test.dontWantMetricNames {
				assertNames(t, false, exporter, dontWant)
			}
		})
	}
}

func assertNames(t *testing.T, wantIncluded bool, exporter *testExporter, expectedNames []string) {
	t.Helper()

	metricNames := make([]string, 0)
	for _, v := range exporter.data {
		metricNames = append(metricNames, v.Descriptor.Name)
	}

	for _, want := range expectedNames {
		if wantIncluded {
			assert.Contains(t, metricNames, want)
		} else {
			assert.NotContains(t, metricNames, want)
		}
	}
}

func TestEnable_RegistersWithGlobalManager(t *testing.T) {
	err := runmetrics.Enable(runmetrics.RunMetricOptions{})
	if err != nil {
		t.Errorf("want: nil, got: %v", err)
	}

	registeredCount := len(metricproducer.GlobalManager().GetAll())
	assert.Equal(t, 1, registeredCount, "expected a producer to be registered")
}

func TestEnable_RegistersNoDuplicates(t *testing.T) {
	err := runmetrics.Enable(runmetrics.RunMetricOptions{})
	if err != nil {
		t.Errorf("want: nil, got: %v", err)
	}

	err = runmetrics.Enable(runmetrics.RunMetricOptions{})
	if err != nil {
		t.Errorf("want: nil, got: %v", err)
	}

	producerCount := len(metricproducer.GlobalManager().GetAll())
	assert.Equal(t, 1, producerCount, "expected one registered producer")
}

func TestDisable(t *testing.T) {
	err := runmetrics.Enable(runmetrics.RunMetricOptions{})
	if err != nil {
		t.Errorf("want: nil, got: %v", err)
	}

	runmetrics.Disable()

	producerCount := len(metricproducer.GlobalManager().GetAll())
	assert.Equal(t, 0, producerCount, "expected one registered producer")
}
