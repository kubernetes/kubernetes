package metrics

import (
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
)

func TestRecordCacherMetrics(t *testing.T) {
	registry := metrics.NewKubeRegistry()

	sinceInSeconds = func(t time.Time) float64 {
		return time.Unix(0, 500*int64(time.Millisecond)).Sub(t).Seconds()
	}

	testedMetrics := []metrics.Registerable{
		cacherIncomingQueueBlockDuration,
		cacheWatcherInputQueueBlockDuration,
	}


	testedMetricsName := make([]string, 0, len(testedMetrics))
	for _, m := range testedMetrics {
		registry.MustRegister(m)
		testedMetricsName = append(testedMetricsName, m.FQName())
	}

	testCases := []struct {
		desc          string
		isIncoming    bool
		groupResource schema.GroupResource
		startTime     time.Time
		want          string
	}{
		{
			desc:          "incoming_queue_block",
			isIncoming:    true,
			groupResource: schema.GroupResource{Group: "foo", Resource: "bar"},
			startTime:     time.Unix(0, 0),

			want: `# HELP apiserver_watch_cache_incoming_queue_block_duration_seconds [ALPHA] Time spent waiting to write to the incoming channel in Cacher.
# TYPE apiserver_watch_cache_incoming_queue_block_duration_seconds histogram
apiserver_watch_cache_incoming_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="0.001"} 0
apiserver_watch_cache_incoming_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="0.005"} 0
apiserver_watch_cache_incoming_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="0.025"} 0
apiserver_watch_cache_incoming_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="0.05"} 0
apiserver_watch_cache_incoming_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="0.1"} 0
apiserver_watch_cache_incoming_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="0.2"} 0
apiserver_watch_cache_incoming_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="0.4"} 0
apiserver_watch_cache_incoming_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="0.6"} 1
apiserver_watch_cache_incoming_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="0.8"} 1
apiserver_watch_cache_incoming_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="1"} 1
apiserver_watch_cache_incoming_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="1.25"} 1
apiserver_watch_cache_incoming_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="1.5"} 1
apiserver_watch_cache_incoming_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="2"} 1
apiserver_watch_cache_incoming_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="3"} 1
apiserver_watch_cache_incoming_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="5"} 1
apiserver_watch_cache_incoming_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="10"} 1
apiserver_watch_cache_incoming_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="+Inf"} 1
apiserver_watch_cache_incoming_queue_block_duration_seconds_sum{group="foo",resource="bar"} 0.5
apiserver_watch_cache_incoming_queue_block_duration_seconds_count{group="foo",resource="bar"} 1
`,
		},
		{
			desc:          "cache_watcher_input_queue_block",
			isIncoming:    false,
			groupResource: schema.GroupResource{Group: "foo", Resource: "bar"},
			startTime:     time.Unix(0, 0),

			want: `# HELP apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds [ALPHA] Time spent waiting to write to the input channel of a cache watcher.
# TYPE apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds histogram
apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="0.001"}	0
apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="0.005"}	0
apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="0.025"}	0
apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="0.05"}	0
apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="0.1"}	0
apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="0.2"}	0
apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="0.4"}	0
apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="0.6"}	1
apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="0.8"}	1
apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="1"}	1
apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="1.25"}	1
apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="1.5"}	1
apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="2"}	1
apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="3"}	1
apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="5"}	1
apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="10"}	1
apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds_bucket{group="foo",resource="bar",le="+Inf"}	1
apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds_sum{group="foo",resource="bar"} 0.5
apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds_count{group="foo",resource="bar"}	1
`,
		},
	}

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			defer registry.Reset()
			if test.isIncoming {
				RecordCacherIncomingQueueBlock(test.groupResource, test.startTime)
			} else {
				RecordCacheWatcherInputQueueBlock(test.groupResource, test.startTime)
			}
			if err := testutil.GatherAndCompare(registry, strings.NewReader(test.want), testedMetricsName...); err != nil {
				t.Fatal(err)
			}
		})
	}
}
