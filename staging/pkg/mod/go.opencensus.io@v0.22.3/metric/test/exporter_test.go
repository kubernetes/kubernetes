package test

import (
	"context"
	"fmt"

	"go.opencensus.io/metric"
	"go.opencensus.io/metric/metricdata"
	"go.opencensus.io/metric/metricexport"
	"go.opencensus.io/stats"
	"go.opencensus.io/stats/view"
	"go.opencensus.io/tag"
)

var (
	myTag    = tag.MustNewKey("my_label")
	myMetric = stats.Int64("my_metric", "description", stats.UnitDimensionless)
)

func init() {
	if err := view.Register(
		&view.View{
			Measure:     myMetric,
			TagKeys:     []tag.Key{myTag},
			Aggregation: view.Sum(),
		},
	); err != nil {
		panic(err)
	}
}

func ExampleExporter_stats() {
	metricReader := metricexport.NewReader()
	metrics := NewExporter(metricReader)
	metrics.ReadAndExport()
	metricBase := getCounter(metrics, myMetric.Name(), newMetricKey("label1"))

	for i := 1; i <= 3; i++ {
		// The code under test begins here.
		stats.RecordWithTags(context.Background(),
			[]tag.Mutator{tag.Upsert(myTag, "label1")},
			myMetric.M(int64(i)))
		// The code under test ends here.

		metrics.ReadAndExport()
		metricValue := getCounter(metrics, myMetric.Name(), newMetricKey("label1"))
		fmt.Printf("increased by %d\n", metricValue-metricBase)
	}
	// Output:
	// increased by 1
	// increased by 3
	// increased by 6
}

type derivedMetric struct {
	i int64
}

func (m *derivedMetric) ToInt64() int64 {
	return m.i
}

func ExampleExporter_metric() {
	metricReader := metricexport.NewReader()
	metrics := NewExporter(metricReader)
	m := derivedMetric{}
	r := metric.NewRegistry()
	g, _ := r.AddInt64DerivedCumulative("derived", metric.WithLabelKeys(myTag.Name()))
	g.UpsertEntry(m.ToInt64, metricdata.NewLabelValue("l1"))
	for i := 1; i <= 3; i++ {
		// The code under test begins here.
		m.i = int64(i)
		// The code under test ends here.

		metrics.ExportMetrics(context.Background(), r.Read())
		metricValue := getCounter(metrics, "derived", newMetricKey("l1"))
		fmt.Println(metricValue)
	}
	// Output:
	// 1
	// 2
	// 3
}

func newMetricKey(v string) map[string]string {
	return map[string]string{myTag.Name(): v}
}

func getCounter(metrics *Exporter, metricName string, metricKey map[string]string) int64 {
	p, ok := metrics.GetPoint(metricName, metricKey)
	if !ok {
		// This is expected before the metric is recorded the first time.
		return 0
	}
	return p.Value.(int64)
}
