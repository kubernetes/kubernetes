package benchmark

type metricCollector interface {
	collectMetrics() []summary
}

// summary is the summary of a particular metric, such as schedulingThroughput.
type summary struct {
	name    string  `json:"name"`
	Average float64 `json:"average"`
	Perc50  float64 `json:"perc50"`
	Perc90  float64 `json:"perc90"`
	Perc99  float64 `json:"perc99"`
}

// Some other dependencies are needed by this collector. But the user should only need to specify the
// metric list.
type prometheusMetricCollector struct {
	// The list of scheduler metrics to collect. For each metric, we calculate avg, p50, p90 and p99.
	metrics []string
}

func (c *prometheusMetricCollector) collectMetrics() []summary {
	return nil
}

type scheduleThroughputCollector struct {

}

func (c *scheduleThroughputCollector) collectMetrics() []summary {
	return nil
}

type myCollector struct {

}

func (c *myCollector) collectMetrics() []summary {
	return nil
}