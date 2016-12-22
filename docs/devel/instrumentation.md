## Instrumenting Kubernetes with a new metric

The following is a step-by-step guide for adding a new metric to the Kubernetes
code base.

We use the Prometheus monitoring system's golang client library for
instrumenting our code. Once you've picked out a file that you want to add a
metric to, you should:

1. Import "github.com/prometheus/client_golang/prometheus".

2. Create a top-level var to define the metric. For this, you have to:

  1. Pick the type of metric. Use a Gauge for things you want to set to a
particular value, a Counter for things you want to increment, or a Histogram or
Summary for histograms/distributions of values (typically for latency).
Histograms are better if you're going to aggregate the values across jobs, while
summaries are better if you just want the job to give you a useful summary of
the values.
  2. Give the metric a name and description.
  3. Pick whether you want to distinguish different categories of things using
labels on the metric. If so, add "Vec" to the name of the type of metric you
want and add a slice of the label names to the definition.

   https://github.com/kubernetes/kubernetes/blob/cd3299307d44665564e1a5c77d0daa0286603ff5/pkg/apiserver/apiserver.go#L53
   https://github.com/kubernetes/kubernetes/blob/cd3299307d44665564e1a5c77d0daa0286603ff5/pkg/kubelet/metrics/metrics.go#L31

3. Register the metric so that prometheus will know to export it.

   https://github.com/kubernetes/kubernetes/blob/cd3299307d44665564e1a5c77d0daa0286603ff5/pkg/kubelet/metrics/metrics.go#L74
   https://github.com/kubernetes/kubernetes/blob/cd3299307d44665564e1a5c77d0daa0286603ff5/pkg/apiserver/apiserver.go#L78

4. Use the metric by calling the appropriate method for your metric type (Set,
Inc/Add, or Observe, respectively for Gauge, Counter, or Histogram/Summary),
first calling WithLabelValues if your metric has any labels

   https://github.com/kubernetes/kubernetes/blob/3ce7fe8310ff081dbbd3d95490193e1d5250d2c9/pkg/kubelet/kubelet.go#L1384
   https://github.com/kubernetes/kubernetes/blob/cd3299307d44665564e1a5c77d0daa0286603ff5/pkg/apiserver/apiserver.go#L87


These are the metric type definitions if you're curious to learn about them or
need more information:

https://github.com/prometheus/client_golang/blob/master/prometheus/gauge.go
https://github.com/prometheus/client_golang/blob/master/prometheus/counter.go
https://github.com/prometheus/client_golang/blob/master/prometheus/histogram.go
https://github.com/prometheus/client_golang/blob/master/prometheus/summary.go


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/instrumentation.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
