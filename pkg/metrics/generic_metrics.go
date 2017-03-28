/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package metrics

import (
	"fmt"
	"io"
	"reflect"
	"strings"

	"github.com/golang/glog"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
)

type Metrics map[string]model.Samples

func (m *Metrics) Equal(o Metrics) bool {
	leftKeySet := []string{}
	rightKeySet := []string{}
	for k := range *m {
		leftKeySet = append(leftKeySet, k)
	}
	for k := range o {
		rightKeySet = append(rightKeySet, k)
	}
	if !reflect.DeepEqual(leftKeySet, rightKeySet) {
		return false
	}
	for _, k := range leftKeySet {
		if !(*m)[k].Equal(o[k]) {
			return false
		}
	}
	return true
}

func PrintSample(sample *model.Sample) string {
	buf := make([]string, 0)
	// Id is a VERY special label. For 'normal' container it's usless, but it's necessary
	// for 'system' containers (e.g. /docker-daemon, /kubelet, etc.). We know if that's the
	// case by checking if there's a label "kubernetes_container_name" present. It's hacky
	// but it works...
	_, normalContainer := sample.Metric["kubernetes_container_name"]
	for k, v := range sample.Metric {
		if strings.HasPrefix(string(k), "__") {
			continue
		}

		if string(k) == "id" && normalContainer {
			continue
		}
		buf = append(buf, fmt.Sprintf("%v=%v", string(k), v))
	}
	return fmt.Sprintf("[%v] = %v", strings.Join(buf, ","), sample.Value)
}

func NewMetrics() Metrics {
	result := make(Metrics)
	return result
}

func parseMetrics(data string, output *Metrics) error {
	dec := expfmt.NewDecoder(strings.NewReader(data), expfmt.FmtText)
	decoder := expfmt.SampleDecoder{
		Dec:  dec,
		Opts: &expfmt.DecodeOptions{},
	}

	for {
		var v model.Vector
		if err := decoder.Decode(&v); err != nil {
			if err == io.EOF {
				// Expected loop termination condition.
				return nil
			}
			glog.Warningf("Invalid Decode. Skipping.")
			continue
		}
		for _, metric := range v {
			name := string(metric.Metric[model.MetricNameLabel])
			(*output)[name] = append((*output)[name], metric)
		}
	}
}

func (g *MetricsGrabber) getMetricsFromPod(podName string, namespace string, port int) (string, error) {
	rawOutput, err := g.client.Core().RESTClient().Get().
		Namespace(namespace).
		Resource("pods").
		SubResource("proxy").
		Name(fmt.Sprintf("%v:%v", podName, port)).
		Suffix("metrics").
		Do().Raw()
	if err != nil {
		return "", err
	}
	return string(rawOutput), nil
}
