/*
Copyright 2020 The Kubernetes Authors.

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

package main

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"text/template"
	"time"

	"gopkg.in/yaml.v2"

	"k8s.io/component-base/metrics"
)

var (
	GOROOT    string = os.Getenv("GOROOT")
	GOOS      string = os.Getenv("GOOS")
	KUBE_ROOT string = os.Getenv("KUBE_ROOT")
)

const (
	templ = `---
title: Kubernetes Metrics Across Components
content_type: instrumentation
---


## Metrics

These are the metrics which are exported in Kubernetes components (i.e. kube-apiserver, scheduler, kube-controller-manager, kube-proxy, cloud-controller-manager). 

(auto-generated {{.GeneratedDate.Format "2006 Jan 02"}})

### List of Kubernetes Metrics

<table class="table">
<thead>
	<tr>
		<td width="20%">Name</td>
		<td width="12%">Stability Level</td>
		<td width="12%">Type</td>
		<td width="30%">Help</td>
		<td width="13%">Labels</td>
		<td width="13%">Const Labels</td>
	</tr>
</thead>
<tbody>
{{range $index, $metric := .Metrics}}<tr><td>{{$metric.Name}}</td><td>{{$metric.StabilityLevel}}</td><td>{{$metric.Type}}</td><td>{{$metric.Help}}</td>{{if not $metric.Labels }}<td>None</td>{{else }}<td>{{range $label := $metric.Labels}}<div>{{$label}}</div>{{end}}</td>{{end}}{{if not $metric.ConstLabels }}<td>None</td>{{else }}<td>{{$metric.ConstLabels}}</td>{{end}}</tr>
{{end}}
</tbody>
</table>
`
)

type templateData struct {
	Metrics       []metric
	GeneratedDate time.Time
}

func main() {
	dat, err := os.ReadFile("test/instrumentation/testdata/documentation-list.yaml")
	if err == nil {
		metrics := []metric{}
		err = yaml.Unmarshal(dat, &metrics)
		if err != nil {
			println("err", err)
		}
		t := template.New("t")
		t, err := t.Parse(templ)
		if err != nil {
			println("err", err)
		}
		var tpl bytes.Buffer
		for i, m := range metrics {
			m.Help = strings.Join(strings.Split(m.Help, "\n"), " ")
			metrics[i] = m
		}
		data := templateData{
			Metrics:       metrics,
			GeneratedDate: time.Now(),
		}
		err = t.Execute(&tpl, data)
		if err != nil {
			println("err", err)
		}
		fmt.Print(tpl.String())
	} else {
		fmt.Fprintf(os.Stderr, "%s\n", err)
	}

}

type metric struct {
	Name              string              `yaml:"name" json:"name"`
	Subsystem         string              `yaml:"subsystem,omitempty" json:"subsystem,omitempty"`
	Namespace         string              `yaml:"namespace,omitempty" json:"namespace,omitempty"`
	Help              string              `yaml:"help,omitempty" json:"help,omitempty"`
	Type              string              `yaml:"type,omitempty" json:"type,omitempty"`
	DeprecatedVersion string              `yaml:"deprecatedVersion,omitempty" json:"deprecatedVersion,omitempty"`
	StabilityLevel    string              `yaml:"stabilityLevel,omitempty" json:"stabilityLevel,omitempty"`
	Labels            []string            `yaml:"labels,omitempty" json:"labels,omitempty"`
	Buckets           []float64           `yaml:"buckets,omitempty" json:"buckets,omitempty"`
	Objectives        map[float64]float64 `yaml:"objectives,omitempty" json:"objectives,omitempty"`
	AgeBuckets        uint32              `yaml:"ageBuckets,omitempty" json:"ageBuckets,omitempty"`
	BufCap            uint32              `yaml:"bufCap,omitempty" json:"bufCap,omitempty"`
	MaxAge            int64               `yaml:"maxAge,omitempty" json:"maxAge,omitempty"`
	ConstLabels       map[string]string   `yaml:"constLabels,omitempty" json:"constLabels,omitempty"`
}

func (m metric) buildFQName() string {
	return metrics.BuildFQName(m.Namespace, m.Subsystem, m.Name)
}
