/*
Copyright 2024 The Kubernetes Authors.

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

package statusz

import (
	"bytes"
	"fmt"
	"html/template"
	"net/http"
	"strings"
	"time"

	"k8s.io/apiserver/pkg/endpoints/metrics"
	"k8s.io/klog/v2"
)

var (
	funcMap = template.FuncMap{
		"ToLower": strings.ToLower,
	}
	tmpl *template.Template
	reg  statuszRegistry = registry{}
)

const (
	statuszTemplate = `
----------------------------
title: Kubernetes Statusz
content_type: reference
auto_generated: true
description: Details of the status data that Kubernetes components report.
----------------------------

## Started: {{.StartTime}}
## Up: {{.Uptime}}

## Build Info
--------------
### Go version: {{.GoVersion}}
### Binary version: {{.BinaryVersion}}
### Emulation version: {{.EmulationVersion}}
### Minimum Compatibility version: {{.MinCompatibilityVersion}}

## List of useful endpoints
--------------
{{- range $name, $link := .UsefulLinks}}
{{$name}}:{{$link -}}
{{- end }}
`
)

type templateData struct {
	StartTime               string
	Uptime                  string
	GoVersion               string
	BinaryVersion           string
	EmulationVersion        string
	MinCompatibilityVersion string
	UsefulLinks             map[string]string
}

type mux interface {
	Handle(path string, handler http.Handler)
}

func init() {
	// Parse the template once during initialization
	var err error
	t := template.New("t").Funcs(funcMap)
	tmpl, err = t.Parse(statuszTemplate)
	if err != nil {
		klog.Errorf("error while parsing gotemplate: %v", err)
	}
}

func Install(m mux) {
	m.Handle("/statusz",
		metrics.InstrumentHandlerFunc("GET",
			/* group = */ "",
			/* version = */ "",
			/* resource = */ "",
			/* subresource = */ "/statusz",
			/* scope = */ "",
			/* component = */ "",
			/* deprecated */ false,
			/* removedRelease */ "",
			handleStatusz(tmpl)))
}

func handleStatusz(tmpl *template.Template) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		statuszData, err := populateStatuszData(tmpl)
		if err != nil {
			klog.Errorf("error while populating statusz data: %v", err)
			http.Error(w, "error while populating statusz data", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		fmt.Fprint(w, statuszData)
	}
}

func populateStatuszData(tmpl *template.Template) (string, error) {
	if tmpl == nil {
		return "", fmt.Errorf("received nil template")
	}

	data := templateData{
		StartTime:               reg.processStartTime().Format(time.UnixDate),
		Uptime:                  uptime(reg.processStartTime()),
		GoVersion:               reg.goVersion(),
		BinaryVersion:           reg.binaryVersion().String(),
		EmulationVersion:        reg.emulationVersion().String(),
		MinCompatibilityVersion: reg.minCompatibilityVersion().String(),
		UsefulLinks:             reg.usefulLinks(),
	}

	var tpl bytes.Buffer
	err := tmpl.Execute(&tpl, data)
	if err != nil {
		return "", fmt.Errorf("error executing statusz template: %v", err)
	}

	return tpl.String(), nil
}

func uptime(t time.Time) string {
	upSince := int64(time.Since(t).Seconds())
	return fmt.Sprintf("%d hr %02d min %02d sec",
		upSince/3600, (upSince/60)%60, upSince%60)
}
