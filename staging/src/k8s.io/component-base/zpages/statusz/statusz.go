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

	"k8s.io/klog/v2"
)

var (
	funcMap = template.FuncMap{
		"ToLower": strings.ToLower,
	}
	headerTmpl *template.Template
	dataTmpl   *template.Template
	reg        statuszRegistry = registry{}
)

const (
	headerTemplate = `
------------------------------------------------------------------------
title: {{.ComponentName}} statusz
description: details of the status data that {{.ComponentName}} reports.
warning: This endpoint is not meant to be machine parseable and is for debugging purposes only.
------------------------------------------------------------------------
`

	dataTemplate = `
Started: {{.StartTime}}
Up: {{.Uptime}}
Go version: {{.GoVersion}}
Binary version: {{.BinaryVersion}}
Emulation version: {{.EmulationVersion}}
Minimum Compatibility version: {{.MinCompatibilityVersion}}

List of useful endpoints
--------------
{{- range $name, $link := .UsefulLinks}}
{{$name}}:{{$link -}}
{{- end }}
`
)

type headerFields struct {
	ComponentName string
}

type contentFields struct {
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

func Install(m mux, componentName string) {
	err := initializeTemplates()
	if err != nil {
		klog.Errorf("error while parsing gotemplates: %v", err)
		return
	}
	m.Handle("/statusz", handleStatusz(componentName, headerTmpl, dataTmpl))
}

func initializeTemplates() error {
	var err error
	h := template.New("header").Funcs(funcMap)
	headerTmpl, err = h.Parse(headerTemplate)
	if err != nil {
		return err
	}

	d := template.New("data").Funcs(funcMap)
	dataTmpl, err = d.Parse(dataTemplate)
	if err != nil {
		return err
	}

	return nil
}

func handleStatusz(componentName string, headerTmpl, dataTmpl *template.Template) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		header, err := populateHeader(componentName, headerTmpl)
		if err != nil {
			klog.Errorf("error while populating statusz header: %v", err)
			http.Error(w, "error while populating statusz header", http.StatusInternalServerError)
			return
		}

		data, err := populateStatuszData(dataTmpl)
		if err != nil {
			klog.Errorf("error while populating statusz data: %v", err)
			http.Error(w, "error while populating statusz data", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		fmt.Fprint(w, header)
		fmt.Fprint(w, data)
	}
}

func populateHeader(componentName string, tmpl *template.Template) (string, error) {
	if tmpl == nil {
		return "", fmt.Errorf("received nil template")
	}

	data := headerFields{
		ComponentName: componentName,
	}

	var tpl bytes.Buffer
	err := tmpl.Execute(&tpl, data)
	if err != nil {
		return "", err
	}

	return tpl.String(), nil
}

func populateStatuszData(tmpl *template.Template) (string, error) {
	if tmpl == nil {
		return "", fmt.Errorf("received nil template")
	}

	data := contentFields{
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
		return "", fmt.Errorf("error executing statusz template: %w", err)
	}

	return tpl.String(), nil
}

func uptime(t time.Time) string {
	upSince := int64(time.Since(t).Seconds())
	return fmt.Sprintf("%d hr %02d min %02d sec",
		upSince/3600, (upSince/60)%60, upSince%60)
}
