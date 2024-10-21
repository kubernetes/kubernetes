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

package flagz

import (
	"bytes"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"text/template"

	"k8s.io/klog/v2"
)

const (
	flagzTemplate = `
----------------------------
title: {{.ComponentName}} flagz
description: flags enabled in {{.ComponentName}}
warning: This endpoint is not meant to be machine parseable and is for debugging purposes only.
----------------------------

`
)

var (
	funcMap = template.FuncMap{
		"ToLower": strings.ToLower,
	}
	tmpl           *template.Template
	flagzRegistry  = &Registry{}
	cachedResponse []byte
)

type Registry struct {
	response bytes.Buffer
	once     sync.Once
}

type Flagz struct{}

type mux interface {
	Handle(path string, handler http.Handler)
}

type templateData struct {
	ComponentName string
}

func (f Flagz) Install(m mux, componentName string, flagSets []Flag) {
	flagzRegistry.installHandler(m, componentName, flagSets)
}

func (reg *Registry) installHandler(m mux, componentName string, flags []Flag) {
	err := initializeTemplates()
	if err != nil {
		klog.Errorf("error while parsing gotemplate: %v", err)
		return
	}
	m.Handle("/flagz", reg.handleFlags(componentName, tmpl, flags))
}

func (reg *Registry) handleFlags(componentName string, tmpl *template.Template, flags []Flag) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		reg.once.Do(func() {
			header, err := populateHeader(componentName, tmpl)
			if err != nil {
				klog.Errorf("error while populating flagz header: %v", err)
				http.Error(w, "error while populating flagz header", http.StatusInternalServerError)
				return
			}

			fmt.Fprint(&reg.response, header)
			printFlags(&reg.response, flags)
			cachedResponse = reg.response.Bytes()
		})
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, err := w.Write(cachedResponse)
		if err != nil {
			klog.Errorf("error writing response: %v", err)
			http.Error(w, "error writing response", http.StatusInternalServerError)
		}
	}
}

func initializeTemplates() error {
	var err error
	t := template.New("t").Funcs(funcMap)
	tmpl, err = t.Parse(flagzTemplate)
	if err != nil {
		return err
	}

	return nil
}

func populateHeader(componentName string, tmpl *template.Template) (string, error) {
	if tmpl == nil {
		return "", fmt.Errorf("received nil template")
	}

	data := templateData{
		ComponentName: componentName,
	}

	var tpl bytes.Buffer
	err := tmpl.Execute(&tpl, data)
	if err != nil {
		return "", fmt.Errorf("error populating flagz header: %w", err)
	}

	return tpl.String(), nil
}
