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
	"sort"
	"strings"
	"sync"
	"text/template"

	"github.com/spf13/pflag"

	"k8s.io/apiserver/pkg/endpoints/metrics"
	"k8s.io/klog/v2"

	cliflag "k8s.io/component-base/cli/flag"
)

const (
	flagzTemplate = `
----------------------------
title: {{.ComponentName}} flagz
content_type: reference
auto_generated: true
description: flags enabled in {{.ComponentName}}
----------------------------

`
)

var (
	funcMap = template.FuncMap{
		"ToLower": strings.ToLower,
	}
	tmpl          *template.Template
	flagzRegistry = &Registry{}
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

func init() {
	var err error
	t := template.New("t").Funcs(funcMap)
	tmpl, err = t.Parse(flagzTemplate)
	if err != nil {
		klog.Errorf("error while parsing gotemplate: %v", err)
	}
}

func (f Flagz) Install(m mux, componentName string, flagSets []*cliflag.NamedFlagSets) {
	flagzRegistry.installHandler(m, componentName, flagSets)
}

func (reg *Registry) installHandler(m mux, componentName string, flagSets []*cliflag.NamedFlagSets) {
	m.Handle("/flagz",
		metrics.InstrumentHandlerFunc("GET",
			/* group = */ "",
			/* version = */ "",
			/* resource = */ "",
			/* subresource = */ "/flagz",
			/* scope = */ "",
			/* component = */ "",
			/* deprecated */ false,
			/* removedRelease */ "",
			reg.handleFlags(componentName, tmpl, flagSets)))
}

func (reg *Registry) handleFlags(componentName string, tmpl *template.Template, flagSets []*cliflag.NamedFlagSets) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		reg.once.Do(func() {
			header, err := populateHeader(componentName, tmpl)
			if err != nil {
				klog.Errorf("error while populating flagz header: %v", err)
				http.Error(w, "error while populating flagz header", http.StatusInternalServerError)
				return
			}

			fmt.Fprint(&reg.response, header)
			flags := sortedFlags(flagSets)

			for _, flag := range flags {
				if set, ok := flag.Annotations["classified"]; !ok || len(set) == 0 {
					fmt.Fprint(&reg.response, flag.Name, "=", flag.Value, "\n")
				} else {
					fmt.Fprint(&reg.response, flag.Name, "=", "CLASSIFIED", "\n")
				}
			}
		})
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		var cachedResponse bytes.Buffer
		_, err := cachedResponse.Write(reg.response.Bytes())
		if err != nil {
			klog.Errorf("error writing to cachedResponse: %v", err)
			http.Error(w, "error writing to cachedResponse", http.StatusInternalServerError)
			return
		}
		_, err = cachedResponse.WriteTo(w)
		if err != nil {
			klog.Errorf("error writing response: %v", err)
			http.Error(w, "error writing response", http.StatusInternalServerError)
		}
	}
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

func sortedFlags(flagSets []*cliflag.NamedFlagSets) []*pflag.Flag {
	var flags []*pflag.Flag
	for _, flagset := range flagSets {
		for _, fs := range flagset.FlagSets {
			fs.VisitAll(func(flag *pflag.Flag) {
				if flag.Value != nil && flag.Value.String() != "" && flag.Value.String() != "[]" {
					flags = append(flags, flag)
				}
			})
		}
	}

	sort.Slice(flags, func(i, j int) bool {
		return flags[i].Name < flags[j].Name
	})

	return flags
}
