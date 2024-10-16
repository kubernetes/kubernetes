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

<<<<<<< HEAD
<<<<<<< HEAD
=======
	"k8s.io/apiserver/pkg/endpoints/metrics"
>>>>>>> fd6090c7231 (address comments)
=======
>>>>>>> 5931d5ad74d (address comments)
	"k8s.io/klog/v2"

	cliflag "k8s.io/component-base/cli/flag"
)

const (
	flagzTemplate = `
----------------------------
title: {{.ComponentName}} flagz
<<<<<<< HEAD
<<<<<<< HEAD
description: flags enabled in {{.ComponentName}}
warning: This endpoint is not meant to be machine parseable and is for debugging purposes only.
=======
content_type: reference
auto_generated: true
description: flags enabled in {{.ComponentName}}
>>>>>>> fd6090c7231 (address comments)
=======
description: flags enabled in {{.ComponentName}}
warning: This endpoint is not meant to be machine parseable and is for debugging purposes only.
>>>>>>> 5931d5ad74d (address comments)
----------------------------

`
)

var (
	funcMap = template.FuncMap{
		"ToLower": strings.ToLower,
	}
<<<<<<< HEAD
<<<<<<< HEAD
	tmpl           *template.Template
	flagzRegistry  = &Registry{}
	cachedResponse []byte
=======
	tmpl          *template.Template
	flagzRegistry = &Registry{}
>>>>>>> fd6090c7231 (address comments)
=======
	tmpl           *template.Template
	flagzRegistry  = &Registry{}
	cachedResponse []byte
>>>>>>> 5931d5ad74d (address comments)
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

<<<<<<< HEAD
<<<<<<< HEAD
=======
func init() {
	var err error
	t := template.New("t").Funcs(funcMap)
	tmpl, err = t.Parse(flagzTemplate)
	if err != nil {
		klog.Errorf("error while parsing gotemplate: %v", err)
	}
}

>>>>>>> fd6090c7231 (address comments)
=======
>>>>>>> 5931d5ad74d (address comments)
func (f Flagz) Install(m mux, componentName string, flagSets []*cliflag.NamedFlagSets) {
	flagzRegistry.installHandler(m, componentName, flagSets)
}

func (reg *Registry) installHandler(m mux, componentName string, flagSets []*cliflag.NamedFlagSets) {
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 5931d5ad74d (address comments)
	err := initializeTemplates()
	if err != nil {
		klog.Errorf("error while parsing gotemplate: %v", err)
		return
	}
	m.Handle("/flagz", reg.handleFlags(componentName, tmpl, flagSets))
<<<<<<< HEAD
=======
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
>>>>>>> fd6090c7231 (address comments)
=======
>>>>>>> 5931d5ad74d (address comments)
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
<<<<<<< HEAD
<<<<<<< HEAD
			reg.printFlags(sortedFlags(flagSets))
			cachedResponse = reg.response.Bytes()
		})
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, err := w.Write(cachedResponse)
=======
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
>>>>>>> fd6090c7231 (address comments)
=======
			reg.printFlags(sortedFlags(flagSets))
			cachedResponse = reg.response.Bytes()
		})
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, err := w.Write(cachedResponse)
>>>>>>> 5931d5ad74d (address comments)
		if err != nil {
			klog.Errorf("error writing response: %v", err)
			http.Error(w, "error writing response", http.StatusInternalServerError)
		}
	}
}

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 5931d5ad74d (address comments)
func initializeTemplates() error {
	var err error
	t := template.New("t").Funcs(funcMap)
	tmpl, err = t.Parse(flagzTemplate)
	if err != nil {
		return err
	}

	return nil
}

<<<<<<< HEAD
=======
>>>>>>> fd6090c7231 (address comments)
=======
>>>>>>> 5931d5ad74d (address comments)
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
<<<<<<< HEAD
<<<<<<< HEAD
				if flag.Value != nil {
=======
				if flag.Value != nil && flag.Value.String() != "" && flag.Value.String() != "[]" {
>>>>>>> fd6090c7231 (address comments)
=======
				if flag.Value != nil {
>>>>>>> 5931d5ad74d (address comments)
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
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 5931d5ad74d (address comments)

func (reg *Registry) printFlags(flags []*pflag.Flag) {
	for _, flag := range flags {
		if set, ok := flag.Annotations["classified"]; !ok || len(set) == 0 {
			fmt.Fprint(&reg.response, flag.Name, "=", flag.Value, "\n")
		} else {
			fmt.Fprint(&reg.response, flag.Name, "=", "CLASSIFIED", "\n")
		}
	}
}
<<<<<<< HEAD
=======
>>>>>>> fd6090c7231 (address comments)
=======
>>>>>>> 5931d5ad74d (address comments)
