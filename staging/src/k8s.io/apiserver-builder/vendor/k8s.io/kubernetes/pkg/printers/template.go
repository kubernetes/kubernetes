/*
Copyright 2017 The Kubernetes Authors.

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

package printers

import (
	"encoding/json"
	"fmt"
	"io"
	"text/template"

	"k8s.io/apimachinery/pkg/runtime"
)

// TemplatePrinter is an implementation of ResourcePrinter which formats data with a Go Template.
type TemplatePrinter struct {
	rawTemplate string
	template    *template.Template
}

func NewTemplatePrinter(tmpl []byte) (*TemplatePrinter, error) {
	t, err := template.New("output").
		Funcs(template.FuncMap{"exists": exists}).
		Parse(string(tmpl))
	if err != nil {
		return nil, err
	}
	return &TemplatePrinter{
		rawTemplate: string(tmpl),
		template:    t,
	}, nil
}

// AllowMissingKeys tells the template engine if missing keys are allowed.
func (p *TemplatePrinter) AllowMissingKeys(allow bool) {
	if allow {
		p.template.Option("missingkey=default")
	} else {
		p.template.Option("missingkey=error")
	}
}

func (p *TemplatePrinter) AfterPrint(w io.Writer, res string) error {
	return nil
}

// PrintObj formats the obj with the Go Template.
func (p *TemplatePrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	var data []byte
	var err error
	data, err = json.Marshal(obj)
	if err != nil {
		return err
	}

	out := map[string]interface{}{}
	if err := json.Unmarshal(data, &out); err != nil {
		return err
	}
	if err = p.safeExecute(w, out); err != nil {
		// It is way easier to debug this stuff when it shows up in
		// stdout instead of just stdin. So in addition to returning
		// a nice error, also print useful stuff with the writer.
		fmt.Fprintf(w, "Error executing template: %v. Printing more information for debugging the template:\n", err)
		fmt.Fprintf(w, "\ttemplate was:\n\t\t%v\n", p.rawTemplate)
		fmt.Fprintf(w, "\traw data was:\n\t\t%v\n", string(data))
		fmt.Fprintf(w, "\tobject given to template engine was:\n\t\t%+v\n\n", out)
		return fmt.Errorf("error executing template %q: %v", p.rawTemplate, err)
	}
	return nil
}

// TODO: implement HandledResources()
func (p *TemplatePrinter) HandledResources() []string {
	return []string{}
}

// safeExecute tries to execute the template, but catches panics and returns an error
// should the template engine panic.
func (p *TemplatePrinter) safeExecute(w io.Writer, obj interface{}) error {
	var panicErr error
	// Sorry for the double anonymous function. There's probably a clever way
	// to do this that has the defer'd func setting the value to be returned, but
	// that would be even less obvious.
	retErr := func() error {
		defer func() {
			if x := recover(); x != nil {
				panicErr = fmt.Errorf("caught panic: %+v", x)
			}
		}()
		return p.template.Execute(w, obj)
	}()
	if panicErr != nil {
		return panicErr
	}
	return retErr
}
