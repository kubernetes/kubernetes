/*
Copyright 2018 The Kubernetes Authors.

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

package framework

import (
	"bytes"
	"fmt"
	"sync"

	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
)

// FlakeReport is a struct for managing the flake report.
type FlakeReport struct {
	lock       sync.RWMutex
	Flakes     []string `json:"flakes"`
	FlakeCount int      `json:"flakeCount"`
}

// NewFlakeReport returns a new flake report.
func NewFlakeReport() *FlakeReport {
	return &FlakeReport{
		Flakes: []string{},
	}
}

func buildDescription(optionalDescription ...interface{}) string {
	switch len(optionalDescription) {
	case 0:
		return ""
	default:
		return fmt.Sprintf(optionalDescription[0].(string), optionalDescription[1:]...)
	}
}

// RecordFlakeIfError records the error (if non-nil) as a flake along with an optional description.
// This can be used as a replacement of framework.ExpectNoError() for non-critical errors that can
// be considered as 'flakes' to avoid causing failures in tests.
func (f *FlakeReport) RecordFlakeIfError(err error, optionalDescription ...interface{}) {
	if err == nil {
		return
	}
	msg := fmt.Sprintf("Unexpected error occurred: %v", err)
	desc := buildDescription(optionalDescription)
	if desc != "" {
		msg = fmt.Sprintf("%v (Description: %v)", msg, desc)
	}
	e2elog.Logf(msg)
	f.lock.Lock()
	defer f.lock.Unlock()
	f.Flakes = append(f.Flakes, msg)
	f.FlakeCount++
}

// GetFlakeCount returns the flake count.
func (f *FlakeReport) GetFlakeCount() int {
	f.lock.RLock()
	defer f.lock.RUnlock()
	return f.FlakeCount
}

// PrintHumanReadable returns string of flake report.
func (f *FlakeReport) PrintHumanReadable() string {
	f.lock.RLock()
	defer f.lock.RUnlock()
	buf := bytes.Buffer{}
	buf.WriteString(fmt.Sprintf("FlakeCount: %v\n", f.FlakeCount))
	buf.WriteString("Flakes:\n")
	for _, flake := range f.Flakes {
		buf.WriteString(fmt.Sprintf("%v\n", flake))
	}
	return buf.String()
}

// PrintJSON returns the summary of frake report with JSON format.
func (f *FlakeReport) PrintJSON() string {
	f.lock.RLock()
	defer f.lock.RUnlock()
	return e2emetrics.PrettyPrintJSON(f)
}

// SummaryKind returns the summary of flake report.
func (f *FlakeReport) SummaryKind() string {
	return "FlakeReport"
}
