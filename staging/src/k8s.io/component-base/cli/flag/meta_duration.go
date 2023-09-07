/*
Copyright 2023 The Kubernetes Authors.

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

package flag

import (
	goflag "flag"
	"fmt"
	"time"

	"github.com/spf13/pflag"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// MetaDuration can set a metav1.Duration from the command line
type MetaDuration struct {
	m *metav1.Duration
}

var _ goflag.Value = &MetaDuration{}
var _ pflag.Value = &MetaDuration{}

// NewMetaDuration takes a pointer to a metav1.Duration and returns the
// MetaDuration flag parsing shim for that variable
func NewMetaDuration(m *metav1.Duration) MetaDuration {
	return MetaDuration{m: m}
}

// Set implements github.com/spf13/pflag.String
func (m *MetaDuration) String() string {
	if m == nil || m.m == nil {
		return ""
	}
	return m.m.String()
}

// Set implements github.com/spf13/pflag.Value
func (m *MetaDuration) Set(val string) error {
	if m == nil {
		return fmt.Errorf("no target (nil pointer to metav1.Duration")
	}
	duration, err := time.ParseDuration(val)
	if err != nil {
		return err
	}
	m.m = &metav1.Duration{Duration: duration}
	return nil
}

// Type implements github.com/spf13/pflag.Value
func (m *MetaDuration) Type() string {
	return "metav1.Duration"
}
