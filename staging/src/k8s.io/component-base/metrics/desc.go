/*
Copyright 2019 The Kubernetes Authors.

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

	"github.com/prometheus/client_golang/prometheus"
)

// Desc is a prometheus.Desc extension.
//
// Use NewDesc to create new Desc instances.
type Desc prometheus.Desc

func (d *Desc) toPromDesc() *prometheus.Desc {
	return (*prometheus.Desc)(d)
}

// NewDesc allocates and initializes a new Desc. Errors are recorded in the Desc
// and will be reported on registration time. variableLabels and constLabels can
// be nil if no such labels should be set. fqName must not be empty.
//
// variableLabels only contain the label names. Their label values are variable
// and therefore not part of the Desc. (They are managed within the Metric.)
//
// For constLabels, the label values are constant. Therefore, they are fully
// specified in the Desc. See the Collector example for a usage pattern.
func NewDesc(fqName string, help string, variableLabels []string, constLabels Labels) *Desc {
	// TODO(RainbowMango): Here just force all metrics defined through a Desc as an ALPHA metric.
	// This should be changed after we have a better solution for this use case.
	helpWithStability := fmt.Sprintf("[ALPHA] %v", help)

	return (*Desc)(prometheus.NewDesc(fqName, helpWithStability, variableLabels, prometheus.Labels(constLabels)))
}
