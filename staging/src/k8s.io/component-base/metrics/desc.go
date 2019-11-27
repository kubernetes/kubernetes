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
	"strings"
	"sync"

	"github.com/blang/semver"
	"github.com/prometheus/client_golang/prometheus"

	"k8s.io/klog"
)

// Desc is a prometheus.Desc extension.
//
// Use NewDesc to create new Desc instances.
type Desc struct {
	// fqName has been built from Namespace, Subsystem, and Name.
	fqName string
	// help provides some helpful information about this metric.
	help string
	// constLabels is the label names. Their label values are variable.
	constLabels Labels
	// variableLabels contains names of labels for which the metric
	// maintains variable values.
	variableLabels []string

	// promDesc is the descriptor used by every Prometheus Metric.
	promDesc *prometheus.Desc

	// stabilityLevel represents the API guarantees for a given defined metric.
	stabilityLevel StabilityLevel
	// deprecatedVersion represents in which version this metric be deprecated.
	deprecatedVersion string

	isDeprecated        bool
	isHidden            bool
	isCreated           bool
	createLock          sync.RWMutex
	markDeprecationOnce sync.Once
	createOnce          sync.Once
	deprecateOnce       sync.Once
	hideOnce            sync.Once
	annotateOnce        sync.Once
}

// NewDesc extends prometheus.NewDesc with stability support.
//
// The stabilityLevel should be valid stability label, such as "metrics.ALPHA"
// and "metrics.STABLE"(Maybe "metrics.BETA" in future). Default value "metrics.ALPHA"
// will be used in case of empty or invalid stability label.
//
// The deprecatedVersion represents in which version this Metric be deprecated.
// The deprecation policy outlined by the control plane metrics stability KEP.
func NewDesc(fqName string, help string, variableLabels []string, constLabels Labels,
	stabilityLevel StabilityLevel, deprecatedVersion string) *Desc {
	d := &Desc{
		fqName:            fqName,
		help:              help,
		variableLabels:    variableLabels,
		constLabels:       constLabels,
		stabilityLevel:    stabilityLevel,
		deprecatedVersion: deprecatedVersion,
	}
	d.stabilityLevel.setDefaults()

	return d
}

// String formats the Desc as a string.
// The stability metadata maybe annotated in 'HELP' section if called after registry,
// otherwise not.
func (d *Desc) String() string {
	if d.isCreated {
		return d.promDesc.String()
	}

	return prometheus.NewDesc(d.fqName, d.help, d.variableLabels, prometheus.Labels(d.constLabels)).String()
}

// toPrometheusDesc transform self to prometheus.Desc
func (d *Desc) toPrometheusDesc() *prometheus.Desc {
	return d.promDesc
}

// DeprecatedVersion returns a pointer to the Version or nil
func (d *Desc) DeprecatedVersion() *semver.Version {
	return parseSemver(d.deprecatedVersion)

}

func (d *Desc) determineDeprecationStatus(version semver.Version) {
	selfVersion := d.DeprecatedVersion()
	if selfVersion == nil {
		return
	}
	d.markDeprecationOnce.Do(func() {
		if selfVersion.LTE(version) {
			d.isDeprecated = true
		}
		if ShouldShowHidden() {
			klog.Warningf("Hidden metrics(%s) have been manually overridden, showing this very deprecated metric.", d.fqName)
			return
		}
		if shouldHide(&version, selfVersion) {
			// TODO(RainbowMango): Remove this log temporarily. https://github.com/kubernetes/kubernetes/issues/85369
			// klog.Warningf("This metric(%s) has been deprecated for more than one release, hiding.", d.fqName)
			d.isHidden = true
		}
	})
}

// IsHidden returns if metric will be hidden
func (d *Desc) IsHidden() bool {
	return d.isHidden
}

// IsDeprecated returns if metric has been deprecated
func (d *Desc) IsDeprecated() bool {
	return d.isDeprecated
}

func (d *Desc) markDeprecated() {
	d.deprecateOnce.Do(func() {
		d.help = fmt.Sprintf("(Deprecated since %s) %s", d.deprecatedVersion, d.help)
	})
}

func (d *Desc) annotateStabilityLevel() {
	d.annotateOnce.Do(func() {
		d.help = fmt.Sprintf("[%v] %v", d.stabilityLevel, d.help)
	})
}

func (d *Desc) initialize() {
	d.annotateStabilityLevel()
	// this actually creates the underlying prometheus desc.
	d.promDesc = prometheus.NewDesc(d.fqName, d.help, d.variableLabels, prometheus.Labels(d.constLabels))
}

func (d *Desc) initializeDeprecatedDesc() {
	d.markDeprecated()
	d.initialize()
}

// GetRawDesc will returns a new *Desc with original parameters provided to NewDesc().
//
// It will be useful in testing scenario that the same Desc be registered to different registry.
//   1. Desc `D` is registered to registry 'A' in TestA (Note: `D` maybe created)
//   2. Desc `D` is registered to registry 'B' in TestB (Note: since 'D' has been created once, thus will be ignored by registry 'B')
func (d *Desc) GetRawDesc() *Desc {
	// remove stability from help if any
	stabilityStr := fmt.Sprintf("[%v] ", d.stabilityLevel)
	rawHelp := strings.Replace(d.help, stabilityStr, "", -1)

	return NewDesc(d.fqName, rawHelp, d.variableLabels, d.constLabels, d.stabilityLevel, d.deprecatedVersion)
}
