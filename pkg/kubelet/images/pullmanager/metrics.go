/*
Copyright 2025 The Kubernetes Authors.

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

package pullmanager

import (
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/klog/v2"
	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
)

var (
	fsPullIntentsSize = metrics.NewGauge(
		&metrics.GaugeOpts{
			Subsystem:      kubeletmetrics.KubeletSubsystem + "_imagemanager",
			Name:           "ondisk_pullintents",
			Help:           "Number of ImagePullIntents stored on disk.",
			StabilityLevel: metrics.ALPHA,
		},
	)
	fsPulledRecordsSize = metrics.NewGauge(
		&metrics.GaugeOpts{
			Subsystem:      kubeletmetrics.KubeletSubsystem + "_imagemanager",
			Name:           "ondisk_pulledrecords",
			Help:           "Number of ImagePulledRecords stored on disk.",
			StabilityLevel: metrics.ALPHA,
		},
	)
)

func init() {
	legacyregistry.MustRegister(fsPullIntentsSize)
	legacyregistry.MustRegister(fsPulledRecordsSize)
}

// meteringRecordsAccessor wraps a PullRecordsAccessor that's capable of reporting its size
// and updates its record metrics on write operations.
type meteringRecordsAccessor struct {
	sizeExposedPullRecordsAccessor
	intentsSize       *metrics.Gauge
	pulledRecordsSize *metrics.Gauge
}

type sizeExposedPullRecordsAccessor interface {
	PullRecordsAccessor
	intentsSize() (uint, error)
	pulledRecordsSize() (uint, error)
}

func NewMeteringRecordsAccessor(pullRecordsAccessor sizeExposedPullRecordsAccessor, intentsSize, pulledRecordsSize *metrics.Gauge) *meteringRecordsAccessor {
	return &meteringRecordsAccessor{
		sizeExposedPullRecordsAccessor: pullRecordsAccessor,
		intentsSize:                    intentsSize,
		pulledRecordsSize:              pulledRecordsSize,
	}
}

func (m *meteringRecordsAccessor) WriteImagePullIntent(image string) error {
	if err := m.sizeExposedPullRecordsAccessor.WriteImagePullIntent(image); err != nil {
		return err
	}
	m.recordIntentsSize()
	return nil
}

func (m *meteringRecordsAccessor) DeleteImagePullIntent(image string) error {
	if err := m.sizeExposedPullRecordsAccessor.DeleteImagePullIntent(image); err != nil {
		return err
	}
	m.recordIntentsSize()
	return nil
}

func (m *meteringRecordsAccessor) WriteImagePulledRecord(record *kubeletconfiginternal.ImagePulledRecord) error {
	if err := m.sizeExposedPullRecordsAccessor.WriteImagePulledRecord(record); err != nil {
		return err
	}
	m.recordPulledRecordsSize()
	return nil
}

func (m *meteringRecordsAccessor) DeleteImagePulledRecord(imageRef string) error {
	if err := m.sizeExposedPullRecordsAccessor.DeleteImagePulledRecord(imageRef); err != nil {
		return err
	}
	m.recordPulledRecordsSize()
	return nil
}

func (m *meteringRecordsAccessor) recordIntentsSize() {
	intentsSize, err := m.sizeExposedPullRecordsAccessor.intentsSize()
	if err != nil {
		klog.V(4).ErrorS(err, "failed to read number of ImagePullIntents, can't update metric", "metricName", m.intentsSize.Name)
		return
	}
	m.intentsSize.Set(float64(intentsSize))
}

func (m *meteringRecordsAccessor) recordPulledRecordsSize() {
	pulledRecordsSize, err := m.sizeExposedPullRecordsAccessor.pulledRecordsSize()
	if err != nil {
		klog.V(4).ErrorS(err, "failed to read number of ImagePulledRecords, can't update metric", "metricName", m.pulledRecordsSize.Name)
		return
	}
	m.pulledRecordsSize.Set(float64(pulledRecordsSize))
}
