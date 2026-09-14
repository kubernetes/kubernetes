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

type mustAttemptImagePullResult string

const (
	checkResultCredentialPolicyAllowed mustAttemptImagePullResult = "credentialPolicyAllowed"
	checkResultCredentialRecordFound   mustAttemptImagePullResult = "credentialRecordFound"
	checkResultMustAuthenticate        mustAttemptImagePullResult = "mustAuthenticate"
	checkResultError                   mustAttemptImagePullResult = "error"
)

var (
	fsPullIntentsSize = metrics.NewGauge(
		&metrics.GaugeOpts{
			Subsystem:      kubeletmetrics.KubeletSubsystem,
			Name:           "imagemanager_ondisk_pullintents",
			Help:           "Number of ImagePullIntents stored on disk.",
			StabilityLevel: metrics.ALPHA,
		},
	)
	fsPulledRecordsSize = metrics.NewGauge(
		&metrics.GaugeOpts{
			Subsystem:      kubeletmetrics.KubeletSubsystem,
			Name:           "imagemanager_ondisk_pulledrecords",
			Help:           "Number of ImagePulledRecords stored on disk.",
			StabilityLevel: metrics.ALPHA,
		},
	)
	inMemIntentsPercent = metrics.NewGauge(
		&metrics.GaugeOpts{
			Subsystem:      kubeletmetrics.KubeletSubsystem,
			Name:           "imagemanager_inmemory_pullintents_usage_percent",
			Help:           "The ImagePullIntents in-memory cache usage in percent.",
			StabilityLevel: metrics.ALPHA,
		},
	)
	inMemPulledRecordsPercent = metrics.NewGauge(
		&metrics.GaugeOpts{
			Subsystem:      kubeletmetrics.KubeletSubsystem,
			Name:           "imagemanager_inmemory_pulledrecords_usage_percent",
			Help:           "The ImagePulledRecords in-memory cache usage in percent.",
			StabilityLevel: metrics.ALPHA,
		},
	)
	mustPullChecksTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      kubeletmetrics.KubeletSubsystem,
			Name:           "imagemanager_image_mustpull_checks_total",
			Help:           "Counter for how many times kubelet checked whether credentials need to be re-verified to access an image",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"result"},
	)
)

func init() {
	legacyregistry.MustRegister(fsPullIntentsSize)
	legacyregistry.MustRegister(fsPulledRecordsSize)
	legacyregistry.MustRegister(inMemIntentsPercent)
	legacyregistry.MustRegister(inMemPulledRecordsPercent)
	legacyregistry.MustRegister(mustPullChecksTotal)
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

func (m *meteringRecordsAccessor) WriteImagePullIntent(logger klog.Logger, image string) error {
	if err := m.sizeExposedPullRecordsAccessor.WriteImagePullIntent(logger, image); err != nil {
		return err
	}
	m.recordIntentsSize(logger)
	return nil
}

func (m *meteringRecordsAccessor) DeleteImagePullIntent(logger klog.Logger, image string) error {
	if err := m.sizeExposedPullRecordsAccessor.DeleteImagePullIntent(logger, image); err != nil {
		return err
	}
	m.recordIntentsSize(logger)
	return nil
}

func (m *meteringRecordsAccessor) WriteImagePulledRecord(logger klog.Logger, record *kubeletconfiginternal.ImagePulledRecord) error {
	if err := m.sizeExposedPullRecordsAccessor.WriteImagePulledRecord(logger, record); err != nil {
		return err
	}
	m.recordPulledRecordsSize(logger)
	return nil
}

func (m *meteringRecordsAccessor) DeleteImagePulledRecord(logger klog.Logger, imageRef string) error {
	if err := m.sizeExposedPullRecordsAccessor.DeleteImagePulledRecord(logger, imageRef); err != nil {
		return err
	}
	m.recordPulledRecordsSize(logger)
	return nil
}

func (m *meteringRecordsAccessor) recordIntentsSize(logger klog.Logger) {
	intentsSize, err := m.sizeExposedPullRecordsAccessor.intentsSize()
	if err != nil {
		logger.V(4).Info("failed to read number of ImagePullIntents, can't update metric", "metricName", m.intentsSize.Name, "error", err)
		return
	}
	m.intentsSize.Set(float64(intentsSize))
}

func (m *meteringRecordsAccessor) recordPulledRecordsSize(logger klog.Logger) {
	pulledRecordsSize, err := m.sizeExposedPullRecordsAccessor.pulledRecordsSize()
	if err != nil {
		logger.V(4).Info("failed to read number of ImagePulledRecords, can't update metric", "metricName", m.pulledRecordsSize.Name, "error", err)
		return
	}
	m.pulledRecordsSize.Set(float64(pulledRecordsSize))
}

func recordMustAttemptImagePullResult(result mustAttemptImagePullResult) {
	mustPullChecksTotal.WithLabelValues(string(result)).Inc()
}
