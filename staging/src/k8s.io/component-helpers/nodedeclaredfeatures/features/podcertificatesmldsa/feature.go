/*
Copyright The Kubernetes Authors.

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

package podcertificatesmldsa

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

var _ types.Feature = &podCertificatesMLDSAFeature{}

const (
	PodCertificateMLDSAFeature = "PodCertificateMLDSA"
)

var Feature = &podCertificatesMLDSAFeature{}

type podCertificatesMLDSAFeature struct{}

func (f *podCertificatesMLDSAFeature) Name() string {
	return PodCertificateMLDSAFeature
}

func (f *podCertificatesMLDSAFeature) Discover(cfg *types.NodeConfiguration) bool {
	return cfg.FeatureGates.Enabled(PodCertificateMLDSAFeature)
}

func (f *podCertificatesMLDSAFeature) Requirements() *types.FeatureRequirements {
	return &types.FeatureRequirements{
		EnabledFeatureGates: []string{PodCertificateMLDSAFeature},
	}
}

func (f *podCertificatesMLDSAFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	return podSpecHasProjectedPodCertificateWithMLDSAKeyType(podInfo.Spec)
}

func (f *podCertificatesMLDSAFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	oldHasMLDSAKeyType := podSpecHasProjectedPodCertificateWithMLDSAKeyType(oldPodInfo.Spec)
	newHasMLDSAKeyType := podSpecHasProjectedPodCertificateWithMLDSAKeyType(newPodInfo.Spec)

	// if the old pod spec does not have any podCertificate projected volumes but
	// the new one does, the update has introduced a dependency on the PodCertificateMLDSA feature gate.
	if !oldHasMLDSAKeyType && newHasMLDSAKeyType {
		return true
	}

	return false
}

func (f *podCertificatesMLDSAFeature) MaxVersion() *version.Version {
	return nil
}

func podSpecHasProjectedPodCertificateWithMLDSAKeyType(podSpec *v1.PodSpec) bool {
	if podSpec == nil {
		return false
	}

	for _, volume := range podSpec.Volumes {
		if volume.Projected != nil {
			for _, source := range volume.Projected.Sources {
				if source.PodCertificate != nil && (source.PodCertificate.KeyType == "MLDSA44" || source.PodCertificate.KeyType == "MLDSA65" || source.PodCertificate.KeyType == "MLDSA87") {
					return true
				}
			}
		}
	}

	return false
}
