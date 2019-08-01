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

package upgrade

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestEnforceVersionPolicies(t *testing.T) {
	tests := []struct {
		name                        string
		vg                          *fakeVersionGetter
		expectedMandatoryErrs       int
		expectedSkippableErrs       int
		allowExperimental, allowRCs bool
		newK8sVersion               string
	}{
		{
			name: "minor upgrade",
			vg: &fakeVersionGetter{
				clusterVersion: constants.MinimumControlPlaneVersion.WithPatch(3).String(),
				kubeletVersion: constants.MinimumControlPlaneVersion.WithPatch(3).String(),
				kubeadmVersion: constants.MinimumControlPlaneVersion.WithPatch(5).String(),
			},
			newK8sVersion: constants.MinimumControlPlaneVersion.WithPatch(5).String(),
		},
		{
			name: "major upgrade",
			vg: &fakeVersionGetter{
				clusterVersion: constants.MinimumControlPlaneVersion.WithPatch(3).String(),
				kubeletVersion: constants.MinimumControlPlaneVersion.WithPatch(2).String(),
				kubeadmVersion: constants.CurrentKubernetesVersion.WithPatch(1).String(),
			},
			newK8sVersion: constants.CurrentKubernetesVersion.String(),
		},
		{
			name: "downgrade",
			vg: &fakeVersionGetter{
				clusterVersion: constants.MinimumControlPlaneVersion.WithPatch(3).String(),
				kubeletVersion: constants.MinimumKubeletVersion.String(),
				kubeadmVersion: constants.MinimumControlPlaneVersion.WithPatch(3).String(),
			},
			newK8sVersion: constants.MinimumControlPlaneVersion.WithPatch(2).String(),
		},
		{
			name: "same version upgrade",
			vg: &fakeVersionGetter{
				clusterVersion: constants.MinimumControlPlaneVersion.WithPatch(3).String(),
				kubeletVersion: constants.MinimumKubeletVersion.WithPatch(3).String(),
				kubeadmVersion: constants.MinimumControlPlaneVersion.WithPatch(3).String(),
			},
			newK8sVersion: constants.MinimumControlPlaneVersion.WithPatch(3).String(),
		},
		{
			name: "new version must be higher than v1.12.0",
			vg: &fakeVersionGetter{
				clusterVersion: "v1.12.3",
				kubeletVersion: "v1.12.3",
				kubeadmVersion: "v1.12.3",
			},
			newK8sVersion:         "v1.11.10",
			expectedMandatoryErrs: 1, // version must be higher than v1.12.0
			expectedSkippableErrs: 1, // can't upgrade old k8s with newer kubeadm
		},
		{
			name: "upgrading two minor versions in one go is not supported",
			vg: &fakeVersionGetter{
				clusterVersion: "v1.11.3",
				kubeletVersion: "v1.11.3",
				kubeadmVersion: constants.CurrentKubernetesVersion.String(),
			},
			newK8sVersion:         constants.CurrentKubernetesVersion.String(),
			expectedMandatoryErrs: 1, // can't upgrade two minor versions
			expectedSkippableErrs: 1, // kubelet <-> apiserver skew too large
		},
		{
			name: "downgrading two minor versions in one go is not supported",
			vg: &fakeVersionGetter{
				clusterVersion: constants.CurrentKubernetesVersion.WithMinor(constants.CurrentKubernetesVersion.Minor() + 2).String(),
				kubeletVersion: constants.CurrentKubernetesVersion.WithMinor(constants.CurrentKubernetesVersion.Minor() + 2).String(),
				kubeadmVersion: constants.CurrentKubernetesVersion.String(),
			},
			newK8sVersion:         constants.CurrentKubernetesVersion.String(),
			expectedMandatoryErrs: 1, // can't downgrade two minor versions
		},
		{
			name: "kubeadm version must be higher than the new kube version. However, patch version skews may be forced",
			vg: &fakeVersionGetter{
				clusterVersion: constants.MinimumControlPlaneVersion.WithPatch(3).String(),
				kubeletVersion: constants.MinimumKubeletVersion.WithPatch(3).String(),
				kubeadmVersion: constants.MinimumControlPlaneVersion.WithPatch(3).String(),
			},
			newK8sVersion:         constants.MinimumControlPlaneVersion.WithPatch(5).String(),
			expectedSkippableErrs: 1,
		},
		{
			name: "kubeadm version must be higher than the new kube version. Trying to upgrade k8s to a higher minor version than kubeadm itself should never be supported",
			vg: &fakeVersionGetter{
				clusterVersion: constants.MinimumControlPlaneVersion.WithPatch(3).String(),
				kubeletVersion: constants.MinimumKubeletVersion.WithPatch(3).String(),
				kubeadmVersion: constants.MinimumControlPlaneVersion.WithPatch(3).String(),
			},
			newK8sVersion:         constants.CurrentKubernetesVersion.String(),
			expectedMandatoryErrs: 1,
		},
		{
			name: "the maximum skew between the cluster version and the kubelet versions should be one minor version. This may be forced through though.",
			vg: &fakeVersionGetter{
				clusterVersion: constants.MinimumControlPlaneVersion.WithPatch(3).String(),
				kubeletVersion: "v1.12.8",
				kubeadmVersion: constants.CurrentKubernetesVersion.String(),
			},
			newK8sVersion:         constants.CurrentKubernetesVersion.String(),
			expectedSkippableErrs: 1,
		},
		{
			name: "experimental upgrades supported if the flag is set",
			vg: &fakeVersionGetter{
				clusterVersion: constants.MinimumControlPlaneVersion.WithPatch(3).String(),
				kubeletVersion: constants.MinimumKubeletVersion.WithPatch(3).String(),
				kubeadmVersion: constants.CurrentKubernetesVersion.WithPreRelease("beta.1").String(),
			},
			newK8sVersion:     constants.CurrentKubernetesVersion.WithPreRelease("beta.1").String(),
			allowExperimental: true,
		},
		{
			name: "release candidate upgrades supported if the flag is set",
			vg: &fakeVersionGetter{
				clusterVersion: constants.MinimumControlPlaneVersion.WithPatch(3).String(),
				kubeletVersion: constants.MinimumKubeletVersion.WithPatch(3).String(),
				kubeadmVersion: constants.CurrentKubernetesVersion.WithPreRelease("rc.1").String(),
			},
			newK8sVersion: constants.CurrentKubernetesVersion.WithPreRelease("rc.1").String(),
			allowRCs:      true,
		},
		{
			name: "release candidate upgrades supported if the flag is set",
			vg: &fakeVersionGetter{
				clusterVersion: constants.MinimumControlPlaneVersion.WithPatch(3).String(),
				kubeletVersion: constants.MinimumKubeletVersion.WithPatch(3).String(),
				kubeadmVersion: constants.CurrentKubernetesVersion.WithPreRelease("rc.1").String(),
			},
			newK8sVersion:     constants.CurrentKubernetesVersion.WithPreRelease("rc.1").String(),
			allowExperimental: true,
		},
		{
			name: "the user should not be able to upgrade to an experimental version if they haven't opted into that",
			vg: &fakeVersionGetter{
				clusterVersion: constants.MinimumControlPlaneVersion.WithPatch(3).String(),
				kubeletVersion: constants.MinimumKubeletVersion.WithPatch(3).String(),
				kubeadmVersion: constants.CurrentKubernetesVersion.WithPreRelease("beta.1").String(),
			},
			newK8sVersion:         constants.CurrentKubernetesVersion.WithPreRelease("beta.1").String(),
			allowRCs:              true,
			expectedSkippableErrs: 1,
		},
		{
			name: "the user should not be able to upgrade to an release candidate version if they haven't opted into that",
			vg: &fakeVersionGetter{
				clusterVersion: constants.MinimumControlPlaneVersion.WithPatch(3).String(),
				kubeletVersion: constants.MinimumKubeletVersion.WithPatch(3).String(),
				kubeadmVersion: constants.CurrentKubernetesVersion.WithPreRelease("rc.1").String(),
			},
			newK8sVersion:         constants.CurrentKubernetesVersion.WithPreRelease("rc.1").String(),
			expectedSkippableErrs: 1,
		},
		{
			name: "the user can't use a newer minor version of kubeadm to upgrade an older version of kubeadm",
			vg: &fakeVersionGetter{
				clusterVersion: constants.MinimumControlPlaneVersion.WithPatch(3).String(),
				kubeletVersion: constants.MinimumKubeletVersion.WithPatch(3).String(),
				kubeadmVersion: constants.CurrentKubernetesVersion.String(),
			},
			newK8sVersion:         constants.MinimumControlPlaneVersion.WithPatch(6).String(),
			expectedSkippableErrs: 1, // can't upgrade old k8s with newer kubeadm
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {

			newK8sVer, err := version.ParseSemantic(rt.newK8sVersion)
			if err != nil {
				t.Fatalf("couldn't parse version %s: %v", rt.newK8sVersion, err)
			}

			actualSkewErrs := EnforceVersionPolicies(rt.vg, rt.newK8sVersion, newK8sVer, rt.allowExperimental, rt.allowRCs)
			if actualSkewErrs == nil {
				// No errors were seen. Report unit test failure if we expected to see errors
				if rt.expectedMandatoryErrs+rt.expectedSkippableErrs > 0 {
					t.Errorf("failed TestEnforceVersionPolicies\n\texpected errors but got none")
				}
				// Otherwise, just move on with the next test
				return
			}

			if len(actualSkewErrs.Skippable) != rt.expectedSkippableErrs {
				t.Errorf("failed TestEnforceVersionPolicies\n\texpected skippable errors: %d\n\tgot skippable errors: %d\n%#v\n%#v", rt.expectedSkippableErrs, len(actualSkewErrs.Skippable), *rt.vg, actualSkewErrs)
			}
			if len(actualSkewErrs.Mandatory) != rt.expectedMandatoryErrs {
				t.Errorf("failed TestEnforceVersionPolicies\n\texpected mandatory errors: %d\n\tgot mandatory errors: %d\n%#v\n%#v", rt.expectedMandatoryErrs, len(actualSkewErrs.Mandatory), *rt.vg, actualSkewErrs)
			}
		})
	}
}
