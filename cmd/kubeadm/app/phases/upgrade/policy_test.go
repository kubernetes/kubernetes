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

	"k8s.io/kubernetes/pkg/util/version"
)

func TestEnforceVersionPolicies(t *testing.T) {
	tests := []struct {
		vg                          *fakeVersionGetter
		expectedMandatoryErrs       int
		expectedSkippableErrs       int
		allowExperimental, allowRCs bool
		newK8sVersion               string
	}{
		{ // everything ok
			vg: &fakeVersionGetter{
				clusterVersion: "v1.7.3",
				kubeletVersion: "v1.7.3",
				kubeadmVersion: "v1.7.5",
			},
			newK8sVersion: "v1.7.5",
		},
		{ // everything ok
			vg: &fakeVersionGetter{
				clusterVersion: "v1.7.3",
				kubeletVersion: "v1.7.2",
				kubeadmVersion: "v1.8.1",
			},
			newK8sVersion: "v1.8.0",
		},
		{ // downgrades not supported
			vg: &fakeVersionGetter{
				clusterVersion: "v1.7.3",
				kubeletVersion: "v1.7.3",
				kubeadmVersion: "v1.7.3",
			},
			newK8sVersion:         "v1.7.2",
			expectedSkippableErrs: 1,
		},
		{ // upgrades without bumping the version number not supported yet. TODO: Change this?
			vg: &fakeVersionGetter{
				clusterVersion: "v1.7.3",
				kubeletVersion: "v1.7.3",
				kubeadmVersion: "v1.7.3",
			},
			newK8sVersion:         "v1.7.3",
			expectedSkippableErrs: 1,
		},
		{ // new version must be higher than v1.7.0
			vg: &fakeVersionGetter{
				clusterVersion: "v1.7.3",
				kubeletVersion: "v1.7.3",
				kubeadmVersion: "v1.7.3",
			},
			newK8sVersion:         "v1.6.10",
			expectedMandatoryErrs: 1, // version must be higher than v1.7.0
			expectedSkippableErrs: 1, // version shouldn't be downgraded
		},
		{ // upgrading two minor versions in one go is not supported
			vg: &fakeVersionGetter{
				clusterVersion: "v1.7.3",
				kubeletVersion: "v1.7.3",
				kubeadmVersion: "v1.9.0",
			},
			newK8sVersion:         "v1.9.0",
			expectedMandatoryErrs: 1, // can't upgrade two minor versions
			expectedSkippableErrs: 1, // kubelet <-> apiserver skew too large
		},
		{ // kubeadm version must be higher than the new kube version. However, patch version skews may be forced
			vg: &fakeVersionGetter{
				clusterVersion: "v1.7.3",
				kubeletVersion: "v1.7.3",
				kubeadmVersion: "v1.7.3",
			},
			newK8sVersion:         "v1.7.5",
			expectedSkippableErrs: 1,
		},
		{ // kubeadm version must be higher than the new kube version. Trying to upgrade k8s to a higher minor version than kubeadm itself should never be supported
			vg: &fakeVersionGetter{
				clusterVersion: "v1.7.3",
				kubeletVersion: "v1.7.3",
				kubeadmVersion: "v1.7.3",
			},
			newK8sVersion:         "v1.8.0",
			expectedMandatoryErrs: 1,
		},
		{ // the maximum skew between the cluster version and the kubelet versions should be one minor version. This may be forced through though.
			vg: &fakeVersionGetter{
				clusterVersion: "v1.7.3",
				kubeletVersion: "v1.6.8",
				kubeadmVersion: "v1.8.0",
			},
			newK8sVersion:         "v1.8.0",
			expectedSkippableErrs: 1,
		},
		{ // experimental upgrades supported if the flag is set
			vg: &fakeVersionGetter{
				clusterVersion: "v1.7.3",
				kubeletVersion: "v1.7.3",
				kubeadmVersion: "v1.8.0-beta.1",
			},
			newK8sVersion:     "v1.8.0-beta.1",
			allowExperimental: true,
		},
		{ // release candidate upgrades supported if the flag is set
			vg: &fakeVersionGetter{
				clusterVersion: "v1.7.3",
				kubeletVersion: "v1.7.3",
				kubeadmVersion: "v1.8.0-rc.1",
			},
			newK8sVersion: "v1.8.0-rc.1",
			allowRCs:      true,
		},
		{ // release candidate upgrades supported if the flag is set
			vg: &fakeVersionGetter{
				clusterVersion: "v1.7.3",
				kubeletVersion: "v1.7.3",
				kubeadmVersion: "v1.8.0-rc.1",
			},
			newK8sVersion:     "v1.8.0-rc.1",
			allowExperimental: true,
		},
		{ // the user should not be able to upgrade to an experimental version if they haven't opted into that
			vg: &fakeVersionGetter{
				clusterVersion: "v1.7.3",
				kubeletVersion: "v1.7.3",
				kubeadmVersion: "v1.8.0-beta.1",
			},
			newK8sVersion:         "v1.8.0-beta.1",
			allowRCs:              true,
			expectedSkippableErrs: 1,
		},
		{ // the user should not be able to upgrade to an release candidate version if they haven't opted into that
			vg: &fakeVersionGetter{
				clusterVersion: "v1.7.3",
				kubeletVersion: "v1.7.3",
				kubeadmVersion: "v1.8.0-rc.1",
			},
			newK8sVersion:         "v1.8.0-rc.1",
			expectedSkippableErrs: 1,
		},
	}

	for _, rt := range tests {

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
			continue
		}

		if len(actualSkewErrs.Skippable) != rt.expectedSkippableErrs {
			t.Errorf("failed TestEnforceVersionPolicies\n\texpected skippable errors: %d\n\tgot skippable errors: %d %v", rt.expectedSkippableErrs, len(actualSkewErrs.Skippable), *rt.vg)
		}
		if len(actualSkewErrs.Mandatory) != rt.expectedMandatoryErrs {
			t.Errorf("failed TestEnforceVersionPolicies\n\texpected mandatory errors: %d\n\tgot mandatory errors: %d %v", rt.expectedMandatoryErrs, len(actualSkewErrs.Mandatory), *rt.vg)
		}
	}
}
