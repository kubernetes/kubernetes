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

package nodeconfig

import (
	"fmt"

	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	ccv1a1 "k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
)

// badRollback makes an entry in the bad-config-tracking file for `uid` with `reason`, and then
// returns the result of rolling back to the last-known-good config.
// If filesystem issues prevent marking the config bad or rolling back, a fatal error occurs.
func (cc *NodeConfigController) badRollback(uid, reason, detail string) *ccv1a1.KubeletConfiguration {
	if len(detail) > 0 {
		detail = fmt.Sprintf(", %s", detail)
	}
	errorf(fmt.Sprintf("%s%s", reason, detail))
	cc.markBadConfig(uid, reason)
	return cc.lkgRollback(reason, apiv1.ConditionFalse)
}

// lkgRollback loads, verifies, parses, validates, and returns the last-known-good configuration,
// and updates the `cc.configOK` condition regarding the `cause` of the rollback. The `status` argument
// indicates whether the rollback is due to a known-bad config (apiv1.ConditionFalse) or because the system
// couldn't sync a configuration (statusUnknown).
// If the `defaultConfig` or `initConfig` is the last-known-good, the cached copies are immediately returned.
// If the last-known-good fails any of load, verify, parse, or validate,
// attempts to report the associated ConfigOK condition and a fatal error occurs.
// If filesystem issues prevent returning the last-known-good configuration, a fatal error occurs.
func (cc *NodeConfigController) lkgRollback(cause string, status apiv1.ConditionStatus) *ccv1a1.KubeletConfiguration {
	infof("rolling back to last-known-good config")

	// if lkgUID indicates the default should be used, return initConfig or defaultConfig
	lkgUID := cc.lkgUID()
	if len(lkgUID) == 0 {
		if cc.initConfig != nil {
			cc.setConfigOK("using last-known-good (init)", cause, status)
			return cc.initConfig
		}
		cc.setConfigOK("using last-known-good (default)", cause, status)
		return cc.defaultConfig
	}

	// load
	toVerify, err := cc.loadCheckpoint(lkgSymlink)
	if err != nil {
		cause := fmt.Sprintf("failed to load last-known-good (UID: %q)", lkgUID)
		cc.fatalSyncConfigOK(cause)
		fatalf("%s, error: %v", cause, err)
	}

	// verify
	toParse, err := toVerify.verify()
	if err != nil {
		cause := fmt.Sprintf("failed to verify last-known-good (UID: %q)", lkgUID)
		cc.fatalSyncConfigOK(cause)
		fatalf("%s, error: %v", cause, err)
	}

	// parse
	lkg, err := toParse.parse()
	if err != nil {
		cause := fmt.Sprintf("failed to parse last-known-good (UID: %q)", lkgUID)
		cc.fatalSyncConfigOK(cause)
		fatalf("%s, error: %v", cause, err)
	}

	// validate
	if err := validateConfig(lkg); err != nil {
		cause := fmt.Sprintf("failed to validate last-known-good (UID: %q)", lkgUID)
		cc.fatalSyncConfigOK(cause)
		fatalf("%s, error: %v", cause, err)
	}

	// update the ConfigOK status
	cc.setConfigOK("using last-known-good (UID: %q)", cause, status)
	return lkg
}
