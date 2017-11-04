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

package kubeletconfig

import (
	"fmt"

	apiv1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/validation"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/status"
	utillog "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/log"
)

// lkgRollback returns a valid last-known-good configuration, and updates the `cc.configOK` condition
// regarding the `reason` for the rollback, or returns an error if a valid last-known-good could not be produced
func (cc *Controller) lkgRollback(reason, detail string) (*kubeletconfig.KubeletConfiguration, error) {
	utillog.Errorf(fmt.Sprintf("%s, %s", reason, detail))

	lkgUID := ""
	if lkgSource, err := cc.checkpointStore.LastKnownGood(); err != nil {
		return nil, fmt.Errorf("unable to determine last-known-good config, error: %v", err)
	} else if lkgSource != nil {
		lkgUID = lkgSource.UID()
	}

	// if lkgUID indicates the default should be used, return initConfig or defaultConfig
	if len(lkgUID) == 0 {
		if cc.initConfig != nil {
			cc.configOK.Set(status.LkgInitMessage, reason, apiv1.ConditionFalse)
			return cc.initConfig, nil
		}
		cc.configOK.Set(status.LkgDefaultMessage, reason, apiv1.ConditionFalse)
		return cc.defaultConfig, nil
	}

	// load
	checkpoint, err := cc.checkpointStore.Load(lkgUID)
	if err != nil {
		return nil, fmt.Errorf("%s, error: %v", fmt.Sprintf(status.LkgFailLoadReasonFmt, lkgUID), err)
	}

	// parse
	lkg, err := checkpoint.Parse()
	if err != nil {
		return nil, fmt.Errorf("%s, error: %v", fmt.Sprintf(status.LkgFailParseReasonFmt, lkgUID), err)
	}

	// validate
	if err := validation.ValidateKubeletConfiguration(lkg); err != nil {
		return nil, fmt.Errorf("%s, error: %v", fmt.Sprintf(status.LkgFailValidateReasonFmt, lkgUID), err)
	}

	cc.configOK.Set(fmt.Sprintf(status.LkgRemoteMessageFmt, lkgUID), reason, apiv1.ConditionFalse)
	return lkg, nil
}
