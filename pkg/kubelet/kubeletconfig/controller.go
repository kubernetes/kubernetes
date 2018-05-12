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
	"path/filepath"
	"time"

	apiv1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/validation"

	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint/store"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/status"
	utillog "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/log"
	utilpanic "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/panic"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

const (
	storeDir = "store"
	// TODO(mtaufen): We may expose this in a future API, but for the time being we use an internal default,
	// because it is not especially clear where this should live in the API.
	configTrialDuration = 10 * time.Minute
)

// Controller manages syncing dynamic Kubelet configurations
// For more information, see the proposal: https://github.com/kubernetes/community/blob/master/contributors/design-proposals/node/dynamic-kubelet-configuration.md
type Controller struct {
	// defaultConfig is the configuration to use if no initConfig is provided
	defaultConfig *kubeletconfig.KubeletConfiguration

	// pendingConfigSource; write to this channel to indicate that the config source needs to be synced from the API server
	pendingConfigSource chan bool

	// configOk manages the KubeletConfigOk condition that is reported in Node.Status.Conditions
	configOk status.ConfigOkCondition

	// informer is the informer that watches the Node object
	informer cache.SharedInformer

	// checkpointStore persists config source checkpoints to a storage layer
	checkpointStore store.Store
}

// NewController constructs a new Controller object and returns it. Directory paths must be absolute.
func NewController(defaultConfig *kubeletconfig.KubeletConfiguration, dynamicConfigDir string) *Controller {
	return &Controller{
		defaultConfig: defaultConfig,
		// channels must have capacity at least 1, since we signal with non-blocking writes
		pendingConfigSource: make(chan bool, 1),
		configOk:            status.NewConfigOkCondition(),
		checkpointStore:     store.NewFsStore(utilfs.DefaultFs{}, filepath.Join(dynamicConfigDir, storeDir)),
	}
}

// Bootstrap attempts to return a valid KubeletConfiguration based on the configuration of the Controller,
// or returns an error if no valid configuration could be produced. Bootstrap should be called synchronously before StartSync.
func (cc *Controller) Bootstrap() (*kubeletconfig.KubeletConfiguration, error) {
	utillog.Infof("starting controller")

	// ALWAYS validate the local config. This makes incorrectly provisioned nodes an error.
	// It must be valid because it is the default last-known-good config.
	utillog.Infof("validating local config")
	if err := validation.ValidateKubeletConfiguration(cc.defaultConfig); err != nil {
		return nil, fmt.Errorf("local config failed validation, error: %v", err)
	}

	// ensure the filesystem is initialized
	if err := cc.initializeDynamicConfigDir(); err != nil {
		return nil, err
	}

	assigned, curSource, reason, err := cc.loadAssignedConfig(cc.defaultConfig)
	if err == nil {
		// set the status to indicate we will use the assigned config
		if curSource != nil {
			cc.configOk.Set(fmt.Sprintf(status.CurRemoteMessageFmt, curSource.APIPath()), reason, apiv1.ConditionTrue)
		} else {
			cc.configOk.Set(status.CurLocalMessage, reason, apiv1.ConditionTrue)
		}

		// update the last-known-good config if necessary, and start a timer that
		// periodically checks whether the last-known good needs to be updated
		// we only do this when the assigned config loads and passes validation
		// wait.Forever will call the func once before starting the timer
		go wait.Forever(func() { cc.checkTrial(configTrialDuration) }, 10*time.Second)

		return assigned, nil
	} // Assert: the assigned config failed to load, parse, or validate

	// TODO(mtaufen): consider re-attempting download when a load/verify/parse/validate
	// error happens outside trial period, we already made it past the trial so it's probably filesystem corruption
	// or something else scary (unless someone is using a 0-length trial period)
	// load from checkpoint

	// log the reason and error details for the failure to load the assigned config
	utillog.Errorf(fmt.Sprintf("%s, error: %v", reason, err))

	// load the last-known-good config
	lkg, lkgSource, err := cc.loadLastKnownGoodConfig(cc.defaultConfig)
	if err != nil {
		return nil, err
	}

	// set the status to indicate that we had to roll back to the lkg for the reason reported when we tried to load the assigned config
	if lkgSource != nil {
		cc.configOk.Set(fmt.Sprintf(status.LkgRemoteMessageFmt, lkgSource.APIPath()), reason, apiv1.ConditionFalse)
	} else {
		cc.configOk.Set(status.LkgLocalMessage, reason, apiv1.ConditionFalse)
	}

	// return the last-known-good config
	return lkg, nil
}

// StartSync launches the controller's sync loops if `client` is non-nil and `nodeName` is non-empty.
// It will always start the Node condition reporting loop, and will also start the dynamic conifg sync loops
// if dynamic config is enabled on the controller. If `nodeName` is empty but `client` is non-nil, an error is logged.
func (cc *Controller) StartSync(client clientset.Interface, eventClient v1core.EventsGetter, nodeName string) {
	if client == nil {
		utillog.Infof("nil client, will not start sync loops")
		return
	} else if len(nodeName) == 0 {
		utillog.Errorf("cannot start sync loops with empty nodeName")
		return
	}

	// start the ConfigOk condition sync loop
	go utilpanic.HandlePanic(func() {
		utillog.Infof("starting ConfigOk condition sync loop")
		wait.JitterUntil(func() {
			cc.configOk.Sync(client, nodeName)
		}, 10*time.Second, 0.2, true, wait.NeverStop)
	})()

	cc.informer = newSharedNodeInformer(client, nodeName,
		cc.onAddNodeEvent, cc.onUpdateNodeEvent, cc.onDeleteNodeEvent)
	// start the informer loop
	// Rather than use utilruntime.HandleCrash, which doesn't actually crash in the Kubelet,
	// we use HandlePanic to manually call the panic handlers and then crash.
	// We have a better chance of recovering normal operation if we just restart the Kubelet in the event
	// of a Go runtime error.
	go utilpanic.HandlePanic(func() {
		utillog.Infof("starting Node informer sync loop")
		cc.informer.Run(wait.NeverStop)
	})()

	// start the config source sync loop
	go utilpanic.HandlePanic(func() {
		utillog.Infof("starting config source sync loop")
		wait.JitterUntil(func() {
			cc.syncConfigSource(client, eventClient, nodeName)
		}, 10*time.Second, 0.2, true, wait.NeverStop)
	})()

}

// loadAssignedConfig loads the Kubelet's currently assigned config,
// based on the setting in the local checkpoint store.
// It returns the loaded configuration, the checkpoint store's config source record,
// a clean success or failure reason that can be reported in the status, and any error that occurs.
// If the local config should be used, it will be returned. You should validate local before passing it to this function.
func (cc *Controller) loadAssignedConfig(local *kubeletconfig.KubeletConfiguration) (*kubeletconfig.KubeletConfiguration, checkpoint.RemoteConfigSource, string, error) {
	source, err := cc.checkpointStore.Current()
	if err != nil {
		return nil, nil, fmt.Sprintf(status.CurFailLoadReasonFmt, "unknown"), err
	}
	// nil source is the signal to use the local config
	if source == nil {
		return local, source, status.CurLocalOkayReason, nil
	}
	// load KubeletConfiguration from checkpoint
	kc, err := cc.checkpointStore.Load(source)
	if err != nil {
		return nil, source, fmt.Sprintf(status.CurFailLoadReasonFmt, source.APIPath()), err
	}
	if err := validation.ValidateKubeletConfiguration(kc); err != nil {
		return nil, source, fmt.Sprintf(status.CurFailValidateReasonFmt, source.APIPath()), err
	}
	return kc, source, status.CurRemoteOkayReason, nil
}

// loadLastKnownGoodConfig loads the Kubelet's last-known-good config,
// based on the setting in the local checkpoint store.
// It returns the loaded configuration, the checkpoint store's config source record,
// and any error that occurs.
// If the local config should be used, it will be returned. You should validate local before passing it to this function.
func (cc *Controller) loadLastKnownGoodConfig(local *kubeletconfig.KubeletConfiguration) (*kubeletconfig.KubeletConfiguration, checkpoint.RemoteConfigSource, error) {
	source, err := cc.checkpointStore.LastKnownGood()
	if err != nil {
		return nil, nil, fmt.Errorf("unable to determine last-known-good config, error: %v", err)
	}
	// nil source is the signal to use the local config
	if source == nil {
		return local, source, nil
	}
	// load from checkpoint
	kc, err := cc.checkpointStore.Load(source)
	if err != nil {
		return nil, source, fmt.Errorf("%s, error: %v", fmt.Sprintf(status.LkgFailLoadReasonFmt, source.APIPath()), err)
	}
	if err := validation.ValidateKubeletConfiguration(kc); err != nil {
		return nil, source, fmt.Errorf("%s, error: %v", fmt.Sprintf(status.LkgFailValidateReasonFmt, source.APIPath()), err)
	}
	return kc, source, nil
}

// initializeDynamicConfigDir makes sure that the storage layers for various controller components are set up correctly
func (cc *Controller) initializeDynamicConfigDir() error {
	utillog.Infof("ensuring filesystem is set up correctly")
	// initializeDynamicConfigDir local checkpoint storage location
	return cc.checkpointStore.Initialize()
}

// checkTrial checks whether the trial duration has passed, and updates the last-known-good config if necessary
func (cc *Controller) checkTrial(duration time.Duration) {
	// when the trial period is over, the assigned config becomes the last-known-good
	if trial, err := cc.inTrial(duration); err != nil {
		utillog.Errorf("failed to check trial period for assigned config, error: %v", err)
	} else if !trial {
		utillog.Infof("assigned config passed trial period, will set as last-known-good")
		if err := cc.graduateAssignedToLastKnownGood(); err != nil {
			utillog.Errorf("failed to set last-known-good to assigned config, error: %v", err)
		}
	}
}

// inTrial returns true if the time elapsed since the last modification of the current config does not exceed `trialDur`, false otherwise
func (cc *Controller) inTrial(trialDur time.Duration) (bool, error) {
	now := time.Now()
	t, err := cc.checkpointStore.CurrentModified()
	if err != nil {
		return false, err
	}
	if now.Sub(t) <= trialDur {
		return true, nil
	}
	return false, nil
}

// graduateAssignedToLastKnownGood sets the last-known-good in the checkpointStore
// to the same value as the current config maintained by the checkpointStore
func (cc *Controller) graduateAssignedToLastKnownGood() error {
	current, err := cc.checkpointStore.Current()
	if err != nil {
		return err
	}
	err = cc.checkpointStore.SetLastKnownGood(current)
	if err != nil {
		return err
	}
	return nil
}
