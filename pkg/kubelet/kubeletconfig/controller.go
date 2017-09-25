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
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/validation"

	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint/store"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/configfiles"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/status"
	utillog "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/log"
	utilpanic "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/panic"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

const (
	checkpointsDir = "checkpoints"
	initConfigDir  = "init"
)

// Controller is the controller which, among other things:
// - loads configuration from disk
// - checkpoints configuration to disk
// - downloads new configuration from the API server
// - validates configuration
// - tracks the last-known-good configuration, and rolls-back to last-known-good when necessary
// For more information, see the proposal: https://github.com/kubernetes/community/blob/master/contributors/design-proposals/dynamic-kubelet-configuration.md
type Controller struct {
	// dynamicConfig, if true, indicates that we should sync config from the API server
	dynamicConfig bool

	// defaultConfig is the configuration to use if no initConfig is provided
	defaultConfig *kubeletconfig.KubeletConfiguration

	// initConfig is the unmarshaled init config, this will be loaded by the Controller if an initConfigDir is provided
	initConfig *kubeletconfig.KubeletConfiguration

	// initLoader is for loading the Kubelet's init configuration files from disk
	initLoader configfiles.Loader

	// pendingConfigSource; write to this channel to indicate that the config source needs to be synced from the API server
	pendingConfigSource chan bool

	// configOK manages the ConfigOK condition that is reported in Node.Status.Conditions
	configOK status.ConfigOKCondition

	// informer is the informer that watches the Node object
	informer cache.SharedInformer

	// checkpointStore persists config source checkpoints to a storage layer
	checkpointStore store.Store
}

// NewController constructs a new Controller object and returns it. Directory paths must be absolute.
// If the `initConfigDir` is an empty string, skips trying to load the init config.
// If the `dynamicConfigDir` is an empty string, skips trying to load checkpoints or download new config,
// but will still sync the ConfigOK condition if you call StartSync with a non-nil client.
func NewController(defaultConfig *kubeletconfig.KubeletConfiguration,
	initConfigDir string,
	dynamicConfigDir string) (*Controller, error) {
	var err error

	fs := utilfs.DefaultFs{}

	var initLoader configfiles.Loader
	if len(initConfigDir) > 0 {
		initLoader, err = configfiles.NewFsLoader(fs, initConfigDir)
		if err != nil {
			return nil, err
		}
	}
	dynamicConfig := false
	if len(dynamicConfigDir) > 0 {
		dynamicConfig = true
	}

	return &Controller{
		dynamicConfig: dynamicConfig,
		defaultConfig: defaultConfig,
		// channels must have capacity at least 1, since we signal with non-blocking writes
		pendingConfigSource: make(chan bool, 1),
		configOK:            status.NewConfigOKCondition(),
		checkpointStore:     store.NewFsStore(fs, filepath.Join(dynamicConfigDir, checkpointsDir)),
		initLoader:          initLoader,
	}, nil
}

// Bootstrap attempts to return a valid KubeletConfiguration based on the configuration of the Controller,
// or returns an error if no valid configuration could be produced. Bootstrap should be called synchronously before StartSync.
func (cc *Controller) Bootstrap() (*kubeletconfig.KubeletConfiguration, error) {
	utillog.Infof("starting controller")

	// ALWAYS validate the local (default and init) configs. This makes incorrectly provisioned nodes an error.
	// These must be valid because they are the foundational last-known-good configs.
	utillog.Infof("validating combination of defaults and flags")
	if err := validation.ValidateKubeletConfiguration(cc.defaultConfig); err != nil {
		return nil, fmt.Errorf("combination of defaults and flags failed validation, error: %v", err)
	}
	// only attempt to load and validate the init config if the user provided a path
	if cc.initLoader != nil {
		utillog.Infof("loading init config")
		kc, err := cc.initLoader.Load()
		if err != nil {
			return nil, err
		}
		// validate the init config
		utillog.Infof("validating init config")
		if err := validation.ValidateKubeletConfiguration(kc); err != nil {
			return nil, fmt.Errorf("failed to validate the init config, error: %v", err)
		}
		cc.initConfig = kc
	}
	// Assert: the default and init configs are both valid

	// if dynamic config is disabled, skip trying to load any checkpoints because they won't exist
	if !cc.dynamicConfig {
		return cc.localConfig(), nil
	}

	// assert: now we know that a dynamicConfigDir was provided, and we can rely on that existing

	// make sure the filesystem is set up properly
	// TODO(mtaufen): rename this to initializeDynamicConfigDir
	if err := cc.initialize(); err != nil {
		return nil, err
	}

	// determine UID of the current config source
	curUID := ""
	if curSource, err := cc.checkpointStore.Current(); err != nil {
		return nil, err
	} else if curSource != nil {
		curUID = curSource.UID()
	}

	// if curUID indicates the local config should be used, return the correct one of those
	if len(curUID) == 0 {
		return cc.localConfig(), nil
	} // Assert: we will not use the local configurations, unless we roll back to lkg; curUID is non-empty

	// TODO(mtaufen): consider re-verifying integrity and re-attempting download when a load/verify/parse/validate
	// error happens outside trial period, we already made it past the trial so it's probably filesystem corruption
	// or something else scary (unless someone is using a 0-length trial period)

	// load the current config
	checkpoint, err := cc.checkpointStore.Load(curUID)
	if err != nil {
		// TODO(mtaufen): rollback for now, but this could reasonably be handled by re-attempting a download,
		// it probably indicates some sort of corruption
		return cc.lkgRollback(fmt.Sprintf(status.CurFailLoadReasonFmt, curUID), fmt.Sprintf("error: %v", err))
	}

	// parse the checkpoint into a KubeletConfiguration
	cur, err := checkpoint.Parse()
	if err != nil {
		return cc.lkgRollback(fmt.Sprintf(status.CurFailParseReasonFmt, curUID), fmt.Sprintf("error: %v", err))
	}

	// validate current config
	if err := validation.ValidateKubeletConfiguration(cur); err != nil {
		return cc.lkgRollback(fmt.Sprintf(status.CurFailValidateReasonFmt, curUID), fmt.Sprintf("error: %v", err))
	}

	// when the trial period is over, the current config becomes the last-known-good
	if trial, err := cc.inTrial(cur.ConfigTrialDuration.Duration); err != nil {
		return nil, err
	} else if !trial {
		if err := cc.graduateCurrentToLastKnownGood(); err != nil {
			return nil, err
		}
	}

	// update the status to note that we will use the current config
	cc.configOK.Set(fmt.Sprintf(status.CurRemoteMessageFmt, curUID), status.CurRemoteOKReason, apiv1.ConditionTrue)
	return cur, nil
}

// StartSync launches the controller's sync loops if `client` is non-nil and `nodeName` is non-empty.
// It will always start the Node condition reporting loop, and will also start the dynamic conifg sync loops
// if dynamic config is enabled on the controller. If `nodeName` is empty but `client` is non-nil, an error is logged.
func (cc *Controller) StartSync(client clientset.Interface, nodeName string) {
	if client == nil {
		utillog.Infof("nil client, will not start sync loops")
		return
	} else if len(nodeName) == 0 {
		utillog.Errorf("cannot start sync loops with empty nodeName")
		return
	}

	// start the ConfigOK condition sync loop
	go utilpanic.HandlePanic(func() {
		utillog.Infof("starting ConfigOK condition sync loop")
		wait.JitterUntil(func() {
			cc.configOK.Sync(client, nodeName)
		}, 10*time.Second, 0.2, true, wait.NeverStop)
	})()

	// only sync to new, remotely provided configurations if dynamic config was enabled
	if cc.dynamicConfig {
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
				cc.syncConfigSource(client, nodeName)
			}, 10*time.Second, 0.2, true, wait.NeverStop)
		})()
	} else {
		utillog.Infof("dynamic config not enabled, will not sync to remote config")
	}
}

// initialize makes sure that the storage layers for various controller components are set up correctly
func (cc *Controller) initialize() error {
	utillog.Infof("ensuring filesystem is set up correctly")
	// initialize local checkpoint storage location
	if err := cc.checkpointStore.Initialize(); err != nil {
		return err
	}
	return nil
}

// localConfig returns the initConfig if it is loaded, otherwise returns the defaultConfig.
// It also sets the local configOK condition to match the returned config.
func (cc *Controller) localConfig() *kubeletconfig.KubeletConfiguration {
	if cc.initConfig != nil {
		cc.configOK.Set(status.CurInitMessage, status.CurInitOKReason, apiv1.ConditionTrue)
		return cc.initConfig
	}
	cc.configOK.Set(status.CurDefaultMessage, status.CurDefaultOKReason, apiv1.ConditionTrue)
	return cc.defaultConfig
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

// graduateCurrentToLastKnownGood sets the last-known-good UID on the checkpointStore
// to the same value as the current UID maintained by the checkpointStore
func (cc *Controller) graduateCurrentToLastKnownGood() error {
	curUID, err := cc.checkpointStore.Current()
	if err != nil {
		return fmt.Errorf("could not graduate last-known-good config to current config, error: %v", err)
	}
	err = cc.checkpointStore.SetLastKnownGood(curUID)
	if err != nil {
		return fmt.Errorf("could not graduate last-known-good config to current config, error: %v", err)
	}
	return nil
}
