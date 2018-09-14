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

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/cache"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/apis/config/validation"

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

// TransformFunc edits the KubeletConfiguration in-place, and returns an
// error if any of the transformations failed.
type TransformFunc func(kc *kubeletconfig.KubeletConfiguration) error

// Controller manages syncing dynamic Kubelet configurations
// For more information, see the proposal: https://github.com/kubernetes/community/blob/master/contributors/design-proposals/node/dynamic-kubelet-configuration.md
type Controller struct {
	// transform applies an arbitrary transformation to config after loading, and before validation.
	// This can be used, for example, to include config from flags before the controller's validation step.
	// If transform returns an error, loadConfig will fail, and an InternalError will be reported.
	// Be wary if using this function as an extension point, in most cases the controller should
	// probably just be natively extended to do what you need. Injecting flag precedence transformations
	// is something of an exception because the caller of this controller (cmd/) is aware of flags, but this
	// controller's tree (pkg/) is not.
	transform TransformFunc

	// pendingConfigSource; write to this channel to indicate that the config source needs to be synced from the API server
	pendingConfigSource chan bool

	// configStatus manages the status we report on the Node object
	configStatus status.NodeConfigStatus

	// nodeInformer is the informer that watches the Node object
	nodeInformer cache.SharedInformer

	// remoteConfigSourceInformer is the informer that watches the assigned config source
	remoteConfigSourceInformer cache.SharedInformer

	// checkpointStore persists config source checkpoints to a storage layer
	checkpointStore store.Store
}

// NewController constructs a new Controller object and returns it. The dynamicConfigDir
// path must be absolute. transform applies an arbitrary transformation to config after loading, and before validation.
// This can be used, for example, to include config from flags before the controller's validation step.
// If transform returns an error, loadConfig will fail, and an InternalError will be reported.
// Be wary if using this function as an extension point, in most cases the controller should
// probably just be natively extended to do what you need. Injecting flag precedence transformations
// is something of an exception because the caller of this controller (cmd/) is aware of flags, but this
// controller's tree (pkg/) is not.
func NewController(dynamicConfigDir string, transform TransformFunc) *Controller {
	return &Controller{
		transform: transform,
		// channels must have capacity at least 1, since we signal with non-blocking writes
		pendingConfigSource: make(chan bool, 1),
		configStatus:        status.NewNodeConfigStatus(),
		checkpointStore:     store.NewFsStore(utilfs.DefaultFs{}, filepath.Join(dynamicConfigDir, storeDir)),
	}
}

// Bootstrap attempts to return a valid KubeletConfiguration based on the configuration of the Controller,
// or returns an error if no valid configuration could be produced. Bootstrap should be called synchronously before StartSync.
// If the pre-existing local configuration should be used, Bootstrap returns a nil config.
func (cc *Controller) Bootstrap() (*kubeletconfig.KubeletConfiguration, error) {
	utillog.Infof("starting controller")

	// ensure the filesystem is initialized
	if err := cc.initializeDynamicConfigDir(); err != nil {
		return nil, err
	}

	// determine assigned source and set status
	assignedSource, err := cc.checkpointStore.Assigned()
	if err != nil {
		return nil, err
	}
	if assignedSource != nil {
		cc.configStatus.SetAssigned(assignedSource.NodeConfigSource())
	}

	// determine last-known-good source and set status
	lastKnownGoodSource, err := cc.checkpointStore.LastKnownGood()
	if err != nil {
		return nil, err
	}
	if lastKnownGoodSource != nil {
		cc.configStatus.SetLastKnownGood(lastKnownGoodSource.NodeConfigSource())
	}

	// if the assigned source is nil, return nil to indicate local config
	if assignedSource == nil {
		return nil, nil
	}

	// attempt to load assigned config
	assignedConfig, reason, err := cc.loadConfig(assignedSource)
	if err == nil {
		// update the active source to the non-nil assigned source
		cc.configStatus.SetActive(assignedSource.NodeConfigSource())

		// update the last-known-good config if necessary, and start a timer that
		// periodically checks whether the last-known good needs to be updated
		// we only do this when the assigned config loads and passes validation
		// wait.Forever will call the func once before starting the timer
		go wait.Forever(func() { cc.checkTrial(configTrialDuration) }, 10*time.Second)

		return assignedConfig, nil
	} // Assert: the assigned config failed to load or validate

	// TODO(mtaufen): consider re-attempting download when a load/verify/parse/validate
	// error happens outside trial period, we already made it past the trial so it's probably filesystem corruption
	// or something else scary

	// log the reason and error details for the failure to load the assigned config
	utillog.Errorf(fmt.Sprintf("%s, error: %v", reason, err))

	// set status to indicate the failure with the assigned config
	cc.configStatus.SetError(reason)

	// if the last-known-good source is nil, return nil to indicate local config
	if lastKnownGoodSource == nil {
		return nil, nil
	}

	// attempt to load the last-known-good config
	lastKnownGoodConfig, _, err := cc.loadConfig(lastKnownGoodSource)
	if err != nil {
		// we failed to load the last-known-good, so something is really messed up and we just return the error
		return nil, err
	}

	// set status to indicate the active source is the non-nil last-known-good source
	cc.configStatus.SetActive(lastKnownGoodSource.NodeConfigSource())
	return lastKnownGoodConfig, nil
}

// StartSync tells the controller to start the goroutines that sync status/config to/from the API server.
// The clients must be non-nil, and the nodeName must be non-empty.
func (cc *Controller) StartSync(client clientset.Interface, eventClient v1core.EventsGetter, nodeName string) error {
	const errFmt = "cannot start Kubelet config sync: %s"
	if client == nil {
		return fmt.Errorf(errFmt, "nil client")
	}
	if eventClient == nil {
		return fmt.Errorf(errFmt, "nil event client")
	}
	if nodeName == "" {
		return fmt.Errorf(errFmt, "empty nodeName")
	}

	// Rather than use utilruntime.HandleCrash, which doesn't actually crash in the Kubelet,
	// we use HandlePanic to manually call the panic handlers and then crash.
	// We have a better chance of recovering normal operation if we just restart the Kubelet in the event
	// of a Go runtime error.
	// NOTE(mtaufen): utilpanic.HandlePanic returns a function and you have to call it for your thing to run!
	// This was EVIL to debug (difficult to see missing `()`).
	// The code now uses `go name()` instead of `go utilpanic.HandlePanic(func(){...})()` to avoid confusion.

	// status sync worker
	statusSyncLoopFunc := utilpanic.HandlePanic(func() {
		utillog.Infof("starting status sync loop")
		wait.JitterUntil(func() {
			cc.configStatus.Sync(client, nodeName)
		}, 10*time.Second, 0.2, true, wait.NeverStop)
	})
	// remote config source informer, if we have a remote source to watch
	assignedSource, err := cc.checkpointStore.Assigned()
	if err != nil {
		return fmt.Errorf(errFmt, err)
	} else if assignedSource == nil {
		utillog.Infof("local source is assigned, will not start remote config source informer")
	} else {
		cc.remoteConfigSourceInformer = assignedSource.Informer(client, cache.ResourceEventHandlerFuncs{
			AddFunc:    cc.onAddRemoteConfigSourceEvent,
			UpdateFunc: cc.onUpdateRemoteConfigSourceEvent,
			DeleteFunc: cc.onDeleteRemoteConfigSourceEvent,
		},
		)
	}
	remoteConfigSourceInformerFunc := utilpanic.HandlePanic(func() {
		if cc.remoteConfigSourceInformer != nil {
			utillog.Infof("starting remote config source informer")
			cc.remoteConfigSourceInformer.Run(wait.NeverStop)
		}
	})
	// node informer
	cc.nodeInformer = newSharedNodeInformer(client, nodeName,
		cc.onAddNodeEvent, cc.onUpdateNodeEvent, cc.onDeleteNodeEvent)
	nodeInformerFunc := utilpanic.HandlePanic(func() {
		utillog.Infof("starting Node informer")
		cc.nodeInformer.Run(wait.NeverStop)
	})
	// config sync worker
	configSyncLoopFunc := utilpanic.HandlePanic(func() {
		utillog.Infof("starting Kubelet config sync loop")
		wait.JitterUntil(func() {
			cc.syncConfigSource(client, eventClient, nodeName)
		}, 10*time.Second, 0.2, true, wait.NeverStop)
	})

	go statusSyncLoopFunc()
	go remoteConfigSourceInformerFunc()
	go nodeInformerFunc()
	go configSyncLoopFunc()
	return nil
}

// loadConfig loads Kubelet config from a checkpoint
// It returns the loaded configuration or a clean failure reason (for status reporting) and an error.
func (cc *Controller) loadConfig(source checkpoint.RemoteConfigSource) (*kubeletconfig.KubeletConfiguration, string, error) {
	// load KubeletConfiguration from checkpoint
	kc, err := cc.checkpointStore.Load(source)
	if err != nil {
		return nil, status.LoadError, err
	}
	// apply any required transformations to the KubeletConfiguration
	if cc.transform != nil {
		if err := cc.transform(kc); err != nil {
			return nil, status.InternalError, err
		}
	}
	// validate the result
	if err := validation.ValidateKubeletConfiguration(kc); err != nil {
		return nil, status.ValidateError, err
	}
	return kc, "", nil
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
		if err := cc.graduateAssignedToLastKnownGood(); err != nil {
			utillog.Errorf("failed to set last-known-good to assigned config, error: %v", err)
		}
	}
}

// inTrial returns true if the time elapsed since the last modification of the assigned config does not exceed `trialDur`, false otherwise
func (cc *Controller) inTrial(trialDur time.Duration) (bool, error) {
	now := time.Now()
	t, err := cc.checkpointStore.AssignedModified()
	if err != nil {
		return false, err
	}
	if now.Sub(t) <= trialDur {
		return true, nil
	}
	return false, nil
}

// graduateAssignedToLastKnownGood sets the last-known-good in the checkpointStore
// to the same value as the assigned config maintained by the checkpointStore
func (cc *Controller) graduateAssignedToLastKnownGood() error {
	// get assigned
	assigned, err := cc.checkpointStore.Assigned()
	if err != nil {
		return err
	}
	// get last-known-good
	lastKnownGood, err := cc.checkpointStore.LastKnownGood()
	if err != nil {
		return err
	}
	// if the sources are equal, no need to change
	if assigned == lastKnownGood ||
		assigned != nil && lastKnownGood != nil && apiequality.Semantic.DeepEqual(assigned, lastKnownGood) {
		return nil
	}
	// update last-known-good
	err = cc.checkpointStore.SetLastKnownGood(assigned)
	if err != nil {
		return err
	}
	// update the status to reflect the new last-known-good config
	cc.configStatus.SetLastKnownGood(assigned.NodeConfigSource())
	utillog.Infof("updated last-known-good config to %s, UID: %s, ResourceVersion: %s", assigned.APIPath(), assigned.UID(), assigned.ResourceVersion())
	return nil
}
