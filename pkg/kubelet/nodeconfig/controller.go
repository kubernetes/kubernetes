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
	"math/rand"
	"os"
	"runtime"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kuberuntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	ccv1a1 "k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

// NodeConfigController is the controller which, among other things:
// - loads configuration from disk
// - checkpoints configuration to disk
// - downloads new configuration from the API server
// - validates configuration
// - monitors for potential crash-loops caused by new configurations
// - tracks the last-known-good configuration, and rolls-back to last-known-good when necessary
// For more information, see the proposal: https://github.com/kubernetes/kubernetes/pull/29459
type NodeConfigController struct {
	// configDir is the root directory to use for nodeconfig management
	configDir string

	// defaultConfig is the configuration to use if no initConfig is provided
	defaultConfig *ccv1a1.KubeletConfiguration

	// initConfig is the unmarshaled init config, this will be loaded by the NodeConfigController if it exists in the configDir
	initConfig *ccv1a1.KubeletConfiguration

	// client is the clientset for talking to the apiserver.
	client clientset.Interface

	// nodeName is the name of the Node object we should monitor for config
	nodeName string

	// configOK is the current ConfigOK node condition, which will be reported in the Node.status.conditions
	configOK *apiv1.NodeCondition
}

// NewNodeConfigController constructs a new NodeConfigController object and returns it.
// If the client is nil, dynamic configuration (watching the API server) will not be used.
func NewNodeConfigController(client clientset.Interface, nodeName string, configDir string, defaultConfig *ccv1a1.KubeletConfiguration) *NodeConfigController {
	return &NodeConfigController{
		configDir:     configDir,
		defaultConfig: defaultConfig,
		client:        client,
		nodeName:      nodeName,
	}
}

// Bootstrap initiates operation of the NodeConfigController.
// If a valid configuration is found as a result of these operations, that configuration is returned.
// If a valid configuration cannot be found, a fatal error occurs, preventing the Kubelet from continuing with invalid configuration.
// Bootstrap must be called synchronously during Kubelet startup, before any KubeletConfiguration is used.
// If Bootstrap completes successfully, you can optionally call StartSyncLoop to watch the API server for config updates.
func (cc *NodeConfigController) Bootstrap() (finalConfig *ccv1a1.KubeletConfiguration, fatalErr error) {
	var curUID string

	// defer updating status until the end of Run
	defer func() {
		if r := recover(); r != nil {
			if _, ok := r.(runtime.Error); ok {
				panic(r)
			}
			fatalErr = r.(error)
		}
		cc.syncConfigOK()
	}()

	infof("starting controller")

	// make sure the node-config-dir is set up properly
	cc.ensureCfgDir()

	// record the kubelet startup time, used for crashloop detection
	cc.recordStartup()

	// ALWAYS validate the default and init configs. This makes incorrectly provisioned nodes a fatal error.
	// These must be valid because they are the foundational last-known-good configs.
	infof("validating combination of defaults and flags")
	if err := validateConfig(cc.defaultConfig); err != nil {
		fatalf("combination of defaults and flags failed validation, error: %v", err)
	}
	cc.loadInitConfig()
	cc.validateInitConfig()
	// Assert: the default and init configs are both valid

	// if the client is nil, just return the (now validated) init or default configuration
	if cc.client == nil {
		if cc.initConfig != nil {
			finalConfig = cc.initConfig
			return
		}
		finalConfig = cc.defaultConfig
		return
	}

	// check the Node and download any new config
	node, err := cc.client.CoreV1().Nodes().Get(cc.nodeName, metav1.GetOptions{})
	if err != nil {
		// we don't roll back here because it's possible the Node isn't registered yet
		errorf("couldn't get Node %q during initialization, error: %v", cc.nodeName, err)
	} else {
		_, cause, err := cc.syncNodeConfig(node)
		if err != nil {
			// if the Node exists and we can't get the current config, we roll back to the last-known-good,
			// as this is the safest thing to do when we can't figure out what the user wants
			// the sync loop will check again after its full-resync interval
			msg := "failed to sync, desired config unclear"
			errorf("%s, error: %v", msg, err)
			finalConfig = cc.lkgRollback(fmt.Sprintf("%s, cause: %s", msg, cause), apiv1.ConditionUnknown)
			return
		}
	}

	// determine UID of the current config source, empty string if curSymlink targets default
	curUID = cc.curUID()

	// if curUID indicates the default should be used, return initConfig or defaultConfig
	if len(curUID) == 0 {
		if cc.initConfig != nil {
			cc.setConfigOK("using current (init)",
				"current is set to the local default, and an init config was provided", apiv1.ConditionTrue)
			finalConfig = cc.initConfig
			return
		}
		cc.setConfigOK("using current (default)",
			"current is set to the local default, and no init config was provided", apiv1.ConditionTrue)
		finalConfig = cc.defaultConfig
		return
	} // Assert: we will not use the init or default configurations, unless we roll back to lkg; curUID is a real UID

	// check whether the current config is marked bad
	if bad, entry := cc.isBadConfig(curUID); bad {
		infof("current config %q was marked bad for reason %q at time %q", curUID, entry.reason, entry.time)
		finalConfig = cc.lkgRollback(entry.reason, apiv1.ConditionFalse)
		return
	}

	// TODO(mtaufen): consider re-verifying integrity and re-attempting download when a load/verify/parse/validate
	// error happens outside trial period, we already made it past the trial so it's probably filesystem corruption
	// or something else scary

	// load the current config
	toVerify, err := cc.loadCheckpoint(curSymlink)
	if err != nil {
		// TODO(mtaufen): rollback and mark bad for now, but this could reasonably be handled by re-attempting a download,
		// it probably indicates some sort of corruption
		finalConfig = cc.badRollback(curUID, fmt.Sprintf("failed to load current (UID: %q)", curUID), fmt.Sprintf("error: %v", err))
		return
	}

	// verify the integrity of the configuration we just loaded
	toParse, err := toVerify.verify()
	if err != nil {
		finalConfig = cc.badRollback(curUID, fmt.Sprintf("failed to verify current (UID: %q)", curUID), fmt.Sprintf("error: %v", err))
		return
	}

	// parse the configuration we just loaded into a KubeletConfiguration
	cur, err := toParse.parse()
	if err != nil {
		finalConfig = cc.badRollback(curUID, fmt.Sprintf("failed to parse current (UID: %q)", curUID), fmt.Sprintf("error: %v", err))
		return
	}

	// validate current config
	if err := validateConfig(cur); err != nil {
		finalConfig = cc.badRollback(curUID, fmt.Sprintf("failed to validate current (UID: %q)", curUID), fmt.Sprintf("error: %v", err))
		return
	}

	// check for crash loops if we're still in the trial period
	if cc.curInTrial(cur.ConfigTrialDuration.Duration) {
		if cc.crashLooping(*cur.CrashLoopThreshold) {
			finalConfig = cc.badRollback(curUID, fmt.Sprintf("current failed trial period due to crash loop (UID %q)", curUID), "")
			return
		}
	} else if !cc.curIsLkg() {
		// when the trial period is over, the current config becomes the last-known-good
		cc.setCurAsLkg()
	}

	// update the status to note that we will use the current config
	cc.setConfigOK(fmt.Sprintf("using current (UID: %q)", curUID), "passed all checks", apiv1.ConditionTrue)
	finalConfig = cur
	return
}

// StartSyncLoop launches the sync loop that watches the API server for configuration updates
// The `client` passed to StartSyncLoop will replace the controller's current client
func (cc *NodeConfigController) StartSyncLoop(client clientset.Interface) {
	if client != nil {
		cc.client = client
		infof("starting sync loop")
		go func() {
			fieldselector := fmt.Sprintf("metadata.name=%s", cc.nodeName)

			// Add some randomness to resync period, which can help avoid controllers falling into lock-step
			minResyncPeriod := 30 * time.Second
			factor := rand.Float64() + 1
			resyncPeriod := time.Duration(float64(minResyncPeriod.Nanoseconds()) * factor)

			lw := &cache.ListWatch{
				ListFunc: func(options metav1.ListOptions) (kuberuntime.Object, error) {
					return cc.client.Core().Nodes().List(metav1.ListOptions{
						FieldSelector: fieldselector,
					})
				},
				WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
					return cc.client.Core().Nodes().Watch(metav1.ListOptions{
						FieldSelector:   fieldselector,
						ResourceVersion: options.ResourceVersion,
					})
				},
			}

			handler := cache.ResourceEventHandlerFuncs{
				AddFunc: func(n interface{}) {
					cc.onAddNodeEvent(n)
				},
				UpdateFunc: func(old interface{}, nw interface{}) {
					cc.onWatchNodeEvent(nw)
				},
				DeleteFunc: func(n interface{}) {
					return // do nothing, the node is probably being torn down
				},
			}

			informer := cache.NewSharedInformer(lw, &apiv1.Node{}, resyncPeriod)
			informer.AddEventHandler(handler)
			stop := make(chan struct{})
			informer.Run(stop)
			return
		}()
	} else {
		errorf("cannot start sync loop with nil client")
	}
}

// onAddEvent syncs any status that was set before the Node was registered,
// then calls onWatchNodeEvent to complete event handling.
func (cc *NodeConfigController) onAddNodeEvent(n interface{}) {
	defer func() {
		// catch controller-level panics, the actual cause of controller-level
		// fatal-class errors will have already been logged by fatalf (see log.go)
		if r := recover(); r != nil {
			if _, ok := r.(runtime.Error); ok {
				panic(r)
			}
		}
	}()
	cc.syncConfigOK()
	cc.onWatchNodeEvent(n)
}

// onWatchNodeEvent checks for new config and downloads it if necessary.
// If filesystem issues prevent proper operation of syncNodeConfig, a fatal error occurs.
func (cc *NodeConfigController) onWatchNodeEvent(n interface{}) {
	defer func() {
		// catch controller-level panics, the actual cause of controller-level
		// fatal-class errors will have already been logged by fatalf (see log.go)
		if r := recover(); r != nil {
			if _, ok := r.(runtime.Error); ok {
				panic(r)
			}
		}
	}()

	node, ok := n.(*apiv1.Node)
	if !ok {
		errorf("failed to cast watched object to Node, couldn't handle event")
		return
	}

	// check the Node and download any new config
	if updated, cause, err := cc.syncNodeConfig(node); err != nil {
		errorf("failed to sync node config, error: %v", err)
		// Update the ConfigOK status to reflect that we failed to sync, and so we don't know which configuration
		// the user actually wants us to use. In this case, we just continue using the currently-in-use configuration.
		cc.setConfigOK(cc.configOK.Message, fmt.Sprintf("failed to sync, desired config unclear, cause: %s", cause), apiv1.ConditionUnknown)
		cc.syncConfigOK()
	} else if updated {
		infof("config updated, Kubelet will restart to begin using new config")
		os.Exit(0)
	}
}
