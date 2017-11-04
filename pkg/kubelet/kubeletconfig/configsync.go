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
	"os"

	apiv1 "k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/status"
	utillog "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/log"
)

// pokeConfiSourceWorker tells the worker thread that syncs config sources that work needs to be done
func (cc *Controller) pokeConfigSourceWorker() {
	select {
	case cc.pendingConfigSource <- true:
	default:
	}
}

// syncConfigSource checks if work needs to be done to use a new configuration, and does that work if necessary
func (cc *Controller) syncConfigSource(client clientset.Interface, nodeName string) {
	select {
	case <-cc.pendingConfigSource:
	default:
		// no work to be done, return
		return
	}

	// if the sync fails, we want to retry
	var syncerr error
	defer func() {
		if syncerr != nil {
			utillog.Errorf(syncerr.Error())
			cc.pokeConfigSourceWorker()
		}
	}()

	node, err := latestNode(cc.informer.GetStore(), nodeName)
	if err != nil {
		cc.configOK.SetFailSyncCondition(status.FailSyncReasonInformer)
		syncerr = fmt.Errorf("%s, error: %v", status.FailSyncReasonInformer, err)
		return
	}

	// check the Node and download any new config
	if updated, reason, err := cc.doSyncConfigSource(client, node.Spec.ConfigSource); err != nil {
		cc.configOK.SetFailSyncCondition(reason)
		syncerr = fmt.Errorf("%s, error: %v", reason, err)
		return
	} else if updated {
		// TODO(mtaufen): Consider adding a "currently restarting kubelet" ConfigOK message for this case
		utillog.Infof("config updated, Kubelet will restart to begin using new config")
		os.Exit(0)
	}

	// If we get here:
	// - there is no need to restart to update the current config
	// - there was no error trying to sync configuration
	// - if, previously, there was an error trying to sync configuration, we need to clear that error from the condition
	cc.configOK.ClearFailSyncCondition()
}

// doSyncConfigSource checkpoints and sets the store's current config to the new config or resets config,
// depending on the `source`, and returns whether the current config in the checkpoint store was updated as a result
func (cc *Controller) doSyncConfigSource(client clientset.Interface, source *apiv1.NodeConfigSource) (bool, string, error) {
	if source == nil {
		utillog.Infof("Node.Spec.ConfigSource is empty, will reset current and last-known-good to defaults")
		updated, reason, err := cc.resetConfig()
		if err != nil {
			return false, reason, err
		}
		return updated, "", nil
	}

	// if the NodeConfigSource is non-nil, download the config
	utillog.Infof("Node.Spec.ConfigSource is non-empty, will checkpoint source and update config if necessary")
	remote, reason, err := checkpoint.NewRemoteConfigSource(source)
	if err != nil {
		return false, reason, err
	}
	reason, err = cc.checkpointConfigSource(client, remote)
	if err != nil {
		return false, reason, err
	}
	updated, reason, err := cc.setCurrentConfig(remote)
	if err != nil {
		return false, reason, err
	}
	return updated, "", nil
}

// checkpointConfigSource downloads and checkpoints the object referred to by `source` if the checkpoint does not already exist,
// if a failure occurs, returns a sanitized failure reason and an error
func (cc *Controller) checkpointConfigSource(client clientset.Interface, source checkpoint.RemoteConfigSource) (string, error) {
	uid := source.UID()

	// if the checkpoint already exists, skip downloading
	if ok, err := cc.checkpointStore.Exists(uid); err != nil {
		reason := fmt.Sprintf(status.FailSyncReasonCheckpointExistenceFmt, uid)
		return reason, fmt.Errorf("%s, error: %v", reason, err)
	} else if ok {
		utillog.Infof("checkpoint already exists for object with UID %q, skipping download", uid)
		return "", nil
	}

	// download
	checkpoint, reason, err := source.Download(client)
	if err != nil {
		return reason, fmt.Errorf("%s, error: %v", reason, err)
	}

	// save
	err = cc.checkpointStore.Save(checkpoint)
	if err != nil {
		reason := fmt.Sprintf(status.FailSyncReasonSaveCheckpointFmt, checkpoint.UID())
		return reason, fmt.Errorf("%s, error: %v", reason, err)
	}

	return "", nil
}

// setCurrentConfig updates UID of the current checkpoint in the checkpoint store to `uid` and returns whether the
// current UID changed as a result, or a sanitized failure reason and an error.
func (cc *Controller) setCurrentConfig(source checkpoint.RemoteConfigSource) (bool, string, error) {
	updated, err := cc.checkpointStore.SetCurrentUpdated(source)
	if err != nil {
		if source == nil {
			return false, status.FailSyncReasonSetCurrentDefault, err
		}
		return false, fmt.Sprintf(status.FailSyncReasonSetCurrentUIDFmt, source.UID()), err
	}
	return updated, "", nil
}

// resetConfig resets the current and last-known-good checkpoints in the checkpoint store to their default values and
// returns whether the current checkpoint changed as a result, or a sanitized failure reason and an error.
func (cc *Controller) resetConfig() (bool, string, error) {
	updated, err := cc.checkpointStore.Reset()
	if err != nil {
		return false, status.FailSyncReasonReset, err
	}
	return updated, "", nil
}

// latestNode returns the most recent Node with `nodeName` from `store`
func latestNode(store cache.Store, nodeName string) (*apiv1.Node, error) {
	obj, ok, err := store.GetByKey(nodeName)
	if err != nil {
		err := fmt.Errorf("failed to retrieve Node %q from informer's store, error: %v", nodeName, err)
		utillog.Errorf(err.Error())
		return nil, err
	} else if !ok {
		err := fmt.Errorf("Node %q does not exist in the informer's store, can't sync config source", nodeName)
		utillog.Errorf(err.Error())
		return nil, err
	}
	node, ok := obj.(*apiv1.Node)
	if !ok {
		err := fmt.Errorf("failed to cast object from informer's store to Node, can't sync config source for Node %q", nodeName)
		utillog.Errorf(err.Error())
		return nil, err
	}
	return node, nil
}
