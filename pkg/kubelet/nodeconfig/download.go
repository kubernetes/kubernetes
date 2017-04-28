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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
)

// syncNodeConfig downloads and checkpoints the configuration for `node`, if necessary, and also updates the symlinks
// to point to a new configuration if necessary.
// If all operations succeed, returns (bool, nil), where bool indicates whether the current configuration changed. If (true, nil),
// restarting the Kubelet to begin using the new configuration is recommended.
// If downloading fails for a non-fatal reason, an error is returned. See `downloadConfig` and `downloadConfigMap` for fatal reasons.
// If filesystem issues prevent inspecting current configuration or setting symlinks, a fatal error occurs.
// If an error is returned, a cause is also returned in the second position,
// this is a sanitized version of the error that can be reported in the ConfigOK condition.
func (cc *NodeConfigController) syncNodeConfig(node *apiv1.Node) (bool, string, error) {
	// if the NodeConfigSource is non-nil, download the config
	updatedCfg := false
	src := node.Spec.ConfigSource
	if src != nil {
		infof("Node.Spec.ConfigSource is non-empty, will download config if necessary")
		uid, cause, err := cc.downloadConfig(src)
		if err != nil {
			return false, cause, fmt.Errorf("error downloading config, %v", err)
		}
		// if curUID is already correct we can skip updating the symlink
		curUID := cc.curUID()
		if curUID == uid {
			return false, "", nil
		}
		// update curSymlink to point to the new configuration
		cc.setSymlinkUID(curSymlink, uid)
		updatedCfg = true
	} else {
		infof("Node.Spec.ConfigSource is empty, will reset symlinks if necessary")
		// empty config on the node requires both symlinks be reset to the default,
		// we return whether the current configuration changed
		cc.resetSymlink(lkgSymlink)
		updatedCfg = cc.resetSymlink(curSymlink)
	}
	return updatedCfg, "", nil
}

// downloadConfig downloads and checkpoints the configuration source referred to by `src`.
// If downloading and checkpointing succeeds, returns the UID of the checkpointed source.
// If the `src` is invalid, returns an error.
// Otherwise returns errors or fatal errors occur depending on the implementation of the download
// function for the source type used in `src`. Today the only valid source type is a ConfigMap.
// If an error is returned, a cause is also returned in the second position,
// this is a sanitized version of the error that can be reported in the ConfigOK condition.
func (cc *NodeConfigController) downloadConfig(src *apiv1.NodeConfigSource) (string, string, error) {
	if src.ConfigMapRef == nil {
		// exactly one subfield of the config source must be non-nil, toady ConfigMapRef is the only reference
		cause := "invalid NodeConfigSource, exactly one subfield must be non-nil, but all were nil"
		return "", cause, fmt.Errorf("%s", cause)
	}
	return cc.downloadConfigMap(src.ConfigMapRef)
}

// downloadConfigMap downloads and checkpoints the ConfigMap referred to by `ref`
// returns the UID of the downloaded ConfigMap.
// If the checkpoint already exists, skips downloading and returns the UID.
// If the reference is invalid, returns an error.
// If filesystem issues prevent saving the ConfigMap to disk, returns an error.
// If filesystem issues prevent checking whether the checkpoint exists, a fatal error occurs.
// If an error is returned, a cause is also returned in the second position,
// this is a sanitized version of the error that can be reported in the ConfigOK condition.
func (cc *NodeConfigController) downloadConfigMap(ref *apiv1.ObjectReference) (string, string, error) {
	var cause string
	// name, namespace, and UID must all be non-empty
	if ref.Name == "" || ref.Namespace == "" || string(ref.UID) == "" {
		cause = "invalid ObjectReference, all of UID, Name, and Namespace must be specified"
		return "", cause, fmt.Errorf("%s, ObjectReference was: %+v", cause, ref)
	}
	uid := string(ref.UID)

	// if the checkpoint already exists, skip downloading
	if cc.checkpointExists(uid) {
		infof("checkpoint already exists for ConfigMap referred to by %+v, skipping download", ref)
		return uid, "", nil
	}
	infof("attempting to checkpoint ConfigMap with UID %q", uid)

	// get the ConfigMap via namespace/name, there doesn't seem to be a way to get it by UID
	cm, err := cc.client.CoreV1().ConfigMaps(ref.Namespace).Get(ref.Name, metav1.GetOptions{})
	if err != nil {
		cause = fmt.Sprintf("could not download ConfigMap with name %q from namespace %q", ref.Name, ref.Namespace)
		return "", cause, fmt.Errorf("%s, error: %v", cause, err)
	}

	// ensure that UID matches the UID on the reference, the ObjectReference must be unambiguous
	if ref.UID != cm.UID {
		cause = fmt.Sprintf("invalid ObjectReference, UID %q does not match UID of downloaded ConfigMap %q", ref.UID, cm.UID)
		return "", cause, fmt.Errorf("%s", cause)
	}

	// save the ConfigMap to disk
	if err = cc.checkpointConfigMap(cm); err != nil {
		cause = fmt.Sprintf("failed to checkpoint ConfigMap with UID %q", uid)
		return "", cause, fmt.Errorf("%s, error: %v", cause, err)
	}

	infof("successfully checkpointed ConfigMap with UID %q", uid)
	return string(cm.UID), "", nil
}
