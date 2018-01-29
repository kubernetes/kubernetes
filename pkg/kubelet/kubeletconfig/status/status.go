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

package status

import (
	"fmt"
	"sync"
	"time"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kuberuntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	utilequal "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/equal"
	utillog "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/log"
)

// TODO(mtaufen): s/current/assigned, as this is more accurate e.g. if you are using lkg, you aren't currently using "current" :)
const (
	// CurLocalMessage indicates that the Kubelet is using its local config, which consists of defaults, flags, and/or local files
	CurLocalMessage = "using current: local"
	// LkgLocalMessage indicates that the Kubelet is using its local config, which consists of defaults, flags, and/or local files
	LkgLocalMessage = "using last-known-good: local"

	// CurRemoteMessageFmt indicates the Kubelet is using its current config, which is from an API source
	CurRemoteMessageFmt = "using current: %s"
	// LkgRemoteMessageFmt indicates the Kubelet is using its last-known-good config, which is from an API source
	LkgRemoteMessageFmt = "using last-known-good: %s"

	// CurLocalOkayReason indicates that the Kubelet is using its local config
	CurLocalOkayReason = "when the config source is nil, the Kubelet uses its local config"
	// CurRemoteOkayReason indicates that the config referenced by Node.ConfigSource is currently passing all checks
	CurRemoteOkayReason = "passing all checks"

	// CurFailLoadReasonFmt indicates that the Kubelet failed to load the current config checkpoint for an API source
	CurFailLoadReasonFmt = "failed to load current: %s"
	// CurFailValidateReasonFmt indicates that the Kubelet failed to validate the current config checkpoint for an API source
	CurFailValidateReasonFmt = "failed to validate current: %s"

	// LkgFail*ReasonFmt reasons are currently used to print errors in the Kubelet log, but do not appear in Node.Status.Conditions

	// LkgFailLoadReasonFmt indicates that the Kubelet failed to load the last-known-good config checkpoint for an API source
	LkgFailLoadReasonFmt = "failed to load last-known-good: %s"
	// LkgFailValidateReasonFmt indicates that the Kubelet failed to validate the last-known-good config checkpoint for an API source
	LkgFailValidateReasonFmt = "failed to validate last-known-good: %s"

	// FailSyncReasonFmt is used when the system couldn't sync the config, due to a malformed Node.Spec.ConfigSource, a download failure, etc.
	FailSyncReasonFmt = "failed to sync, reason: %s"
	// FailSyncReasonAllNilSubfields is used when no subfields are set
	FailSyncReasonAllNilSubfields = "invalid NodeConfigSource, exactly one subfield must be non-nil, but all were nil"
	// FailSyncReasonPartialConfigMapSource is used when some required subfields remain unset
	FailSyncReasonPartialConfigMapSource = "invalid ConfigSource.ConfigMap, all of UID, Name, Namespace, and KubeletConfigKey must be specified"
	// FailSyncReasonUIDMismatchFmt is used when there is a UID mismatch between the referenced and downloaded ConfigMaps,
	// this can happen because objects must be downloaded by namespace/name, rather than by UID
	FailSyncReasonUIDMismatchFmt = "invalid ConfigSource.ConfigMap.UID: %s does not match %s.UID: %s"
	// FailSyncReasonDownloadFmt is used when the download fails, e.g. due to network issues
	FailSyncReasonDownloadFmt = "failed to download: %s"
	// FailSyncReasonInformer is used when the informer fails to report the Node object
	FailSyncReasonInformer = "failed to read Node from informer object cache"
	// FailSyncReasonReset is used when we can't reset the local configuration references, e.g. due to filesystem issues
	FailSyncReasonReset = "failed to reset to local config"
	// FailSyncReasonCheckpointExistenceFmt is used when we can't determine if a checkpoint already exists, e.g. due to filesystem issues
	FailSyncReasonCheckpointExistenceFmt = "failed to determine whether object %s with UID %s was already checkpointed"
	// FailSyncReasonSaveCheckpointFmt is used when we can't save a checkpoint, e.g. due to filesystem issues
	FailSyncReasonSaveCheckpointFmt = "failed to save config checkpoint for object %s with UID %s"
	// FailSyncReasonSetCurrentDefault is used when we can't set the current config checkpoint to the local default, e.g. due to filesystem issues
	FailSyncReasonSetCurrentLocal = "failed to set current config checkpoint to local config"
	// FailSyncReasonSetCurrentUIDFmt is used when we can't set the current config checkpoint to a checkpointed object, e.g. due to filesystem issues
	FailSyncReasonSetCurrentUIDFmt = "failed to set current config checkpoint to object %s with UID %s"

	// EmptyMessage is a placeholder in the case that we accidentally set the condition's message to the empty string.
	// Doing so can result in a partial patch, and thus a confusing status; this makes it clear that the message was not provided.
	EmptyMessage = "unknown - message not provided"
	// EmptyReason is a placeholder in the case that we accidentally set the condition's reason to the empty string.
	// Doing so can result in a partial patch, and thus a confusing status; this makes it clear that the reason was not provided.
	EmptyReason = "unknown - reason not provided"
)

// ConfigOkCondition represents a ConfigOk NodeCondition
type ConfigOkCondition interface {
	// Set sets the Message, Reason, and Status of the condition
	Set(message, reason string, status apiv1.ConditionStatus)
	// SetFailSyncCondition sets the condition for when syncing Kubelet config fails
	SetFailSyncCondition(reason string)
	// ClearFailSyncCondition clears the overlay from SetFailSyncCondition
	ClearFailSyncCondition()
	// Sync patches the current condition into the Node identified by `nodeName`
	Sync(client clientset.Interface, nodeName string)
}

// configOkCondition implements ConfigOkCondition
type configOkCondition struct {
	// conditionMux is a mutex on the condition, alternate between setting and syncing the condition
	conditionMux sync.Mutex
	// message will appear as the condition's message
	message string
	// reason will appear as the condition's reason
	reason string
	// status will appear as the condition's status
	status apiv1.ConditionStatus
	// failedSyncReason is sent in place of the usual reason when the Kubelet is failing to sync the remote config
	failedSyncReason string
	// pendingCondition; write to this channel to indicate that ConfigOk needs to be synced to the API server
	pendingCondition chan bool
}

// NewConfigOkCondition returns a new ConfigOkCondition
func NewConfigOkCondition() ConfigOkCondition {
	return &configOkCondition{
		message: EmptyMessage,
		reason:  EmptyReason,
		status:  apiv1.ConditionUnknown,
		// channels must have capacity at least 1, since we signal with non-blocking writes
		pendingCondition: make(chan bool, 1),
	}
}

// unsafeSet sets the current state of the condition
// it does not grab the conditionMux lock, so you should generally use setConfigOk unless you need to grab the lock
// at a higher level to synchronize additional operations
func (c *configOkCondition) unsafeSet(message, reason string, status apiv1.ConditionStatus) {
	// We avoid an empty Message, Reason, or Status on the condition. Since we use Patch to update conditions, an empty
	// field might cause a value from a previous condition to leak through, which can be very confusing.

	// message
	if len(message) == 0 {
		message = EmptyMessage
	}
	c.message = message
	// reason
	if len(reason) == 0 {
		reason = EmptyReason
	}
	c.reason = reason
	// status
	if len(string(status)) == 0 {
		status = apiv1.ConditionUnknown
	}
	c.status = status
	// always poke worker after update
	c.pokeSyncWorker()
}

func (c *configOkCondition) Set(message, reason string, status apiv1.ConditionStatus) {
	c.conditionMux.Lock()
	defer c.conditionMux.Unlock()
	c.unsafeSet(message, reason, status)
}

// SetFailSyncCondition updates the ConfigOk status to reflect that we failed to sync to the latest config,
// e.g. due to a malformed Node.Spec.ConfigSource, a download failure, etc.
func (c *configOkCondition) SetFailSyncCondition(reason string) {
	c.conditionMux.Lock()
	defer c.conditionMux.Unlock()
	// set the reason overlay and poke the sync worker to send the update
	c.failedSyncReason = fmt.Sprintf(FailSyncReasonFmt, reason)
	c.pokeSyncWorker()
}

// ClearFailSyncCondition removes the "failed to sync" reason overlay
func (c *configOkCondition) ClearFailSyncCondition() {
	c.conditionMux.Lock()
	defer c.conditionMux.Unlock()
	// clear the reason overlay and poke the sync worker to send the update
	c.failedSyncReason = ""
	c.pokeSyncWorker()
}

// pokeSyncWorker notes that the ConfigOk condition needs to be synced to the API server
func (c *configOkCondition) pokeSyncWorker() {
	select {
	case c.pendingCondition <- true:
	default:
	}
}

// Sync attempts to sync `c.condition` with the Node object for this Kubelet,
// if syncing fails, an error is logged, and work is queued for retry.
func (c *configOkCondition) Sync(client clientset.Interface, nodeName string) {
	select {
	case <-c.pendingCondition:
	default:
		// no work to be done, return
		return
	}

	// grab the lock
	c.conditionMux.Lock()
	defer c.conditionMux.Unlock()

	// if the sync fails, we want to retry
	var err error
	defer func() {
		if err != nil {
			utillog.Errorf(err.Error())
			c.pokeSyncWorker()
		}
	}()

	// get the Node so we can check the current condition
	node, err := client.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
	if err != nil {
		err = fmt.Errorf("could not get Node %q, will not sync ConfigOk condition, error: %v", nodeName, err)
		return
	}

	// construct the node condition
	condition := &apiv1.NodeCondition{
		Type:    apiv1.NodeKubeletConfigOk,
		Message: c.message,
		Reason:  c.reason,
		Status:  c.status,
	}

	// overlay failed sync reason, if necessary
	if len(c.failedSyncReason) > 0 {
		condition.Reason = c.failedSyncReason
		condition.Status = apiv1.ConditionFalse
	}

	// set timestamps
	syncTime := metav1.NewTime(time.Now())
	condition.LastHeartbeatTime = syncTime
	if remote := getKubeletConfigOk(node.Status.Conditions); remote == nil || !utilequal.KubeletConfigOkEq(remote, condition) {
		// update transition time the first time we create the condition,
		// or if we are semantically changing the condition
		condition.LastTransitionTime = syncTime
	} else {
		// since the conditions are semantically equal, use lastTransitionTime from the condition currently on the Node
		// we need to do this because the field will always be represented in the patch generated below, and this copy
		// prevents nullifying the field during the patch operation
		condition.LastTransitionTime = remote.LastTransitionTime
	}

	// generate the patch
	mediaType := "application/json"
	info, ok := kuberuntime.SerializerInfoForMediaType(legacyscheme.Codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		err = fmt.Errorf("unsupported media type %q", mediaType)
		return
	}
	versions := legacyscheme.Registry.RegisteredVersionsForGroup(api.GroupName)
	if len(versions) == 0 {
		err = fmt.Errorf("no enabled versions for group %q", api.GroupName)
		return
	}
	// the "best" version supposedly comes first in the list returned from apiv1.Registry.EnabledVersionsForGroup
	encoder := legacyscheme.Codecs.EncoderForVersion(info.Serializer, versions[0])

	before, err := kuberuntime.Encode(encoder, node)
	if err != nil {
		err = fmt.Errorf(`failed to encode "before" node while generating patch, error: %v`, err)
		return
	}

	patchConfigOk(node, condition)
	after, err := kuberuntime.Encode(encoder, node)
	if err != nil {
		err = fmt.Errorf(`failed to encode "after" node while generating patch, error: %v`, err)
		return
	}

	patch, err := strategicpatch.CreateTwoWayMergePatch(before, after, apiv1.Node{})
	if err != nil {
		err = fmt.Errorf("failed to generate patch for updating ConfigOk condition, error: %v", err)
		return
	}

	// patch the remote Node object
	_, err = client.CoreV1().Nodes().PatchStatus(nodeName, patch)
	if err != nil {
		err = fmt.Errorf("could not update ConfigOk condition, error: %v", err)
		return
	}
}

// patchConfigOk replaces or adds the ConfigOk condition to the node
func patchConfigOk(node *apiv1.Node, configOk *apiv1.NodeCondition) {
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == apiv1.NodeKubeletConfigOk {
			// edit the condition
			node.Status.Conditions[i] = *configOk
			return
		}
	}
	// append the condition
	node.Status.Conditions = append(node.Status.Conditions, *configOk)
}

// getKubeletConfigOk returns the first NodeCondition in `cs` with Type == apiv1.NodeKubeletConfigOk,
// or if no such condition exists, returns nil.
func getKubeletConfigOk(cs []apiv1.NodeCondition) *apiv1.NodeCondition {
	for i := range cs {
		if cs[i].Type == apiv1.NodeKubeletConfigOk {
			return &cs[i]
		}
	}
	return nil
}
