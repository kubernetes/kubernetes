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
	"time"

	"github.com/golang/glog"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/status"
	utillog "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/log"
)

const (
	// KubeletConfigChangedEventReason identifies an event as a change of Kubelet configuration
	KubeletConfigChangedEventReason = "KubeletConfigChanged"
	// LocalEventMessage is sent when the Kubelet restarts to use local config
	LocalEventMessage = "Kubelet restarting to use local config"
	// RemoteEventMessageFmt is sent when the Kubelet restarts to use a remote config
	RemoteEventMessageFmt = "Kubelet restarting to use %s, UID: %s, ResourceVersion: %s, KubeletConfigKey: %s"
)

// pokeConfiSourceWorker tells the worker thread that syncs config sources that work needs to be done
func (cc *Controller) pokeConfigSourceWorker() {
	select {
	case cc.pendingConfigSource <- true:
	default:
	}
}

// syncConfigSource checks if work needs to be done to use a new configuration, and does that work if necessary
func (cc *Controller) syncConfigSource(client clientset.Interface, eventClient v1core.EventsGetter, nodeName string) {
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

	// get the latest Node.Spec.ConfigSource from the informer
	source, err := latestNodeConfigSource(cc.nodeInformer.GetStore(), nodeName)
	if err != nil {
		cc.configStatus.SetErrorOverride(fmt.Sprintf(status.SyncErrorFmt, status.InternalError))
		syncerr = fmt.Errorf("%s, error: %v", status.InternalError, err)
		return
	}

	// a nil source simply means we reset to local defaults
	if source == nil {
		utillog.Infof("Node.Spec.ConfigSource is empty, will reset assigned and last-known-good to defaults")
		if updated, reason, err := cc.resetConfig(); err != nil {
			reason = fmt.Sprintf(status.SyncErrorFmt, reason)
			cc.configStatus.SetErrorOverride(reason)
			syncerr = fmt.Errorf("%s, error: %v", reason, err)
			return
		} else if updated {
			restartForNewConfig(eventClient, nodeName, nil)
		}
		return
	}

	// a non-nil source means we should attempt to download the config, and checkpoint it if necessary
	utillog.Infof("Node.Spec.ConfigSource is non-empty, will checkpoint source and update config if necessary")

	// TODO(mtaufen): It would be nice if we could check the payload's metadata before (re)downloading the whole payload
	//                we at least try pulling the latest configmap out of the local informer store.

	// construct the interface that can dynamically dispatch the correct Download, etc. methods for the given source type
	remote, reason, err := checkpoint.NewRemoteConfigSource(source)
	if err != nil {
		reason = fmt.Sprintf(status.SyncErrorFmt, reason)
		cc.configStatus.SetErrorOverride(reason)
		syncerr = fmt.Errorf("%s, error: %v", reason, err)
		return
	}

	// "download" source, either from informer's in-memory store or directly from the API server, if the informer doesn't have a copy
	payload, reason, err := cc.downloadConfigPayload(client, remote)
	if err != nil {
		reason = fmt.Sprintf(status.SyncErrorFmt, reason)
		cc.configStatus.SetErrorOverride(reason)
		syncerr = fmt.Errorf("%s, error: %v", reason, err)
		return
	}

	// save a checkpoint for the payload, if one does not already exist
	if reason, err := cc.saveConfigCheckpoint(remote, payload); err != nil {
		reason = fmt.Sprintf(status.SyncErrorFmt, reason)
		cc.configStatus.SetErrorOverride(reason)
		syncerr = fmt.Errorf("%s, error: %v", reason, err)
		return
	}

	// update the local, persistent record of assigned config
	if updated, reason, err := cc.setAssignedConfig(remote); err != nil {
		reason = fmt.Sprintf(status.SyncErrorFmt, reason)
		cc.configStatus.SetErrorOverride(reason)
		syncerr = fmt.Errorf("%s, error: %v", reason, err)
		return
	} else if updated {
		restartForNewConfig(eventClient, nodeName, remote)
	}

	// If we get here:
	// - there is no need to restart to use new config
	// - there was no error trying to sync configuration
	// - if, previously, there was an error trying to sync configuration, we need to clear that error from the status
	cc.configStatus.SetErrorOverride("")
}

// Note: source has up-to-date uid and resourceVersion after calling downloadConfigPayload.
func (cc *Controller) downloadConfigPayload(client clientset.Interface, source checkpoint.RemoteConfigSource) (checkpoint.Payload, string, error) {
	var store cache.Store
	if cc.remoteConfigSourceInformer != nil {
		store = cc.remoteConfigSourceInformer.GetStore()
	}
	return source.Download(client, store)
}

func (cc *Controller) saveConfigCheckpoint(source checkpoint.RemoteConfigSource, payload checkpoint.Payload) (string, error) {
	ok, err := cc.checkpointStore.Exists(source)
	if err != nil {
		return status.InternalError, fmt.Errorf("%s, error: %v", status.InternalError, err)
	}
	if ok {
		utillog.Infof("checkpoint already exists for %s, UID: %s, ResourceVersion: %s", source.APIPath(), payload.UID(), payload.ResourceVersion())
		return "", nil
	}
	if err := cc.checkpointStore.Save(payload); err != nil {
		return status.InternalError, fmt.Errorf("%s, error: %v", status.InternalError, err)
	}
	return "", nil
}

// setAssignedConfig updates the assigned checkpoint config in the store.
// Returns whether the assigned config changed as a result, or a sanitized failure reason and an error.
func (cc *Controller) setAssignedConfig(source checkpoint.RemoteConfigSource) (bool, string, error) {
	assigned, err := cc.checkpointStore.Assigned()
	if err != nil {
		return false, status.InternalError, err
	}
	if err := cc.checkpointStore.SetAssigned(source); err != nil {
		return false, status.InternalError, err
	}
	return !checkpoint.EqualRemoteConfigSources(assigned, source), "", nil
}

// resetConfig resets the assigned and last-known-good checkpoints in the checkpoint store to their default values and
// returns whether the assigned checkpoint changed as a result, or a sanitized failure reason and an error.
func (cc *Controller) resetConfig() (bool, string, error) {
	updated, err := cc.checkpointStore.Reset()
	if err != nil {
		return false, status.InternalError, err
	}
	return updated, "", nil
}

// restartForNewConfig presumes the Kubelet is managed by a babysitter, e.g. systemd
// It will send an event before exiting.
func restartForNewConfig(eventClient v1core.EventsGetter, nodeName string, source checkpoint.RemoteConfigSource) {
	message := LocalEventMessage
	if source != nil {
		message = fmt.Sprintf(RemoteEventMessageFmt, source.APIPath(), source.UID(), source.ResourceVersion(), source.KubeletFilename())
	}
	// we directly log and send the event, instead of using the event recorder,
	// because the event recorder won't flush its queue before we exit (we'd lose the event)
	event := makeEvent(nodeName, apiv1.EventTypeNormal, KubeletConfigChangedEventReason, message)
	glog.V(3).Infof("Event(%#v): type: '%v' reason: '%v' %v", event.InvolvedObject, event.Type, event.Reason, event.Message)
	if _, err := eventClient.Events(apiv1.NamespaceDefault).Create(event); err != nil {
		utillog.Errorf("failed to send event, error: %v", err)
	}
	utillog.Infof(message)
	os.Exit(0)
}

// latestNodeConfigSource returns a copy of the most recent NodeConfigSource from the Node with `nodeName` in `store`
func latestNodeConfigSource(store cache.Store, nodeName string) (*apiv1.NodeConfigSource, error) {
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
	// Copy the source, so anyone who modifies it after here doesn't mess up the informer's store!
	// This was previously the cause of a bug that made the Kubelet frequently resync config; Download updated
	// the UID and ResourceVersion on the NodeConfigSource, but the pointer was still drilling all the way
	// into the informer's copy!
	return node.Spec.ConfigSource.DeepCopy(), nil
}

// makeEvent constructs an event
// similar to makeEvent in k8s.io/client-go/tools/record/event.go
func makeEvent(nodeName, eventtype, reason, message string) *apiv1.Event {
	const componentKubelet = "kubelet"
	// NOTE(mtaufen): This is consistent with pkg/kubelet/kubelet.go. Even though setting the node
	// name as the UID looks strange, it appears to be conventional for events sent by the Kubelet.
	ref := apiv1.ObjectReference{
		Kind:      "Node",
		Name:      nodeName,
		UID:       types.UID(nodeName),
		Namespace: "",
	}

	t := metav1.Time{Time: time.Now()}
	namespace := ref.Namespace
	if namespace == "" {
		namespace = metav1.NamespaceDefault
	}
	return &apiv1.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%v.%x", ref.Name, t.UnixNano()),
			Namespace: namespace,
		},
		InvolvedObject: ref,
		Reason:         reason,
		Message:        message,
		FirstTimestamp: t,
		LastTimestamp:  t,
		Count:          1,
		Type:           eventtype,
		Source:         apiv1.EventSource{Component: componentKubelet, Host: string(nodeName)},
	}
}
