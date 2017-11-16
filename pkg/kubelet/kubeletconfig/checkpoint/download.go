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

package checkpoint

import (
	"fmt"

	apiv1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/status"
	utilcodec "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/codec"
	utillog "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/log"
)

// RemoteConfigSource represents a remote config source object that can be downloaded as a Checkpoint
type RemoteConfigSource interface {
	// UID returns the UID of the remote config source object
	UID() string
	// Download downloads the remote config source object returns a Checkpoint backed by the object,
	// or a sanitized failure reason and error if the download fails
	Download(client clientset.Interface) (Checkpoint, string, error)
	// Encode returns a []byte representation of the object behind the RemoteConfigSource
	Encode() ([]byte, error)

	// object returns the underlying source object. If you want to compare sources for equality, use EqualRemoteConfigSources,
	// which compares the underlying source objects for semantic API equality.
	object() interface{}
}

// NewRemoteConfigSource constructs a RemoteConfigSource from a v1/NodeConfigSource object, or returns
// a sanitized failure reason and an error if the `source` is blatantly invalid.
// You should only call this with a non-nil config source.
func NewRemoteConfigSource(source *apiv1.NodeConfigSource) (RemoteConfigSource, string, error) {
	// exactly one subfield of the config source must be non-nil, toady ConfigMapRef is the only reference
	if source.ConfigMapRef == nil {
		return nil, status.FailSyncReasonAllNilSubfields, fmt.Errorf("%s, NodeConfigSource was: %#v", status.FailSyncReasonAllNilSubfields, source)
	}

	// validate the NodeConfigSource:

	// at this point we know we're using the ConfigMapRef subfield
	ref := source.ConfigMapRef

	// name, namespace, and UID must all be non-empty for ConfigMapRef
	if ref.Name == "" || ref.Namespace == "" || string(ref.UID) == "" {
		return nil, status.FailSyncReasonPartialObjectReference, fmt.Errorf("%s, ObjectReference was: %#v", status.FailSyncReasonPartialObjectReference, ref)
	}

	return &remoteConfigMap{source}, "", nil
}

// DecodeRemoteConfigSource is a helper for using the apimachinery to decode serialized RemoteConfigSources;
// e.g. the objects stored in the .cur and .lkg files by checkpoint/store/fsstore.go
func DecodeRemoteConfigSource(data []byte) (RemoteConfigSource, error) {
	// decode the remote config source
	obj, err := runtime.Decode(legacyscheme.Codecs.UniversalDecoder(), data)
	if err != nil {
		return nil, fmt.Errorf("failed to decode, error: %v", err)
	}

	// for now we assume we are trying to load an apiv1.NodeConfigSource,
	// this may need to be extended if e.g. a new version of the api is born

	// convert it to the external NodeConfigSource type, so we're consistently working with the external type outside of the on-disk representation
	cs := &apiv1.NodeConfigSource{}
	err = legacyscheme.Scheme.Convert(obj, cs, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to convert decoded object into a v1 NodeConfigSource, error: %v", err)
	}
	source, _, err := NewRemoteConfigSource(cs)
	return source, err
}

// EqualRemoteConfigSources is a helper for comparing remote config sources by
// comparing the underlying API objects for semantic equality.
func EqualRemoteConfigSources(a, b RemoteConfigSource) bool {
	if a != nil && b != nil {
		return apiequality.Semantic.DeepEqual(a.object(), b.object())
	}
	if a == nil && b == nil {
		return true
	}
	return false
}

// remoteConfigMap implements RemoteConfigSource for v1/ConfigMap config sources
type remoteConfigMap struct {
	source *apiv1.NodeConfigSource
}

func (r *remoteConfigMap) UID() string {
	return string(r.source.ConfigMapRef.UID)
}

func (r *remoteConfigMap) Download(client clientset.Interface) (Checkpoint, string, error) {
	var reason string
	uid := string(r.source.ConfigMapRef.UID)

	utillog.Infof("attempting to download ConfigMap with UID %q", uid)

	// get the ConfigMap via namespace/name, there doesn't seem to be a way to get it by UID
	cm, err := client.CoreV1().ConfigMaps(r.source.ConfigMapRef.Namespace).Get(r.source.ConfigMapRef.Name, metav1.GetOptions{})
	if err != nil {
		reason = fmt.Sprintf(status.FailSyncReasonDownloadFmt, r.source.ConfigMapRef.Name, r.source.ConfigMapRef.Namespace)
		return nil, reason, fmt.Errorf("%s, error: %v", reason, err)
	}

	// ensure that UID matches the UID on the reference, the ObjectReference must be unambiguous
	if r.source.ConfigMapRef.UID != cm.UID {
		reason = fmt.Sprintf(status.FailSyncReasonUIDMismatchFmt, r.source.ConfigMapRef.UID, cm.UID)
		return nil, reason, fmt.Errorf(reason)
	}

	checkpoint, err := NewConfigMapCheckpoint(cm)
	if err != nil {
		reason = fmt.Sprintf("invalid downloaded object")
		return nil, reason, fmt.Errorf("%s, error: %v", reason, err)
	}

	utillog.Infof("successfully downloaded ConfigMap with UID %q", uid)
	return checkpoint, "", nil
}

func (r *remoteConfigMap) Encode() ([]byte, error) {
	encoder, err := utilcodec.NewJSONEncoder(apiv1.GroupName)
	if err != nil {
		return nil, err
	}
	data, err := runtime.Encode(encoder, r.source)
	if err != nil {
		return nil, err
	}
	return data, nil
}

func (r *remoteConfigMap) object() interface{} {
	return r.source
}
