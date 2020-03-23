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
	"context"
	"fmt"
	"math/rand"
	"time"

	apiv1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"
	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/apis/config/scheme"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/status"
	utilcodec "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/codec"
	utillog "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/log"
)

// Payload represents a local copy of a config source (payload) object
type Payload interface {
	// UID returns a globally unique (space and time) identifier for the payload.
	// The return value is guaranteed non-empty.
	UID() string

	// ResourceVersion returns a resource version for the payload.
	// The return value is guaranteed non-empty.
	ResourceVersion() string

	// Files returns a map of filenames to file contents.
	Files() map[string]string

	// object returns the underlying checkpointed object.
	object() interface{}
}

// RemoteConfigSource represents a remote config source object that can be downloaded as a Checkpoint
type RemoteConfigSource interface {
	// KubeletFilename returns the name of the Kubelet config file as it should appear in the keys of Payload.Files()
	KubeletFilename() string

	// APIPath returns the API path to the remote resource, e.g. its SelfLink
	APIPath() string

	// UID returns the globally unique identifier for the most recently downloaded payload targeted by the source.
	UID() string

	// ResourceVersion returns the resource version of the most recently downloaded payload targeted by the source.
	ResourceVersion() string

	// Download downloads the remote config source's target object and returns a Payload backed by the object,
	// or a sanitized failure reason and error if the download fails.
	// Download takes an optional store as an argument. If provided, Download will check this store for the
	// target object prior to contacting the API server.
	// Download updates the local UID and ResourceVersion tracked by this source, based on the downloaded payload.
	Download(client clientset.Interface, store cache.Store) (Payload, string, error)

	// Informer returns an informer that can be used to detect changes to the remote config source
	Informer(client clientset.Interface, handler cache.ResourceEventHandlerFuncs) cache.SharedInformer

	// Encode returns a []byte representation of the object behind the RemoteConfigSource
	Encode() ([]byte, error)

	// NodeConfigSource returns a copy of the underlying apiv1.NodeConfigSource object.
	// All RemoteConfigSources are expected to be backed by a NodeConfigSource,
	// though the convenience methods on the interface will target the source
	// type that was detected in a call to NewRemoteConfigSource.
	NodeConfigSource() *apiv1.NodeConfigSource
}

// NewRemoteConfigSource constructs a RemoteConfigSource from a v1/NodeConfigSource object
// You should only call this with a non-nil config source.
// Note that the API server validates Node.Spec.ConfigSource.
func NewRemoteConfigSource(source *apiv1.NodeConfigSource) (RemoteConfigSource, string, error) {
	// NOTE: Even though the API server validates the config, we check whether all *known* fields are
	// nil here, so that if a new API server allows a new config source type, old clients can send
	// an error message rather than crashing due to a nil pointer dereference.

	// Exactly one reference subfield of the config source must be non-nil.
	// Currently ConfigMap is the only reference subfield.
	if source.ConfigMap == nil {
		return nil, status.AllNilSubfieldsError, fmt.Errorf("%s, NodeConfigSource was: %#v", status.AllNilSubfieldsError, source)
	}
	return &remoteConfigMap{source}, "", nil
}

// DecodeRemoteConfigSource is a helper for using the apimachinery to decode serialized RemoteConfigSources;
// e.g. the metadata stored by checkpoint/store/fsstore.go
func DecodeRemoteConfigSource(data []byte) (RemoteConfigSource, error) {
	// Decode the remote config source. We want this to be non-strict
	// so we don't error out on newer API keys.
	_, codecs, err := scheme.NewSchemeAndCodecs(serializer.DisableStrict)
	if err != nil {
		return nil, err
	}

	obj, err := runtime.Decode(codecs.UniversalDecoder(), data)
	if err != nil {
		return nil, fmt.Errorf("failed to decode, error: %v", err)
	}

	// for now we assume we are trying to load an kubeletconfigv1beta1.SerializedNodeConfigSource,
	// this may need to be extended if e.g. a new version of the api is born
	cs, ok := obj.(*kubeletconfiginternal.SerializedNodeConfigSource)
	if !ok {
		return nil, fmt.Errorf("failed to cast decoded remote config source to *k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig.SerializedNodeConfigSource")
	}

	// we use the v1.NodeConfigSource type on internal and external, so no need to convert to external here
	source, _, err := NewRemoteConfigSource(&cs.Source)
	if err != nil {
		return nil, err
	}

	return source, nil
}

// EqualRemoteConfigSources is a helper for comparing remote config sources by
// comparing the underlying API objects for semantic equality.
func EqualRemoteConfigSources(a, b RemoteConfigSource) bool {
	if a != nil && b != nil {
		return apiequality.Semantic.DeepEqual(a.NodeConfigSource(), b.NodeConfigSource())
	}
	return a == b
}

// remoteConfigMap implements RemoteConfigSource for v1/ConfigMap config sources
type remoteConfigMap struct {
	source *apiv1.NodeConfigSource
}

var _ RemoteConfigSource = (*remoteConfigMap)(nil)

func (r *remoteConfigMap) KubeletFilename() string {
	return r.source.ConfigMap.KubeletConfigKey
}

const configMapAPIPathFmt = "/api/v1/namespaces/%s/configmaps/%s"

func (r *remoteConfigMap) APIPath() string {
	ref := r.source.ConfigMap
	return fmt.Sprintf(configMapAPIPathFmt, ref.Namespace, ref.Name)
}

func (r *remoteConfigMap) UID() string {
	return string(r.source.ConfigMap.UID)
}

func (r *remoteConfigMap) ResourceVersion() string {
	return r.source.ConfigMap.ResourceVersion
}

func (r *remoteConfigMap) Download(client clientset.Interface, store cache.Store) (Payload, string, error) {
	var (
		cm  *apiv1.ConfigMap
		err error
	)
	// check the in-memory store for the ConfigMap, so we can skip unnecessary downloads
	if store != nil {
		utillog.Infof("checking in-memory store for %s", r.APIPath())
		cm, err = getConfigMapFromStore(store, r.source.ConfigMap.Namespace, r.source.ConfigMap.Name)
		if err != nil {
			// just log the error, we'll attempt a direct download instead
			utillog.Errorf("failed to check in-memory store for %s, error: %v", r.APIPath(), err)
		} else if cm != nil {
			utillog.Infof("found %s in in-memory store, UID: %s, ResourceVersion: %s", r.APIPath(), cm.UID, cm.ResourceVersion)
		} else {
			utillog.Infof("did not find %s in in-memory store", r.APIPath())
		}
	}
	// if we didn't find the ConfigMap in the in-memory store, download it from the API server
	if cm == nil {
		utillog.Infof("attempting to download %s", r.APIPath())
		cm, err = client.CoreV1().ConfigMaps(r.source.ConfigMap.Namespace).Get(context.TODO(), r.source.ConfigMap.Name, metav1.GetOptions{})
		if err != nil {
			return nil, status.DownloadError, fmt.Errorf("%s, error: %v", status.DownloadError, err)
		}
		utillog.Infof("successfully downloaded %s, UID: %s, ResourceVersion: %s", r.APIPath(), cm.UID, cm.ResourceVersion)
	} // Assert: Now we have a non-nil ConfigMap
	// construct Payload from the ConfigMap
	payload, err := NewConfigMapPayload(cm)
	if err != nil {
		// We only expect an error here if ObjectMeta is lacking UID or ResourceVersion. This should
		// never happen on objects in the informer's store, or objects downloaded from the API server
		// directly, so we report InternalError.
		return nil, status.InternalError, fmt.Errorf("%s, error: %v", status.InternalError, err)
	}
	// update internal UID and ResourceVersion based on latest ConfigMap
	r.source.ConfigMap.UID = cm.UID
	r.source.ConfigMap.ResourceVersion = cm.ResourceVersion
	return payload, "", nil
}

func (r *remoteConfigMap) Informer(client clientset.Interface, handler cache.ResourceEventHandlerFuncs) cache.SharedInformer {
	// select ConfigMap by name
	fieldSelector := fields.OneTermEqualSelector("metadata.name", r.source.ConfigMap.Name)

	// add some randomness to resync period, which can help avoid controllers falling into lock-step
	minResyncPeriod := 15 * time.Minute
	factor := rand.Float64() + 1
	resyncPeriod := time.Duration(float64(minResyncPeriod.Nanoseconds()) * factor)

	lw := cache.NewListWatchFromClient(client.CoreV1().RESTClient(), "configmaps", r.source.ConfigMap.Namespace, fieldSelector)

	informer := cache.NewSharedInformer(lw, &apiv1.ConfigMap{}, resyncPeriod)
	informer.AddEventHandler(handler)

	return informer
}

func (r *remoteConfigMap) Encode() ([]byte, error) {
	encoder, err := utilcodec.NewKubeletconfigYAMLEncoder(kubeletconfigv1beta1.SchemeGroupVersion)
	if err != nil {
		return nil, err
	}

	data, err := runtime.Encode(encoder, &kubeletconfigv1beta1.SerializedNodeConfigSource{Source: *r.source})
	if err != nil {
		return nil, err
	}
	return data, nil
}

func (r *remoteConfigMap) NodeConfigSource() *apiv1.NodeConfigSource {
	return r.source.DeepCopy()
}

func getConfigMapFromStore(store cache.Store, namespace, name string) (*apiv1.ConfigMap, error) {
	key := fmt.Sprintf("%s/%s", namespace, name)
	obj, ok, err := store.GetByKey(key)
	if err != nil || !ok {
		return nil, err
	}
	cm, ok := obj.(*apiv1.ConfigMap)
	if !ok {
		err := fmt.Errorf("failed to cast object %s from informer's store to ConfigMap", key)
		utillog.Errorf(err.Error())
		return nil, err
	}
	return cm, nil
}
