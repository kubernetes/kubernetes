/*
Copyright 2019 The Kubernetes Authors.

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

package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"path"
	"time"

	"go.etcd.io/etcd/client/pkg/v3/transport"
	clientv3 "go.etcd.io/etcd/client/v3"
	"google.golang.org/grpc"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

// EtcdObjectReader provides direct access to custom resource objects stored in etcd.
type EtcdObjectReader struct {
	etcdClient    *clientv3.Client
	storagePrefix string
	crd           *apiextensionsv1.CustomResourceDefinition
}

// NewEtcdObjectReader creates a reader for accessing custom resource objects directly from etcd.
func NewEtcdObjectReader(etcdClient *clientv3.Client, restOptions *generic.RESTOptions, crd *apiextensionsv1.CustomResourceDefinition) *EtcdObjectReader {
	return &EtcdObjectReader{etcdClient, restOptions.StorageConfig.Prefix, crd}
}

// WaitForStorageVersion calls the updateObjFn periodically and waits for the version of the custom resource stored in etcd to be set to the provided version.
// Typically updateObjFn should perform a noop update to the object so that when stored version of a CRD changes, the object is written at the updated storage version.
// If the timeout is exceeded a error is returned.
// This is useful when updating the stored version of an existing CRD because the update does not take effect immediately.
func (s *EtcdObjectReader) WaitForStorageVersion(version string, ns, name string, timeout time.Duration, updateObjFn func()) error {
	waitCh := time.After(timeout)
	for {
		storage, err := s.GetStoredCustomResource(ns, name)
		if err != nil {
			return err
		}
		if storage.GetObjectKind().GroupVersionKind().Version == version {
			return nil
		}
		select {
		case <-waitCh:
			return fmt.Errorf("timed out after %v waiting for storage version to be %s for object (namespace:%s name:%s)", timeout, version, ns, name)
		case <-time.After(10 * time.Millisecond):
			updateObjFn()
		}
	}
}

// GetStoredCustomResource gets the storage representation of a custom resource from etcd.
func (s *EtcdObjectReader) GetStoredCustomResource(ns, name string) (*unstructured.Unstructured, error) {
	key := path.Join("/", s.storagePrefix, s.crd.Spec.Group, s.crd.Spec.Names.Plural, ns, name)
	resp, err := s.etcdClient.KV.Get(context.Background(), key)
	if err != nil {
		return nil, fmt.Errorf("error getting storage object %s, %s from etcd at key %s: %v", ns, name, key, err)
	}
	if len(resp.Kvs) == 0 {
		return nil, fmt.Errorf("no storage object found for %s, %s in etcd for key %s", ns, name, key)
	}
	raw := resp.Kvs[0].Value
	u := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := json.Unmarshal(raw, u); err != nil {
		return nil, fmt.Errorf("error deserializing object %s: %v", string(raw), err)
	}
	return u, nil
}

// SetStoredCustomResource writes the storage representation of a custom resource to etcd.
func (s *EtcdObjectReader) SetStoredCustomResource(ns, name string, obj *unstructured.Unstructured) error {
	bs, err := obj.MarshalJSON()
	if err != nil {
		return err
	}

	key := path.Join("/", s.storagePrefix, s.crd.Spec.Group, s.crd.Spec.Names.Plural, ns, name)
	if _, err := s.etcdClient.KV.Put(context.Background(), key, string(bs)); err != nil {
		return fmt.Errorf("error setting storage object %s, %s from etcd at key %s: %v", ns, name, key, err)
	}
	return nil
}

// GetEtcdClients returns an initialized  clientv3.Client and clientv3.KV.
func GetEtcdClients(config storagebackend.TransportConfig) (*clientv3.Client, clientv3.KV, error) {
	tlsInfo := transport.TLSInfo{
		CertFile:      config.CertFile,
		KeyFile:       config.KeyFile,
		TrustedCAFile: config.TrustedCAFile,
	}

	tlsConfig, err := tlsInfo.ClientConfig()
	if err != nil {
		return nil, nil, err
	}

	cfg := clientv3.Config{
		Endpoints:   config.ServerList,
		DialTimeout: 20 * time.Second,
		DialOptions: []grpc.DialOption{
			grpc.WithBlock(), // block until the underlying connection is up
		},
		TLS: tlsConfig,
	}

	c, err := clientv3.New(cfg)
	if err != nil {
		return nil, nil, err
	}

	return c, clientv3.NewKV(c), nil
}
