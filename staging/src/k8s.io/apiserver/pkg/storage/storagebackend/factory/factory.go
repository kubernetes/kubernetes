/*
Copyright 2016 The Kubernetes Authors.

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

package factory

import (
	"fmt"

	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

// DestroyFunc is to destroy any resources used by the storage returned in Create() together.
type DestroyFunc func()

// Create creates a storage backend based on given config.
func Create(c storagebackend.Config) (storage.Interface, DestroyFunc, error) {
	switch c.Type {
	// Warning: changing the interpretation of the default (StorageTypeUnset)
	// is a breaking change for users, and is subject to the kubernetes
	// deprecation policy.  Also note that it's probably a bad idea, because
	// then we end up with API servers whose default varies depending on when
	// they were built.  More details in #45457.
	case storagebackend.StorageTypeUnset, storagebackend.StorageTypeETCD2:
		return newETCD2Storage(c)
	case storagebackend.StorageTypeETCD3:
		// TODO: We have the following features to implement:
		// - Support secure connection by using key, cert, and CA files.
		// - Honor "https" scheme to support secure connection in gRPC.
		// - Support non-quorum read.
		return newETCD3Storage(c)
	default:
		return nil, nil, fmt.Errorf("unknown storage type: %s", c.Type)
	}
}
