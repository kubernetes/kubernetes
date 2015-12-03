/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	etcdutil "k8s.io/kubernetes/pkg/storage/etcd/util"
)

// IsNotFound returns true if and only if err is "key" not found error.
func IsNotFound(err error) bool {
	// TODO: add alternate storage error here
	return etcdutil.IsEtcdNotFound(err)
}

// IsNodeExist returns true if and only if err is an node already exist error.
func IsNodeExist(err error) bool {
	// TODO: add alternate storage error here
	return etcdutil.IsEtcdNodeExist(err)
}

// IsUnreachable returns true if and only if err indicates the server could not be reached.
func IsUnreachable(err error) bool {
	// TODO: add alternate storage error here
	return etcdutil.IsEtcdUnreachable(err)
}

// IsTestFailed returns true if and only if err is a write conflict.
func IsTestFailed(err error) bool {
	// TODO: add alternate storage error here
	return etcdutil.IsEtcdTestFailed(err)
}
