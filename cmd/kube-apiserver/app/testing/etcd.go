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

package testing

import (
	"os"
	"path"
	"sync"
	"testing"

	"github.com/pborman/uuid"
	"k8s.io/kubernetes/pkg/api/legacyscheme"

	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

var (
	lock             sync.Mutex
	sharedEtcd       *etcdtesting.EtcdTestServer
	sharedEtcdConfig *storagebackend.Config
)

// SharedInProcessEtcd creates a storage config for a shared in-process etcd server.
// The first call creates the server. It's never terminated.
func SharedInProcessEtcd(t *testing.T) *storagebackend.Config {
	lock.Lock()
	defer lock.Unlock()

	if sharedEtcd == nil {
		sharedEtcd, sharedEtcdConfig = etcdtesting.NewUnsecuredEtcd3TestClientServer(t, legacyscheme.Scheme)
	}
	cfg := *sharedEtcdConfig
	cfg.Prefix = path.Join(cfg.Prefix, uuid.New(), "registry")
	return &cfg
}

// SharedInProcessEtcdMain is test running with a shared in-process etcd server. It is used in a TestMain function
// in order to cleanup the etcd server unix sockets created in the current directory.
//
// Example:
//
//   func TestMain(m *testing.M) {
//     SharedInProcessEtcdMain(m.Run)
//   }
func SharedInProcessEtcdMain(tests func() int) {
	t := &testing.T{}

	os.Exit(func() int {
		if sharedEtcd == nil {
			func() {
				lock.Lock()
				defer lock.Unlock()
				sharedEtcd, sharedEtcdConfig = etcdtesting.NewUnsecuredEtcd3TestClientServer(t, legacyscheme.Scheme)
			}()

			defer sharedEtcd.Terminate(t)
		}

		return tests()
	}())
}
