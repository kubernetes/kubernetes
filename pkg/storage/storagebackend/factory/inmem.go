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
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/storage/storagebackend"

	"k8s.io/kubernetes/pkg/storage/inmem"
	"github.com/golang/glog"
)

func newInmemStorage(c storagebackend.Config) (storage.Interface, DestroyFunc, error) {
	//store, err := inmem.NewStore(c.Codec)
	//if err != nil {
	//	return nil, nil, err
	//}
	//ctx, cancel := context.WithCancel(context.Background())
	//inmem.StartCompactor(ctx, client)
	destroyFunc := func() {
		// TODO: what is the behaviour of this currently?
		glog.Infof("inmem destroy function called")
	}
	return inmem.NewStore(c.Codec), destroyFunc, nil
}
