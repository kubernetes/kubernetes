/*
Copyright 2015 The Kubernetes Authors.

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

package etcd

import (
	"fmt"
	"time"

	etcd "github.com/coreos/etcd/client"
	"golang.org/x/net/context"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/components/framework/frameworkid"
	etcdutil "k8s.io/kubernetes/pkg/storage/etcd/util"
)

type storage struct {
	frameworkid.LookupFunc
	frameworkid.StoreFunc
	frameworkid.RemoveFunc
}

func Store(api etcd.KeysAPI, path string, ttl time.Duration) frameworkid.Storage {
	// TODO(jdef) validate Config
	return &storage{
		LookupFunc: func(ctx context.Context) (string, error) {
			if response, err := api.Get(ctx, path, nil); err != nil {
				if !etcdutil.IsEtcdNotFound(err) {
					return "", fmt.Errorf("unexpected failure attempting to load framework ID from etcd: %v", err)
				}
			} else {
				return response.Node.Value, nil
			}
			return "", nil
		},
		RemoveFunc: func(ctx context.Context) (err error) {
			if _, err = api.Delete(ctx, path, &etcd.DeleteOptions{Recursive: true}); err != nil {
				if !etcdutil.IsEtcdNotFound(err) {
					return fmt.Errorf("failed to delete framework ID from etcd: %v", err)
				}
			}
			return
		},
		StoreFunc: func(ctx context.Context, id string) (err error) {
			_, err = api.Set(ctx, path, id, &etcd.SetOptions{TTL: ttl})
			return
		},
	}
}
