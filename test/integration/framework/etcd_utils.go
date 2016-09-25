/*
Copyright 2014 The Kubernetes Authors.

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

package framework

import (
	"fmt"
	"math/rand"

	etcd "github.com/coreos/etcd/client"
	"github.com/golang/glog"
	"golang.org/x/net/context"

	"k8s.io/kubernetes/pkg/util/env"
)

// If you need to start an etcd instance by hand, you also need to insert a key
// for this check to pass (*any* key will do, eg:
//curl -L http://127.0.0.1:2379/v2/keys/message -XPUT -d value="Hello world").

var testing_etcd = false

func GetEtcdURLFromEnv() string {
	url := env.GetEnvAsStringOrFallback("KUBE_INTEGRATION_ETCD_URL", "http://127.0.0.1:2379")
	glog.V(4).Infof("Using KUBE_INTEGRATION_ETCD_URL=%q", url)
	return url
}

func NewEtcdClient() etcd.Client {
	// gaurded to avoid infinite recursion, check etcd.
	if testing_etcd {
		RequireEtcd()
	}
	cfg := etcd.Config{
		Endpoints: []string{GetEtcdURLFromEnv()},
	}
	client, err := etcd.New(cfg)
	if err != nil {
		glog.Fatalf("unable to connect to etcd for testing: %v", err)
	}
	return client
}

func RequireEtcd() {
	testing_etcd = true
	defer func() {
		testing_etcd = false
	}()
	if _, err := etcd.NewKeysAPI(NewEtcdClient()).Get(context.TODO(), "/", nil); err != nil {
		glog.Fatalf("unable to connect to etcd for testing: %v", err)
	}

}

func WithEtcdKey(f func(string)) {
	prefix := fmt.Sprintf("/test-%d", rand.Int63())
	defer etcd.NewKeysAPI(NewEtcdClient()).Delete(context.TODO(), prefix, &etcd.DeleteOptions{Recursive: true})
	f(prefix)
}
