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

package objectmeta

import (
	"testing"

	etcd "github.com/coreos/etcd/client"
	"github.com/golang/glog"
	"github.com/stretchr/testify/assert"
	"golang.org/x/net/context"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/genericapiserver"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/storage/etcd/etcdtest"
	"k8s.io/kubernetes/test/integration/framework"
)

// TODO: Eliminate this v2 client dependency.
func newEtcdClient() etcd.Client {
	cfg := etcd.Config{
		Endpoints: []string{framework.GetEtcdURLFromEnv()},
	}
	client, err := etcd.New(cfg)
	if err != nil {
		glog.Fatalf("unable to connect to etcd for testing: %v", err)
	}
	return client
}

func TestIgnoreClusterName(t *testing.T) {
	config := framework.NewMasterConfig()
	prefix := config.StorageFactory.(*genericapiserver.DefaultStorageFactory).StorageConfig.Prefix
	_, s := framework.RunAMaster(config)
	defer s.Close()

	client := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(v1.GroupName).GroupVersion}})
	etcdClient := newEtcdClient()
	etcdStorage := etcdstorage.NewEtcdStorage(etcdClient, testapi.Default.Codec(),
		prefix+"/namespaces/", false, etcdtest.DeserializationCacheSize)
	ctx := context.TODO()

	ns := v1.Namespace{
		ObjectMeta: v1.ObjectMeta{
			Name:        "test-namespace",
			ClusterName: "cluster-name-to-ignore",
		},
	}
	nsNew, err := client.Core().Namespaces().Create(&ns)
	assert.Nil(t, err)
	assert.Equal(t, ns.Name, nsNew.Name)
	assert.Empty(t, nsNew.ClusterName)

	nsEtcd := v1.Namespace{}
	err = etcdStorage.Get(ctx, ns.Name, &nsEtcd, false)
	assert.Nil(t, err)
	assert.Equal(t, ns.Name, nsEtcd.Name)
	assert.Empty(t, nsEtcd.ClusterName)

	nsNew, err = client.Core().Namespaces().Update(&ns)
	assert.Nil(t, err)
	assert.Equal(t, ns.Name, nsNew.Name)
	assert.Empty(t, nsNew.ClusterName)

	nsEtcd = v1.Namespace{}
	err = etcdStorage.Get(ctx, ns.Name, &nsEtcd, false)
	assert.Nil(t, err)
	assert.Equal(t, ns.Name, nsEtcd.Name)
	assert.Empty(t, nsEtcd.ClusterName)
}
