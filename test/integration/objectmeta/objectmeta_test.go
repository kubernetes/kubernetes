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
	"github.com/stretchr/testify/assert"
	"golang.org/x/net/context"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/genericapiserver"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/storage/etcd/etcdtest"
	"k8s.io/kubernetes/test/integration/framework"
	"testing"
)

func TestIgnoreClusterName(t *testing.T) {
	config := framework.NewMasterConfig()
	prefix := config.StorageFactory.(*genericapiserver.DefaultStorageFactory).StorageConfig.Prefix
	_, s := framework.RunAMaster(config)
	defer s.Close()

	client := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	etcdClient := framework.NewEtcdClient()
	etcdStorage := etcdstorage.NewEtcdStorage(etcdClient, testapi.Default.Codec(),
		prefix+"/namespaces/", false, etcdtest.DeserializationCacheSize)
	ctx := context.TODO()

	ns := api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:        "test-namespace",
			ClusterName: "cluster-name-to-ignore",
		},
	}
	nsNew, err := client.Namespaces().Create(&ns)
	assert.Nil(t, err)
	assert.Equal(t, ns.Name, nsNew.Name)
	assert.Empty(t, nsNew.ClusterName)

	nsEtcd := api.Namespace{}
	err = etcdStorage.Get(ctx, ns.Name, &nsEtcd, false)
	assert.Nil(t, err)
	assert.Equal(t, ns.Name, nsEtcd.Name)
	assert.Empty(t, nsEtcd.ClusterName)

	nsNew, err = client.Namespaces().Update(&ns)
	assert.Nil(t, err)
	assert.Equal(t, ns.Name, nsNew.Name)
	assert.Empty(t, nsNew.ClusterName)

	nsEtcd = api.Namespace{}
	err = etcdStorage.Get(ctx, ns.Name, &nsEtcd, false)
	assert.Nil(t, err)
	assert.Equal(t, ns.Name, nsEtcd.Name)
	assert.Empty(t, nsEtcd.ClusterName)
}
