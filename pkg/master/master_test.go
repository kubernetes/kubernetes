/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package master

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/expapi"
	explatest "k8s.io/kubernetes/pkg/expapi/latest"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"

	"github.com/emicklei/go-restful"
)

func TestGetServersToValidate(t *testing.T) {
	master := Master{}
	config := Config{}
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.Machines = []string{"http://machine1:4001", "http://machine2", "http://machine3:4003"}
	config.DatabaseStorage = etcdstorage.NewEtcdStorage(fakeClient, latest.Codec, etcdtest.PathPrefix())
	config.ExpDatabaseStorage = etcdstorage.NewEtcdStorage(fakeClient, explatest.Codec, etcdtest.PathPrefix())

	master.nodeRegistry = registrytest.NewNodeRegistry([]string{"node1", "node2"}, api.NodeResources{})

	servers := master.getServersToValidate(&config)

	if len(servers) != 5 {
		t.Errorf("unexpected server list: %#v", servers)
	}
	for _, server := range []string{"scheduler", "controller-manager", "etcd-0", "etcd-1", "etcd-2"} {
		if _, ok := servers[server]; !ok {
			t.Errorf("server list missing: %s", server)
		}
	}
}

func TestFindExternalAddress(t *testing.T) {
	expectedIP := "172.0.0.1"

	nodes := []*api.Node{new(api.Node), new(api.Node), new(api.Node)}
	nodes[0].Status.Addresses = []api.NodeAddress{{"ExternalIP", expectedIP}}
	nodes[1].Status.Addresses = []api.NodeAddress{{"LegacyHostIP", expectedIP}}
	nodes[2].Status.Addresses = []api.NodeAddress{{"ExternalIP", expectedIP}, {"LegacyHostIP", "172.0.0.2"}}

	for _, node := range nodes {
		ip, err := findExternalAddress(node)
		if err != nil {
			t.Errorf("error getting node external address: %s", err)
		}
		if ip != expectedIP {
			t.Errorf("expected ip to be %s, but was %s", expectedIP, ip)
		}
	}

	_, err := findExternalAddress(new(api.Node))
	if err == nil {
		t.Errorf("expected findExternalAddress to fail on a node with missing ip information")
	}
}

var versionsToTest = []string{"v1", "v3"}

type Foo struct {
	api.TypeMeta   `json:",inline"`
	api.ObjectMeta `json:"metadata,omitempty" description:"standard object metadata"`

	SomeField  string `json:"someField"`
	OtherField int    `json:"otherField"`
}

type FooList struct {
	api.TypeMeta `json:",inline"`
	api.ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://docs.k8s.io/api-conventions.md#metadata"`

	items []Foo `json:"items"`
}

func initThirdParty(t *testing.T, version string) (*tools.FakeEtcdClient, *httptest.Server) {
	master := &Master{}
	api := &expapi.ThirdPartyResource{
		ObjectMeta: api.ObjectMeta{
			Name: "foo.company.com",
		},
		Versions: []expapi.APIVersion{
			{
				APIGroup: "group",
				Name:     version,
			},
		},
	}
	master.handlerContainer = restful.NewContainer()

	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.Machines = []string{"http://machine1:4001", "http://machine2", "http://machine3:4003"}
	master.thirdPartyStorage = etcdstorage.NewEtcdStorage(fakeClient, explatest.Codec, etcdtest.PathPrefix())

	if err := master.InstallThirdPartyAPI(api); err != nil {
		t.Errorf("unexpected error: %v", err)
		t.FailNow()
	}

	server := httptest.NewServer(master.handlerContainer.ServeMux)
	return fakeClient, server
}

func TestInstallThirdPartyAPIList(t *testing.T) {
	for _, version := range versionsToTest {
		testInstallThirdPartyAPIListVersion(t, version)
	}
}

func testInstallThirdPartyAPIListVersion(t *testing.T, version string) {
	fakeClient, server := initThirdParty(t, version)
	defer server.Close()

	fakeClient.ExpectNotFoundGet(etcdtest.PathPrefix() + "/ThirdPartyResourceData/company.com/foos/default")

	resp, err := http.Get(server.URL + "/thirdparty/company.com/" + version + "/namespaces/default/foos")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("unexpected status: %v", resp)
	}

	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	list := FooList{}
	if err := json.Unmarshal(data, &list); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func encodeToThirdParty(name string, obj interface{}) ([]byte, error) {
	serial, err := json.Marshal(obj)
	if err != nil {
		return nil, err
	}
	thirdPartyData := expapi.ThirdPartyResourceData{
		ObjectMeta: api.ObjectMeta{Name: name},
		Data:       serial,
	}
	return latest.Codec.Encode(&thirdPartyData)
}

func storeToEtcd(fakeClient *tools.FakeEtcdClient, path, name string, obj interface{}) error {
	data, err := encodeToThirdParty(name, obj)
	if err != nil {
		return err
	}
	_, err = fakeClient.Set(etcdtest.PathPrefix()+path, string(data), 0)
	return err
}

func decodeResponse(resp *http.Response, obj interface{}) error {
	defer resp.Body.Close()

	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	if err := json.Unmarshal(data, obj); err != nil {
		return err
	}
	return nil
}

func TestInstallThirdPartyAPIGet(t *testing.T) {
	for _, version := range versionsToTest {
		testInstallThirdPartyAPIGetVersion(t, version)
	}
}

func testInstallThirdPartyAPIGetVersion(t *testing.T, version string) {
	fakeClient, server := initThirdParty(t, version)
	defer server.Close()

	expectedObj := Foo{
		ObjectMeta: api.ObjectMeta{
			Name: "test",
		},
		TypeMeta: api.TypeMeta{
			Kind:       "Foo",
			APIVersion: version,
		},
		SomeField:  "test field",
		OtherField: 10,
	}
	if err := storeToEtcd(fakeClient, "/ThirdPartyResourceData/company.com/foos/default/test", "test", expectedObj); err != nil {
		t.Errorf("unexpected error: %v", err)
		t.FailNow()
		return
	}

	resp, err := http.Get(server.URL + "/thirdparty/company.com/" + version + "/namespaces/default/foos/test")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}

	if resp.StatusCode != http.StatusOK {
		t.Errorf("unexpected status: %v", resp)
	}
	item := Foo{}
	if err := decodeResponse(resp, &item); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	// Fill in data that the apiserver injects
	expectedObj.SelfLink = item.SelfLink
	if !reflect.DeepEqual(item, expectedObj) {
		t.Errorf("expected:\n%#v\nsaw:\n%#v\n", expectedObj, item)
	}
}

func TestInstallThirdPartyAPIPost(t *testing.T) {
	for _, version := range versionsToTest {
		testInstallThirdPartyAPIPostForVersion(t, version)
	}
}

func testInstallThirdPartyAPIPostForVersion(t *testing.T, version string) {
	fakeClient, server := initThirdParty(t, version)
	defer server.Close()

	inputObj := Foo{
		ObjectMeta: api.ObjectMeta{
			Name: "test",
		},
		TypeMeta: api.TypeMeta{
			Kind:       "Foo",
			APIVersion: version,
		},
		SomeField:  "test field",
		OtherField: 10,
	}
	data, err := json.Marshal(inputObj)
	if err != nil {
		t.Errorf("unexpected error: %v")
		return
	}

	resp, err := http.Post(server.URL+"/thirdparty/company.com/"+version+"/namespaces/default/foos", "application/json", bytes.NewBuffer(data))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}

	if resp.StatusCode != http.StatusCreated {
		t.Errorf("unexpected status: %v", resp)
	}

	item := Foo{}
	if err := decodeResponse(resp, &item); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// fill in fields set by the apiserver
	expectedObj := inputObj
	expectedObj.SelfLink = item.SelfLink
	expectedObj.Namespace = item.Namespace
	expectedObj.UID = item.UID
	expectedObj.CreationTimestamp = item.CreationTimestamp
	if !reflect.DeepEqual(item, expectedObj) {
		t.Errorf("expected:\n%v\nsaw:\n%v\n", expectedObj, item)
	}

	etcdResp, err := fakeClient.Get(etcdtest.PathPrefix()+"/ThirdPartyResourceData/company.com/foos/default/test", false, false)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		t.FailNow()
	}
	obj, err := explatest.Codec.Decode([]byte(etcdResp.Node.Value))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	thirdPartyObj, ok := obj.(*expapi.ThirdPartyResourceData)
	if !ok {
		t.Errorf("unexpected object: %v", obj)
	}
	item = Foo{}
	if err := json.Unmarshal(thirdPartyObj.Data, &item); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(item, inputObj) {
		t.Errorf("expected:\n%v\nsaw:\n%v\n", inputObj, item)
	}
}

func TestInstallThirdPartyAPIDelete(t *testing.T) {
	for _, version := range versionsToTest {
		testInstallThirdPartyAPIDeleteVersion(t, version)
	}
}

func testInstallThirdPartyAPIDeleteVersion(t *testing.T, version string) {
	fakeClient, server := initThirdParty(t, version)
	defer server.Close()

	expectedObj := Foo{
		ObjectMeta: api.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		TypeMeta: api.TypeMeta{
			Kind: "Foo",
		},
		SomeField:  "test field",
		OtherField: 10,
	}
	if err := storeToEtcd(fakeClient, "/ThirdPartyResourceData/company.com/foos/default/test", "test", expectedObj); err != nil {
		t.Errorf("unexpected error: %v", err)
		t.FailNow()
		return
	}

	resp, err := http.Get(server.URL + "/thirdparty/company.com/" + version + "/namespaces/default/foos/test")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}

	if resp.StatusCode != http.StatusOK {
		t.Errorf("unexpected status: %v", resp)
	}

	item := Foo{}
	if err := decodeResponse(resp, &item); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Fill in fields set by the apiserver
	expectedObj.SelfLink = item.SelfLink
	expectedObj.Namespace = item.Namespace
	if !reflect.DeepEqual(item, expectedObj) {
		t.Errorf("expected:\n%v\nsaw:\n%v\n", expectedObj, item)
	}

	resp, err = httpDelete(server.URL + "/thirdparty/company.com/" + version + "/namespaces/default/foos/test")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}

	if resp.StatusCode != http.StatusOK {
		t.Errorf("unexpected status: %v", resp)
	}

	resp, err = http.Get(server.URL + "/thirdparty/company.com/" + version + "/namespaces/default/foos/test")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("unexpected status: %v", resp)
	}
	expectDeletedKeys := []string{etcdtest.PathPrefix() + "/ThirdPartyResourceData/company.com/foos/default/test"}
	if !reflect.DeepEqual(fakeClient.DeletedKeys, expectDeletedKeys) {
		t.Errorf("unexpected deleted keys: %v", fakeClient.DeletedKeys)
	}
}

func httpDelete(url string) (*http.Response, error) {
	req, err := http.NewRequest("DELETE", url, nil)
	if err != nil {
		return nil, err
	}
	client := &http.Client{}
	return client.Do(req)
}
