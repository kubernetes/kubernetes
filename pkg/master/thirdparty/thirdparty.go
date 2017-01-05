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

package thirdparty

import (
	"fmt"
	"strings"
	"sync"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/genericapiserver"
	extensionsrest "k8s.io/kubernetes/pkg/registry/extensions/rest"
	"k8s.io/kubernetes/pkg/registry/extensions/thirdpartyresourcedata"
	thirdpartyresourcedataetcd "k8s.io/kubernetes/pkg/registry/extensions/thirdpartyresourcedata/etcd"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage/storagebackend"
)

// dynamicLister is used to list resources for dynamic third party
// apis. It implements the apiserver.APIResourceLister interface
type dynamicLister struct {
	m    *ThirdPartyResourceServer
	path string
}

func (d dynamicLister) ListAPIResources() []unversioned.APIResource {
	return d.m.getExistingThirdPartyResources(d.path)
}

var _ apiserver.APIResourceLister = &dynamicLister{}

type ThirdPartyResourceServer struct {
	genericAPIServer *genericapiserver.GenericAPIServer

	deleteCollectionWorkers int

	// storage for third party objects
	thirdPartyStorageConfig *storagebackend.Config
	// map from api path to a tuple of (storage for the objects, APIGroup)
	thirdPartyResources map[string]*thirdPartyEntry
	// protects the map
	thirdPartyResourcesLock sync.RWMutex

	// Useful for reliable testing.  Shouldn't be used otherwise.
	disableThirdPartyControllerForTesting bool
}

func NewThirdPartyResourceServer(genericAPIServer *genericapiserver.GenericAPIServer, storageFactory genericapiserver.StorageFactory) *ThirdPartyResourceServer {
	ret := &ThirdPartyResourceServer{
		genericAPIServer:    genericAPIServer,
		thirdPartyResources: map[string]*thirdPartyEntry{},
	}

	var err error
	ret.thirdPartyStorageConfig, err = storageFactory.NewConfig(extensions.Resource("thirdpartyresources"))
	if err != nil {
		glog.Fatalf("Error building third party storage: %v", err)
	}

	return ret
}

// thirdPartyEntry combines objects storage and API group into one struct
// for easy lookup.
type thirdPartyEntry struct {
	// Map from plural resource name to entry
	storage map[string]*thirdpartyresourcedataetcd.REST
	group   unversioned.APIGroup
}

// HasThirdPartyResource returns true if a particular third party resource currently installed.
func (m *ThirdPartyResourceServer) HasThirdPartyResource(rsrc *extensions.ThirdPartyResource) (bool, error) {
	kind, group, err := thirdpartyresourcedata.ExtractApiGroupAndKind(rsrc)
	if err != nil {
		return false, err
	}
	path := extensionsrest.MakeThirdPartyPath(group)
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	entry := m.thirdPartyResources[path]
	if entry == nil {
		return false, nil
	}
	plural, _ := meta.KindToResource(unversioned.GroupVersionKind{
		Group:   group,
		Version: rsrc.Versions[0].Name,
		Kind:    kind,
	})
	_, found := entry.storage[plural.Resource]
	return found, nil
}

func (m *ThirdPartyResourceServer) removeThirdPartyStorage(path, resource string) error {
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	entry, found := m.thirdPartyResources[path]
	if !found {
		return nil
	}
	storage, found := entry.storage[resource]
	if !found {
		return nil
	}
	if err := m.removeAllThirdPartyResources(storage); err != nil {
		return err
	}
	delete(entry.storage, resource)
	if len(entry.storage) == 0 {
		delete(m.thirdPartyResources, path)
		m.genericAPIServer.RemoveAPIGroupForDiscovery(extensionsrest.GetThirdPartyGroupName(path))
	} else {
		m.thirdPartyResources[path] = entry
	}
	return nil
}

// RemoveThirdPartyResource removes all resources matching `path`.  Also deletes any stored data
func (m *ThirdPartyResourceServer) RemoveThirdPartyResource(path string) error {
	ix := strings.LastIndex(path, "/")
	if ix == -1 {
		return fmt.Errorf("expected <api-group>/<resource-plural-name>, saw: %s", path)
	}
	resource := path[ix+1:]
	path = path[0:ix]

	if err := m.removeThirdPartyStorage(path, resource); err != nil {
		return err
	}

	services := m.genericAPIServer.HandlerContainer.RegisteredWebServices()
	for ix := range services {
		root := services[ix].RootPath()
		if root == path || strings.HasPrefix(root, path+"/") {
			m.genericAPIServer.HandlerContainer.Remove(services[ix])
		}
	}
	return nil
}

func (m *ThirdPartyResourceServer) removeAllThirdPartyResources(registry *thirdpartyresourcedataetcd.REST) error {
	ctx := api.NewDefaultContext()
	existingData, err := registry.List(ctx, nil)
	if err != nil {
		return err
	}
	list, ok := existingData.(*extensions.ThirdPartyResourceDataList)
	if !ok {
		return fmt.Errorf("expected a *ThirdPartyResourceDataList, got %#v", list)
	}
	for ix := range list.Items {
		item := &list.Items[ix]
		if _, err := registry.Delete(ctx, item.Name, nil); err != nil {
			return err
		}
	}
	return nil
}

// ListThirdPartyResources lists all currently installed third party resources
// The format is <path>/<resource-plural-name>
func (m *ThirdPartyResourceServer) ListThirdPartyResources() []string {
	m.thirdPartyResourcesLock.RLock()
	defer m.thirdPartyResourcesLock.RUnlock()
	result := []string{}
	for key := range m.thirdPartyResources {
		for rsrc := range m.thirdPartyResources[key].storage {
			result = append(result, key+"/"+rsrc)
		}
	}
	return result
}

func (m *ThirdPartyResourceServer) getExistingThirdPartyResources(path string) []unversioned.APIResource {
	result := []unversioned.APIResource{}
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	entry := m.thirdPartyResources[path]
	if entry != nil {
		for key, obj := range entry.storage {
			result = append(result, unversioned.APIResource{
				Name:       key,
				Namespaced: true,
				Kind:       obj.Kind(),
			})
		}
	}
	return result
}

func (m *ThirdPartyResourceServer) hasThirdPartyGroupStorage(path string) bool {
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	_, found := m.thirdPartyResources[path]
	return found
}

func (m *ThirdPartyResourceServer) addThirdPartyResourceStorage(path, resource string, storage *thirdpartyresourcedataetcd.REST, apiGroup unversioned.APIGroup) {
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	entry, found := m.thirdPartyResources[path]
	if entry == nil {
		entry = &thirdPartyEntry{
			group:   apiGroup,
			storage: map[string]*thirdpartyresourcedataetcd.REST{},
		}
		m.thirdPartyResources[path] = entry
	}
	entry.storage[resource] = storage
	if !found {
		m.genericAPIServer.AddAPIGroupForDiscovery(apiGroup)
	}
}

// InstallThirdPartyResource installs a third party resource specified by 'rsrc'.  When a resource is
// installed a corresponding RESTful resource is added as a valid path in the web service provided by
// the master.
//
// For example, if you install a resource ThirdPartyResource{ Name: "foo.company.com", Versions: {"v1"} }
// then the following RESTful resource is created on the server:
//   http://<host>/apis/company.com/v1/foos/...
func (m *ThirdPartyResourceServer) InstallThirdPartyResource(rsrc *extensions.ThirdPartyResource) error {
	kind, group, err := thirdpartyresourcedata.ExtractApiGroupAndKind(rsrc)
	if err != nil {
		return err
	}
	if len(rsrc.Versions) == 0 {
		return fmt.Errorf("ThirdPartyResource %s has no defined versions", rsrc.Name)
	}
	plural, _ := meta.KindToResource(unversioned.GroupVersionKind{
		Group:   group,
		Version: rsrc.Versions[0].Name,
		Kind:    kind,
	})
	path := extensionsrest.MakeThirdPartyPath(group)

	groupVersion := unversioned.GroupVersionForDiscovery{
		GroupVersion: group + "/" + rsrc.Versions[0].Name,
		Version:      rsrc.Versions[0].Name,
	}
	apiGroup := unversioned.APIGroup{
		Name:             group,
		Versions:         []unversioned.GroupVersionForDiscovery{groupVersion},
		PreferredVersion: groupVersion,
	}

	thirdparty := m.thirdpartyapi(group, kind, rsrc.Versions[0].Name, plural.Resource)

	// If storage exists, this group has already been added, just update
	// the group with the new API
	if m.hasThirdPartyGroupStorage(path) {
		m.addThirdPartyResourceStorage(path, plural.Resource, thirdparty.Storage[plural.Resource].(*thirdpartyresourcedataetcd.REST), apiGroup)
		return thirdparty.UpdateREST(m.genericAPIServer.HandlerContainer.Container)
	}

	if err := thirdparty.InstallREST(m.genericAPIServer.HandlerContainer.Container); err != nil {
		glog.Errorf("Unable to setup thirdparty api: %v", err)
	}
	m.genericAPIServer.HandlerContainer.Add(apiserver.NewGroupWebService(api.Codecs, path, apiGroup))

	m.addThirdPartyResourceStorage(path, plural.Resource, thirdparty.Storage[plural.Resource].(*thirdpartyresourcedataetcd.REST), apiGroup)
	registered.AddThirdPartyAPIGroupVersions(unversioned.GroupVersion{Group: group, Version: rsrc.Versions[0].Name})
	return nil
}

func (m *ThirdPartyResourceServer) thirdpartyapi(group, kind, version, pluralResource string) *apiserver.APIGroupVersion {
	resourceStorage := thirdpartyresourcedataetcd.NewREST(
		generic.RESTOptions{
			StorageConfig:           m.thirdPartyStorageConfig,
			Decorator:               generic.UndecoratedStorage,
			DeleteCollectionWorkers: m.deleteCollectionWorkers,
		},
		group,
		kind,
	)

	storage := map[string]rest.Storage{
		pluralResource: resourceStorage,
	}

	optionsExternalVersion := registered.GroupOrDie(api.GroupName).GroupVersion
	internalVersion := unversioned.GroupVersion{Group: group, Version: runtime.APIVersionInternal}
	externalVersion := unversioned.GroupVersion{Group: group, Version: version}

	apiRoot := extensionsrest.MakeThirdPartyPath("")
	return &apiserver.APIGroupVersion{
		Root:         apiRoot,
		GroupVersion: externalVersion,

		Creater:   thirdpartyresourcedata.NewObjectCreator(group, version, api.Scheme),
		Convertor: api.Scheme,
		Copier:    api.Scheme,
		Typer:     api.Scheme,

		Mapper:                 thirdpartyresourcedata.NewMapper(registered.GroupOrDie(extensions.GroupName).RESTMapper, kind, version, group),
		Linker:                 registered.GroupOrDie(extensions.GroupName).SelfLinker,
		Storage:                storage,
		OptionsExternalVersion: &optionsExternalVersion,

		Serializer:     thirdpartyresourcedata.NewNegotiatedSerializer(api.Codecs, kind, externalVersion, internalVersion),
		ParameterCodec: thirdpartyresourcedata.NewThirdPartyParameterCodec(api.ParameterCodec),

		Context: m.genericAPIServer.RequestContextMapper(),

		MinRequestTimeout: m.genericAPIServer.MinRequestTimeout(),

		ResourceLister: dynamicLister{m, extensionsrest.MakeThirdPartyPath(group)},
	}
}
