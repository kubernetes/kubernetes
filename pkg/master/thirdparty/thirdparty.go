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
	"reflect"
	"regexp"
	"strings"
	"sync"

	"github.com/emicklei/go-restful"
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
	"k8s.io/kubernetes/pkg/util/sets"
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
	// map from api path to a tuple of (versioned storage for the objects, APIGroup)
	thirdPartyResources map[string]*thirdPartyGroup
	// protects the map
	thirdPartyResourcesLock sync.RWMutex

	// Useful for reliable testing.  Shouldn't be used otherwise.
	disableThirdPartyControllerForTesting bool
}

func NewThirdPartyResourceServer(genericAPIServer *genericapiserver.GenericAPIServer, storageFactory genericapiserver.StorageFactory) *ThirdPartyResourceServer {
	ret := &ThirdPartyResourceServer{
		genericAPIServer:    genericAPIServer,
		thirdPartyResources: map[string]*thirdPartyGroup{},
	}

	var err error
	ret.thirdPartyStorageConfig, err = storageFactory.NewConfig(extensions.Resource("thirdpartyresources"))
	if err != nil {
		glog.Fatalf("Error building third party storage: %v", err)
	}

	return ret
}

type thirdPartyGroup struct {
	versionedEntry map[string]*thirdPartyEntry
	group          unversioned.APIGroup
}

// thirdPartyEntry combines objects storage and API group into one struct
// for easy lookup.
type thirdPartyEntry struct {
	// Map from plural resource name to entry
	storage map[string]*thirdpartyresourcedataetcd.REST
}

func (g *thirdPartyGroup) versionInstalled(version string) bool {
	for installedVersion := range g.versionedEntry {
		if reflect.DeepEqual(version, installedVersion) {
			return true
		}
	}
	return false
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
	tpg := m.thirdPartyResources[path]
	if tpg == nil {
		return false, nil
	}
	plural, _ := meta.KindToResource(unversioned.GroupVersionKind{
		Group:   group,
		Version: rsrc.Versions[0].Name,
		Kind:    kind,
	})
	for _, version := range rsrc.Versions {
		entry, _ := tpg.versionedEntry[version.Name]
		if entry == nil {
			return false, nil
		}
		_, found := entry.storage[plural.Resource]
		if !found {
			return false, nil
		}
	}
	return true, nil
}

func (m *ThirdPartyResourceServer) removeThirdPartyStorage(path, resource string) error {
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	ix := strings.LastIndex(path, "/")
	if ix == -1 {
		return fmt.Errorf("expected <api-group>/<version>, saw: %s", path)
	}
	version := path[ix+1:]
	group := path[0:ix]
	tpg, found := m.thirdPartyResources[group]
	if !found {
		return nil
	}
	entry, found := tpg.versionedEntry[version]
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
		delete(tpg.versionedEntry, version)
	}
	if len(tpg.versionedEntry) == 0 {
		delete(m.thirdPartyResources, group)
		m.genericAPIServer.RemoveAPIGroupForDiscovery(group)
	} else {
		m.thirdPartyResources[group] = tpg
	}
	return nil
}

// RemoveThirdPartyResource removes all resources matching `path`.  Also deletes any stored data
func (m *ThirdPartyResourceServer) RemoveThirdPartyResource(path string) error {
	ix := strings.LastIndex(path, "/")
	if ix == -1 {
		return fmt.Errorf("expected <api-group>/<version>/<resource-plural-name>, saw: %s", path)
	}
	resource := path[ix+1:]
	path = path[0:ix]

	if err := m.removeThirdPartyStorage(path, resource); err != nil {
		return err
	}

	var ws *restful.WebService = nil
	services := m.genericAPIServer.HandlerContainer.RegisteredWebServices()
	for ix := range services {
		root := services[ix].RootPath()
		if root == path || strings.HasPrefix(root, path+"/") {
			ws = services[ix]
			break
		}
	}

	if ws != nil {
		pattern := path + "/.*/" + resource + ".*"
		routesToRemove := []struct {
			method string
			path   string
		}{}
		routes := ws.Routes()
		for _, route := range routes {
			match, _ := regexp.MatchString(pattern, route.Path)
			if match {
				routesToRemove = append(routesToRemove, struct{ method, path string }{route.Method, route.Path})
			}
		}

		for _, routeToRemove := range routesToRemove {
			ws.RemoveRoute(routeToRemove.path, routeToRemove.method)
		}
		m.thirdPartyResourcesLock.Lock()
		_, found := m.thirdPartyResources[path]
		m.thirdPartyResourcesLock.Unlock()
		if !found {
			m.genericAPIServer.HandlerContainer.Remove(ws)
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
	for group := range m.thirdPartyResources {
		for version, entry := range m.thirdPartyResources[group].versionedEntry {
			for rsrc := range entry.storage {
				result = append(result, group+"/"+version+"/"+rsrc)
			}
		}
	}
	return result
}

func (m *ThirdPartyResourceServer) getExistingThirdPartyResources(path string) []unversioned.APIResource {
	result := []unversioned.APIResource{}
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	tpg := m.thirdPartyResources[path]
	if tpg != nil {
		for _, entry := range tpg.versionedEntry {
			for key, obj := range entry.storage {
				result = append(result, unversioned.APIResource{
					Name:       key,
					Namespaced: true,
					Kind:       obj.Kind(),
				})
			}
		}
	}
	return result
}

func (m *ThirdPartyResourceServer) getThirdPartyGroup(path string) *thirdPartyGroup {
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	entry, _ := m.thirdPartyResources[path]
	return entry
}

func (m *ThirdPartyResourceServer) hasThirdPartyGroupStorage(path string) bool {
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	_, found := m.thirdPartyResources[path]
	return found
}

func (m *ThirdPartyResourceServer) addThirdPartyResourceStorage(path, version, resource string, storage *thirdpartyresourcedataetcd.REST) {
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	group, _ := m.thirdPartyResources[path]
	if group == nil {
		group = &thirdPartyGroup{
			group:          unversioned.APIGroup{},
			versionedEntry: map[string]*thirdPartyEntry{},
		}
		m.thirdPartyResources[path] = group
	}
	entry, _ := group.versionedEntry[version]
	if entry == nil {
		entry = &thirdPartyEntry{
			storage: map[string]*thirdpartyresourcedataetcd.REST{},
		}
		group.versionedEntry[version] = entry
	}
	entry.storage[resource] = storage
}

func (m *ThirdPartyResourceServer) setupGroupForDiscovery(group string, versions ...string) {
	path := extensionsrest.MakeThirdPartyPath(group)
	m.thirdPartyResourcesLock.Lock()
	tpg, _ := m.thirdPartyResources[path]
	m.thirdPartyResourcesLock.Unlock()
	apiVersionsForDiscovery := []unversioned.GroupVersionForDiscovery{}
	var preferredVersion string
	for i, version := range versions {
		if i == 0 {
			preferredVersion = version
		}
		apiVersionsForDiscovery = append(apiVersionsForDiscovery, unversioned.GroupVersionForDiscovery{
			GroupVersion: group + "/" + version,
			Version:      version,
		})
	}
	preferedVersionForDiscovery := unversioned.GroupVersionForDiscovery{
		GroupVersion: group + "/" + preferredVersion,
		Version:      preferredVersion,
	}
	apiGroup := unversioned.APIGroup{
		Name:             group,
		Versions:         apiVersionsForDiscovery,
		PreferredVersion: preferedVersionForDiscovery,
	}
	tpg.group = apiGroup
	m.genericAPIServer.HandlerContainer.Add(apiserver.NewGroupWebService(api.Codecs, "/apis/"+apiGroup.Name, apiGroup))
	m.genericAPIServer.AddAPIGroupForDiscovery(apiGroup)
}

func (m *ThirdPartyResourceServer) updateGroupForDiscovery(group string, versions ...string) {
	path := extensionsrest.MakeThirdPartyPath(group)
	m.thirdPartyResourcesLock.Lock()
	tpg, _ := m.thirdPartyResources[path]
	m.thirdPartyResourcesLock.Unlock()
	existingAPIGroup := tpg.group
	existingAPIVersions := existingAPIGroup.Versions
	newVersions := sets.NewString()
	for _, existingAPIVersion := range existingAPIVersions {
		newVersions.Insert(existingAPIVersion.Version)
	}
	newVersions.Insert(versions...)
	apiVersionsForDiscovery := []unversioned.GroupVersionForDiscovery{}
	for _, version := range newVersions.List() {
		apiVersionsForDiscovery = append(apiVersionsForDiscovery, unversioned.GroupVersionForDiscovery{
			GroupVersion: group + "/" + version,
			Version:      version,
		})
	}
	newAPIGroup := unversioned.APIGroup{
		Name:             group,
		Versions:         apiVersionsForDiscovery,
		PreferredVersion: existingAPIGroup.PreferredVersion,
	}
	tpg.group = newAPIGroup
	m.genericAPIServer.RemoveAPIGroupForDiscovery(existingAPIGroup.Name)
	m.genericAPIServer.AddAPIGroupForDiscovery(newAPIGroup)
	m.genericAPIServer.HandlerContainer.Remove(apiserver.NewGroupWebService(api.Codecs, "/apis/"+existingAPIGroup.Name, existingAPIGroup))
	m.genericAPIServer.HandlerContainer.Add(apiserver.NewGroupWebService(api.Codecs, "/apis/"+newAPIGroup.Name, newAPIGroup))
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

	path := extensionsrest.MakeThirdPartyPath(group)
	var firstSeen, hasNewVersion bool
	for _, version := range rsrc.Versions {
		plural, _ := meta.KindToResource(unversioned.GroupVersionKind{
			Group:   group,
			Version: version.Name,
			Kind:    kind,
		})

		thirdparty := m.thirdpartyapi(group, kind, version.Name, plural.Resource)
		thirdPartyGroup := m.getThirdPartyGroup(path)
		// If thirdPartyGroup is nil, this group has not ever been seen
		if thirdPartyGroup == nil {
			firstSeen = true
			if err := thirdparty.InstallREST(m.genericAPIServer.HandlerContainer.Container); err != nil {
				glog.Errorf("Unable to setup thirdparty api: %v", err)
				return fmt.Errorf("Unable to setup thirdparty api: %v", err)
			}

			m.addThirdPartyResourceStorage(path, version.Name, plural.Resource, thirdparty.Storage[plural.Resource].(*thirdpartyresourcedataetcd.REST))
			continue
		}

		// if version installed, just update the group with the new API
		if thirdPartyGroup.versionInstalled(version.Name) {
			if err := thirdparty.UpdateREST(m.genericAPIServer.HandlerContainer.Container); err != nil {
				glog.Errorf("Unable to update thirdparty api: %v", err)
			}

			m.addThirdPartyResourceStorage(path, version.Name, plural.Resource, thirdparty.Storage[plural.Resource].(*thirdpartyresourcedataetcd.REST))
			continue
		}

		hasNewVersion = true
		if err := thirdparty.InstallREST(m.genericAPIServer.HandlerContainer.Container); err != nil {
			glog.Errorf("Unable to setup thirdparty api: %v", err)
			return fmt.Errorf("Unable to setup thirdparty api: %v", err)
		}
		m.addThirdPartyResourceStorage(path, version.Name, plural.Resource, thirdparty.Storage[plural.Resource].(*thirdpartyresourcedataetcd.REST))
	}

	if firstSeen {
		// setup discovery
		versions := []string{}
		for _, version := range rsrc.Versions {
			versions = append(versions, version.Name)
		}
		m.setupGroupForDiscovery(group, versions...)
		return nil
	}
	if hasNewVersion {
		versions := []string{}
		for _, version := range rsrc.Versions {
			versions = append(versions, version.Name)
		}
		m.updateGroupForDiscovery(group, versions...)
	}
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
