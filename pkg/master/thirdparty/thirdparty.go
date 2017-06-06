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

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsserver "k8s.io/apiextensions-apiserver/pkg/apiserver"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/internalclientset/typed/apiextensions/internalversion"
	"k8s.io/apiextensions-apiserver/pkg/registry/customresource"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/json"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	genericapi "k8s.io/apiserver/pkg/endpoints"
	"k8s.io/apiserver/pkg/endpoints/discovery"
	"k8s.io/apiserver/pkg/endpoints/request"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorgage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	discoveryclient "k8s.io/client-go/discovery"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extensionsrest "k8s.io/kubernetes/pkg/registry/extensions/rest"
	"k8s.io/kubernetes/pkg/registry/extensions/thirdpartyresourcedata"
	thirdpartyresourcedatastore "k8s.io/kubernetes/pkg/registry/extensions/thirdpartyresourcedata/storage"
)

// dynamicLister is used to list resources for dynamic third party
// apis. It implements the genericapihandlers.APIResourceLister interface
type dynamicLister struct {
	m    *ThirdPartyResourceServer
	path string
}

func (d dynamicLister) ListAPIResources() []metav1.APIResource {
	return d.m.getExistingThirdPartyResources(d.path)
}

var _ discovery.APIResourceLister = &dynamicLister{}

type ThirdPartyResourceServer struct {
	genericAPIServer *genericapiserver.GenericAPIServer

	availableGroupManager discovery.GroupManager

	deleteCollectionWorkers int

	// storage for third party objects
	thirdPartyStorageConfig *storagebackend.Config
	// map from api path to a tuple of (storage for the objects, APIGroup)
	thirdPartyResources map[string]*thirdPartyEntry
	// protects the map
	thirdPartyResourcesLock sync.RWMutex

	// Useful for reliable testing.  Shouldn't be used otherwise.
	disableThirdPartyControllerForTesting bool

	crdRESTOptionsGetter generic.RESTOptionsGetter
}

func NewThirdPartyResourceServer(genericAPIServer *genericapiserver.GenericAPIServer, availableGroupManager discovery.GroupManager, storageFactory serverstorgage.StorageFactory, crdRESTOptionsGetter generic.RESTOptionsGetter) *ThirdPartyResourceServer {
	ret := &ThirdPartyResourceServer{
		genericAPIServer:      genericAPIServer,
		thirdPartyResources:   map[string]*thirdPartyEntry{},
		availableGroupManager: availableGroupManager,
		crdRESTOptionsGetter:  crdRESTOptionsGetter,
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
	storage map[string]*thirdpartyresourcedatastore.REST
	group   metav1.APIGroup
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
	plural, _ := meta.UnsafeGuessKindToResource(schema.GroupVersionKind{
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
	if err := m.removeThirdPartyResourceData(&entry.group, resource, storage); err != nil {
		return err
	}
	delete(entry.storage, resource)
	if len(entry.storage) == 0 {
		delete(m.thirdPartyResources, path)
		m.availableGroupManager.RemoveGroup(extensionsrest.GetThirdPartyGroupName(path))
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

	services := m.genericAPIServer.Handler.GoRestfulContainer.RegisteredWebServices()
	for ix := range services {
		root := services[ix].RootPath()
		if root == path || strings.HasPrefix(root, path+"/") {
			m.genericAPIServer.Handler.GoRestfulContainer.Remove(services[ix])
		}
	}
	return nil
}

func (m *ThirdPartyResourceServer) removeThirdPartyResourceData(group *metav1.APIGroup, resource string, registry *thirdpartyresourcedatastore.REST) error {
	// Freeze TPR data to prevent new writes via this apiserver process.
	// Other apiservers can still write. This is best-effort because there
	// are worse problems with TPR data than the possibility of going back
	// in time when migrating to CRD [citation needed].
	registry.Freeze()

	ctx := genericapirequest.NewContext()
	existingData, err := registry.List(ctx, nil)
	if err != nil {
		return err
	}
	list, ok := existingData.(*extensions.ThirdPartyResourceDataList)
	if !ok {
		return fmt.Errorf("expected a *ThirdPartyResourceDataList, got %T", existingData)
	}

	// Migrate TPR data to CRD if requested.
	gvk := schema.GroupVersionKind{Group: group.Name, Version: group.PreferredVersion.Version, Kind: registry.Kind()}
	migrationRequested, err := m.migrateThirdPartyResourceData(gvk, resource, list)
	if err != nil {
		// Migration is best-effort. Log and continue.
		utilruntime.HandleError(fmt.Errorf("failed to migrate TPR data: %v", err))
	}

	// Skip deletion of TPR data if migration was requested (whether or not it succeeded).
	// This leaves the etcd data around for rollback, and to avoid sending DELETE watch events.
	if migrationRequested {
		return nil
	}

	for i := range list.Items {
		item := &list.Items[i]

		// Use registry.Store.Delete() to bypass the frozen registry.Delete().
		if _, _, err := registry.Store.Delete(genericapirequest.WithNamespace(ctx, item.Namespace), item.Name, nil); err != nil {
			return err
		}
	}
	return nil
}

func (m *ThirdPartyResourceServer) findMatchingCRD(gvk schema.GroupVersionKind, resource string) (*apiextensions.CustomResourceDefinition, error) {
	// CustomResourceDefinitionList does not implement the protobuf marshalling interface.
	config := *m.genericAPIServer.LoopbackClientConfig
	config.ContentType = "application/json"
	crdClient, err := apiextensionsclient.NewForConfig(&config)
	if err != nil {
		return nil, fmt.Errorf("can't create apiextensions client: %v", err)
	}
	crdList, err := crdClient.CustomResourceDefinitions().List(metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("can't list CustomResourceDefinitions: %v", err)
	}
	for i := range crdList.Items {
		item := &crdList.Items[i]
		if item.Spec.Scope == apiextensions.NamespaceScoped &&
			item.Spec.Group == gvk.Group && item.Spec.Version == gvk.Version &&
			item.Status.AcceptedNames.Kind == gvk.Kind && item.Status.AcceptedNames.Plural == resource {
			return item, nil
		}
	}
	return nil, nil
}

func (m *ThirdPartyResourceServer) migrateThirdPartyResourceData(gvk schema.GroupVersionKind, resource string, dataList *extensions.ThirdPartyResourceDataList) (bool, error) {
	// A matching CustomResourceDefinition implies migration is requested.
	crd, err := m.findMatchingCRD(gvk, resource)
	if err != nil {
		return false, fmt.Errorf("can't determine if TPR should migrate: %v", err)
	}
	if crd == nil {
		// No migration requested.
		return false, nil
	}

	// Talk directly to CustomResource storage.
	// We have to bypass the API server because TPR is shadowing CRD at this point.
	storage := customresource.NewREST(
		schema.GroupResource{Group: crd.Spec.Group, Resource: crd.Spec.Names.Plural},
		schema.GroupVersionKind{Group: crd.Spec.Group, Version: crd.Spec.Version, Kind: crd.Spec.Names.ListKind},
		apiextensionsserver.UnstructuredCopier{},
		customresource.NewStrategy(discoveryclient.NewUnstructuredObjectTyper(nil), true, gvk),
		m.crdRESTOptionsGetter,
	)

	// Copy TPR data to CustomResource.
	var errs []error
	ctx := request.NewContext()
	for i := range dataList.Items {
		item := &dataList.Items[i]

		// Convert TPR data to Unstructured.
		objMap := make(map[string]interface{})
		if err := json.Unmarshal(item.Data, &objMap); err != nil {
			errs = append(errs, fmt.Errorf("can't unmarshal TPR data %q: %v", item.Name, err))
			continue
		}

		// Convert metadata to Unstructured and merge with data.
		// cf. thirdpartyresourcedata.encodeToJSON()
		metaMap := make(map[string]interface{})
		buf, err := json.Marshal(&item.ObjectMeta)
		if err != nil {
			errs = append(errs, fmt.Errorf("can't marshal metadata for TPR data %q: %v", item.Name, err))
			continue
		}
		if err := json.Unmarshal(buf, &metaMap); err != nil {
			errs = append(errs, fmt.Errorf("can't unmarshal TPR data %q: %v", item.Name, err))
			continue
		}
		// resourceVersion cannot be set when creating objects.
		delete(metaMap, "resourceVersion")
		objMap["metadata"] = metaMap

		// Store CustomResource.
		obj := &unstructured.Unstructured{Object: objMap}
		createCtx := request.WithNamespace(ctx, obj.GetNamespace())
		if _, err := storage.Create(createCtx, obj, false); err != nil {
			errs = append(errs, fmt.Errorf("can't create CustomResource for TPR data %q: %v", item.Name, err))
			continue
		}
	}
	return true, utilerrors.NewAggregate(errs)
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

func (m *ThirdPartyResourceServer) getExistingThirdPartyResources(path string) []metav1.APIResource {
	result := []metav1.APIResource{}
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	entry := m.thirdPartyResources[path]
	if entry != nil {
		for key, obj := range entry.storage {
			result = append(result, metav1.APIResource{
				Name:       key,
				Namespaced: true,
				Kind:       obj.Kind(),
				Verbs: metav1.Verbs([]string{
					"delete", "deletecollection", "get", "list", "patch", "create", "update", "watch",
				}),
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

func (m *ThirdPartyResourceServer) addThirdPartyResourceStorage(path, resource string, storage *thirdpartyresourcedatastore.REST, apiGroup metav1.APIGroup) {
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	entry, found := m.thirdPartyResources[path]
	if entry == nil {
		entry = &thirdPartyEntry{
			group:   apiGroup,
			storage: map[string]*thirdpartyresourcedatastore.REST{},
		}
		m.thirdPartyResources[path] = entry
	}
	entry.storage[resource] = storage
	if !found {
		m.availableGroupManager.AddGroup(apiGroup)
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
	plural, _ := meta.UnsafeGuessKindToResource(schema.GroupVersionKind{
		Group:   group,
		Version: rsrc.Versions[0].Name,
		Kind:    kind,
	})
	path := extensionsrest.MakeThirdPartyPath(group)

	groupVersion := metav1.GroupVersionForDiscovery{
		GroupVersion: group + "/" + rsrc.Versions[0].Name,
		Version:      rsrc.Versions[0].Name,
	}
	apiGroup := metav1.APIGroup{
		Name:             group,
		Versions:         []metav1.GroupVersionForDiscovery{groupVersion},
		PreferredVersion: groupVersion,
	}

	thirdparty := m.thirdpartyapi(group, kind, rsrc.Versions[0].Name, plural.Resource)

	// If storage exists, this group has already been added, just update
	// the group with the new API
	if m.hasThirdPartyGroupStorage(path) {
		m.addThirdPartyResourceStorage(path, plural.Resource, thirdparty.Storage[plural.Resource].(*thirdpartyresourcedatastore.REST), apiGroup)
		return thirdparty.UpdateREST(m.genericAPIServer.Handler.GoRestfulContainer)
	}

	if err := thirdparty.InstallREST(m.genericAPIServer.Handler.GoRestfulContainer); err != nil {
		glog.Errorf("Unable to setup thirdparty api: %v", err)
	}
	m.genericAPIServer.Handler.GoRestfulContainer.Add(discovery.NewAPIGroupHandler(api.Codecs, apiGroup, m.genericAPIServer.RequestContextMapper()).WebService())

	m.addThirdPartyResourceStorage(path, plural.Resource, thirdparty.Storage[plural.Resource].(*thirdpartyresourcedatastore.REST), apiGroup)
	api.Registry.AddThirdPartyAPIGroupVersions(schema.GroupVersion{Group: group, Version: rsrc.Versions[0].Name})
	return nil
}

func (m *ThirdPartyResourceServer) thirdpartyapi(group, kind, version, pluralResource string) *genericapi.APIGroupVersion {
	resourceStorage := thirdpartyresourcedatastore.NewREST(
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

	optionsExternalVersion := api.Registry.GroupOrDie(api.GroupName).GroupVersion
	internalVersion := schema.GroupVersion{Group: group, Version: runtime.APIVersionInternal}
	externalVersion := schema.GroupVersion{Group: group, Version: version}

	apiRoot := extensionsrest.MakeThirdPartyPath("")
	return &genericapi.APIGroupVersion{
		Root:         apiRoot,
		GroupVersion: externalVersion,

		Creater:         thirdpartyresourcedata.NewObjectCreator(group, version, api.Scheme),
		Convertor:       api.Scheme,
		Copier:          api.Scheme,
		Defaulter:       api.Scheme,
		Typer:           api.Scheme,
		UnsafeConvertor: api.Scheme,

		Mapper:                 thirdpartyresourcedata.NewMapper(api.Registry.GroupOrDie(extensions.GroupName).RESTMapper, kind, version, group),
		Linker:                 api.Registry.GroupOrDie(extensions.GroupName).SelfLinker,
		Storage:                storage,
		OptionsExternalVersion: &optionsExternalVersion,

		Serializer:     thirdpartyresourcedata.NewNegotiatedSerializer(api.Codecs, kind, externalVersion, internalVersion),
		ParameterCodec: thirdpartyresourcedata.NewThirdPartyParameterCodec(api.ParameterCodec),

		Context: m.genericAPIServer.RequestContextMapper(),

		MinRequestTimeout: m.genericAPIServer.MinRequestTimeout(),

		ResourceLister: dynamicLister{m, extensionsrest.MakeThirdPartyPath(group)},
	}
}
