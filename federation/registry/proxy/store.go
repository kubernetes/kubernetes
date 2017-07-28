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

package proxy

import (
	"fmt"
	"sync"
	"time"

	"github.com/golang/glog"

	apiv1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
	apirest "k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/client-go/rest"
	fedv1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedclient "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api"
	kubeclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"reflect"
)

const (
	userAgentName = "federation-apiserver"
	retryTimes    = 3
)

type clusterInfo struct {
	cluster   *fedv1.Cluster
	clientset *kubeclient.Clientset
	err       error
	obj       runtime.Object
}

// Store implements apiserver/pkg/registry/rest.StandardStorage meant to be used in federation control plane.
// It provides access to the objects in the federation as well as that in the clusters by proxying the requests.
type Store struct {
	// NewFunc returns a new instance of the type this registry returns for a GET of a single object.
	NewFunc func() runtime.Object
	// NewListFunc returns a new list of the type this registry; it is the type returned when the resource is listed.
	NewListFunc func() runtime.Object
	// RESTClientFunc retrieve the client-go/rest.Interface for objects in this registry from a client to the cluster.
	RESTClientFunc func(kubeClientset kubeclient.Interface) rest.Interface
	// QualifiedResource is the pluralized name of the resource.
	QualifiedResource schema.GroupResource
	// NamespaceScoped specifies whether the object in this registry is namespaced.
	NamespaceScoped bool

	// FedClient is the federation control plane client to query for avaliable clusters.
	FedClient fedclient.Interface
	// FedStore is the storage of the objects in federation control plane itself; can be nil if no objects of this type in federation.
	FedStore apirest.StandardStorage
}

// New implements StandardStorage.New.
func (s *Store) New() runtime.Object {
	return s.NewFunc()
}

// NewList implements StandardStorage.NewList.
func (s *Store) NewList() runtime.Object {
	return s.NewListFunc()
}

// List implements StandardStorage.List.
func (s *Store) List(ctx request.Context, options *metainternalversion.ListOptions) (runtime.Object, error) {
	// not setting or empty cluster selector means federation control plane only
	if options == nil || options.ClusterSelector == nil || options.ClusterSelector.Empty() {
		if s.FedStore != nil {
			return s.FedStore.List(ctx, options)
		}
		return s.NewList(), nil
	}

	result := s.NewList()
	errs := []error{}

	// match cluster selector to federation control plane
	if options.ClusterSelector.Matches(fedv1.FederationControlPlaneLabels) {
		if s.FedStore != nil {
			obj, err := s.FedStore.List(ctx, options)
			if err != nil {
				errs = append(errs, err)
			} else {
				mergeList(result, obj)
			}
		}
	}

	userInfo, _ := request.UserFrom(ctx)
	clusterInfos, err := s.listClusters(options.ClusterSelector, userInfo)
	if err != nil {
		errs = append(errs, err)
		return nil, utilerrors.NewAggregate(errs)
	}

	var wg sync.WaitGroup
	wg.Add(len(clusterInfos))
	// call api of all the clusters
	for i := range clusterInfos {
		go func(c *clusterInfo) {
			if c.err == nil {
				optsv1 := metav1.ListOptions{}
				api.Scheme.Convert(options, &optsv1, nil)
				c.obj, c.err = s.tryList(ctx, &optsv1, c.cluster.Name, c.clientset)
			}
			wg.Done()
		}(&clusterInfos[i])
	}
	wg.Wait()

	// collect results
	for _, c := range clusterInfos {
		if c.err != nil {
			errs = append(errs, c.err)
		} else {
			mergeList(result, c.obj)
		}
	}
	if len(errs) != 0 {
		return nil, utilerrors.NewAggregate(errs)
	}

	return result, nil
}

// Get implements StandardStorage.Get.
func (s *Store) Get(ctx request.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {

	// from federation
	if len(options.ClusterName) == 0 {
		if s.FedStore == nil {
			return nil, errors.NewNotFound(s.QualifiedResource, name)
		}
		return s.FedStore.Get(ctx, name, options)
	}

	// from specified cluster
	userInfo, _ := request.UserFrom(ctx)
	clusterClient, err := s.getClusterClientByName(options.ClusterName, userInfo)
	if err != nil {
		glog.Warningf("error creating cluster client for cluster %s, err: %v", options.ClusterName, err)
		return nil, err
	}

	obj, err := s.tryGet(ctx, name, options.ClusterName, options, clusterClient)
	if err != nil {
		return nil, err
	}

	accessor, _ := meta.Accessor(obj)
	accessor.SetClusterName(options.ClusterName)
	return obj, nil
}

func (s *Store) tryList(ctx request.Context, opts *metav1.ListOptions, clusterName string, clientset kubeclient.Interface) (runtime.Object, error) {
	var err error
	delaySecs := 0
	for i := 0; i < retryTimes; i++ {
		time.Sleep(time.Duration(delaySecs) * time.Second)

		objs := s.NewList()
		err = s.RESTClientFunc(clientset).Get().
			NamespaceIfScoped(request.NamespaceValue(ctx), s.NamespaceScoped).
			Resource(s.QualifiedResource.Resource).
			VersionedParams(opts, metav1.ParameterCodec).
			Do().Into(objs)
		if err == nil {
			meta.EachListItem(objs, func(obj runtime.Object) error {
				accessor, _ := meta.Accessor(obj)
				accessor.SetClusterName(clusterName)
				return nil
			})
			return objs, nil
		}

		var retry bool
		delaySecs, retry = errors.SuggestsClientDelay(err)
		if !retry {
			return nil, err
		}
	}
	return nil, err
}

func (s *Store) tryGet(ctx request.Context, name string, clusterName string, options *metav1.GetOptions, clientset kubeclient.Interface) (runtime.Object, error) {
	var err error
	delaySecs := 0
	for i := 0; i < retryTimes; i++ {
		time.Sleep(time.Duration(delaySecs) * time.Second)

		obj := s.New()
		err := s.RESTClientFunc(clientset).Get().
			NamespaceIfScoped(request.NamespaceValue(ctx), s.NamespaceScoped).
			Name(name).
			Resource(s.QualifiedResource.Resource).
			VersionedParams(options, metav1.ParameterCodec).
			Do().Into(obj)
		if err == nil {
			accessor, _ := meta.Accessor(obj)
			accessor.SetClusterName(clusterName)
			return obj, nil
		}

		var retry bool
		delaySecs, retry = errors.SuggestsClientDelay(err)
		if !retry {
			return nil, err
		}
	}
	return nil, err
}

func (s *Store) listClusters(clusterSelector labels.Selector, userInfo user.Info) ([]clusterInfo, error) {
	clusterClients := []clusterInfo{}
	clusters, err := s.FedClient.Federation().Clusters().List(metav1.ListOptions{LabelSelector: clusterSelector.String()})
	if err != nil {
		glog.Warningf("error listing clusters, err: %v", err)
		return nil, err
	}
	for i := range clusters.Items {
		clientset, err := s.getClusterClient(&clusters.Items[i], userInfo)
		clusterClients = append(clusterClients, clusterInfo{
			cluster:   &clusters.Items[i],
			clientset: clientset,
			err:       err,
		})
	}

	return clusterClients, nil
}

func (s *Store) getClusterClientByName(clusterName string, userInfo user.Info) (*kubeclient.Clientset, error) {
	cluster, err := s.FedClient.Federation().Clusters().Get(clusterName, metav1.GetOptions{})
	if err != nil {
		glog.Warningf("error getting cluster %s, err: %v", clusterName, err)
		return nil, err
	}
	return s.getClusterClient(cluster, userInfo)
}

func (s *Store) getClusterClient(cluster *fedv1.Cluster, userInfo user.Info) (*kubeclient.Clientset, error) {
	if !isClusterReady(cluster) {
		glog.Warningf("cluster %s is not ready, conditions: %v", cluster.Name, cluster.Status.Conditions)
		return nil, fmt.Errorf("cluster %s is not ready", cluster.Name)
	}
	clusterConfig, err := fedutil.BuildClusterConfig(cluster)
	if err != nil {
		glog.Warningf("error build cluster config for cluster %s, err: %v", cluster.Name, err)
		return nil, err
	}
	// TODO: impersonate or subject access review?
	if userInfo != nil {
		clusterConfig.Impersonate = rest.ImpersonationConfig{
			UserName: userInfo.GetName(),
			Groups:   userInfo.GetGroups(),
			Extra:    userInfo.GetExtra(),
		}
	}
	return kubeclient.NewForConfig(rest.AddUserAgent(clusterConfig, userAgentName))
}

func isClusterReady(cluster *fedv1.Cluster) bool {
	for _, condition := range cluster.Status.Conditions {
		if condition.Type == fedv1.ClusterReady {
			if condition.Status == apiv1.ConditionTrue {
				return true
			}
		}
	}
	return false
}

// write ops, and watch, are not allowed for underlying clusters, just delegate them to the fed store

// Create implements StandardStorage.Create.
func (s *Store) Create(ctx request.Context, obj runtime.Object, includeUninitialized bool) (runtime.Object, error) {
	if s.FedStore == nil {
		return nil, errors.NewMethodNotSupported(s.QualifiedResource, "create")
	}
	return s.FedStore.Create(ctx, obj, includeUninitialized)
}

// Update implements StandardStorage.Update.
func (s *Store) Update(ctx request.Context, name string, objInfo apirest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	if s.FedStore == nil {
		return nil, false, errors.NewMethodNotSupported(s.QualifiedResource, "update")
	}
	return s.FedStore.Update(ctx, name, objInfo)

}

// Delete implements StandardStorage.Delete.
func (s *Store) Delete(ctx request.Context, name string, options *metav1.DeleteOptions) (runtime.Object, bool, error) {
	if s.FedStore == nil {
		return nil, false, errors.NewMethodNotSupported(s.QualifiedResource, "delete")
	}
	return s.FedStore.Delete(ctx, name, options)
}

// DeleteCollection implements StandardStorage.DeleteCollection.
func (s *Store) DeleteCollection(ctx request.Context, options *metav1.DeleteOptions, listOptions *metainternalversion.ListOptions) (runtime.Object, error) {
	if s.FedStore == nil {
		return nil, errors.NewMethodNotSupported(s.QualifiedResource, "delete-collection")
	}
	return s.FedStore.DeleteCollection(ctx, options, listOptions)
}

// Watch implements StandardStorage.Watch.
func (s *Store) Watch(ctx request.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	if s.FedStore == nil {
		return nil, errors.NewMethodNotSupported(s.QualifiedResource, "watch")
	}
	return s.FedStore.Watch(ctx, options)
}

func mergeList(target runtime.Object, src runtime.Object) error {
	targetPtr, err := meta.GetItemsPtr(target)
	if err != nil {
		return err
	}
	targetItems, err := conversion.EnforcePtr(targetPtr)
	if err != nil {
		return err
	}
	srcPtr, err := meta.GetItemsPtr(src)
	if err != nil {
		return err
	}
	srcItems, err := conversion.EnforcePtr(srcPtr)
	if err != nil {
		return err
	}

	merged := reflect.AppendSlice(targetItems, srcItems)
	targetItems.Set(merged)
	return nil
}
