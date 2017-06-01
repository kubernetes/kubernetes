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

// Package app does all of the work necessary to create a Kubernetes
// APIServer by binding together the API, master and APIServer infrastructure.
// It can be configured and called directly or via the hyperkube framework.
package app

import (
	"fmt"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/runtime"
	//"k8s.io/apimachinery/pkg/runtime/schema"
	"bytes"
	"context"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/storagebackend/factory"
	"k8s.io/kubernetes/cmd/kubebackup/app/options"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	apisextensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	apisrbac "k8s.io/kubernetes/pkg/apis/rbac/v1beta1"
	apisstorage "k8s.io/kubernetes/pkg/apis/storage/v1"

	"archive/zip"
	"io/ioutil"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	_ "k8s.io/kubernetes/pkg/apis/rbac/install"
	_ "k8s.io/kubernetes/pkg/apis/storage/install"
	"os"
	"path/filepath"
	"strings"
)

// Run runs the specified APIServer.  This should never exit.
func Run(s *options.ServerRunOptions) error {
	config, _, err := BuildMasterConfig(s)
	if err != nil {
		return err
	}

	yaml, ok := runtime.SerializerInfoForMediaType(api.Codecs.SupportedMediaTypes(), "application/yaml")
	if !ok {
		glog.Fatalf("no YAML serializer registered")
	}

	dumper := &Dumper{
		encoder:        yaml.Serializer,
		storageFactory: config.StorageFactory,
	}

	if s.RestoreFromFile != "" {
		glog.Infof("Reading from %s\n", s.RestoreFromFile)

		reader, err := zip.OpenReader(s.RestoreFromFile)
		if err != nil {
			return fmt.Errorf("error opening RestoreFromFile %q: %v", s.RestoreFromFile, err)
		}
		defer reader.Close()

		if err := dumper.Restore(reader); err != nil {
			return err
		}

		// If we are restoring, we don't backup
		return nil
	}

	if s.DestFile != "" {
		f, err := os.Create(s.DestFile)
		if err != nil {
			return fmt.Errorf("error opening DestFile %q: %v", s.DestFile, err)
		}
		defer f.Close()

		glog.Infof("Writing to %s\n", s.DestFile)

		dumper.zipfile = zip.NewWriter(f)
	}

	if err := dumper.Dump(); err != nil {
		return err
	}

	if dumper.zipfile != nil {
		// Make sure to check the error on Close.
		if err := dumper.zipfile.Close(); err != nil {
			return fmt.Errorf("error closing zipfile: %v", err)
		}
	}

	if s.DestFile != "" {
		abs, err := filepath.Abs(s.DestFile)
		if err != nil {
			return fmt.Errorf("error computing absolute path for %q: %v", s.DestFile, err)
		}

		fmt.Printf("Wrote to %s\n", abs)
	} else {
		fmt.Printf("#Finished")
	}

	return nil
}

type Dumper struct {
	namespaces     []string
	storageFactory serverstorage.StorageFactory
	encoder        runtime.Encoder
	zipfile        *zip.Writer
}

func (d *Dumper) Dump() error {
	var namespaces []string
	{
		list := &v1.NamespaceList{}
		if err := d.getAll("namespaces", "", list); err != nil {
			return err
		}
		for i := range list.Items {
			obj := &list.Items[i]
			if err := d.backupObject("namespaces", obj, v1.SchemeGroupVersion); err != nil {
				return err
			}
			namespaces = append(namespaces, obj.Name)
		}
	}
	d.namespaces = namespaces

	// TODO: Any way to eliminate the copy pasta here?
	// config.StorageFactory.ResourcePrefix seems to be the human string
	// api.Scheme.AllKnownTypes() seems to get us everything
	// Not sure how to identify lists or enumerate through Items

	{
		list := &apisstorage.StorageClassList{}
		if err := d.getAll("storageclasses", "", list); err != nil {
			return err
		}
		for i := range list.Items {
			obj := &list.Items[i]
			if err := d.backupObject("storageclasses", obj, apisstorage.SchemeGroupVersion); err != nil {
				return err
			}
		}
	}

	{
		list := &apisrbac.ClusterRoleList{}
		if err := d.getAll("clusterroles", "", list); err != nil {
			return err
		}
		for i := range list.Items {
			obj := &list.Items[i]
			if err := d.backupObject("clusterroles", obj, apisrbac.SchemeGroupVersion); err != nil {
				return err
			}
		}
	}

	{
		list := &v1.NodeList{}
		if err := d.getAll("minions", "", list); err != nil {
			return err
		}
		for i := range list.Items {
			obj := &list.Items[i]
			if err := d.backupObject("minions", obj, v1.SchemeGroupVersion); err != nil {
				return err
			}
		}
	}

	// TODO: Will ranges be repaired?

	for _, namespace := range namespaces {
		{
			list := &v1.PodList{}
			if err := d.getAll("pods", namespace, list); err != nil {
				return err
			}
			for i := range list.Items {
				obj := &list.Items[i]
				if err := d.backupObject("pods", obj, v1.SchemeGroupVersion); err != nil {
					return err
				}
			}
		}

		{
			list := &v1.ServiceList{}
			if err := d.getAll("services", namespace, list); err != nil {
				return err
			}
			for i := range list.Items {
				obj := &list.Items[i]
				if err := d.backupObject("services", obj, v1.SchemeGroupVersion); err != nil {
					return err
				}
			}
		}

		{
			list := &v1.SecretList{}
			if err := d.getAll("secrets", namespace, list); err != nil {
				return err
			}
			for i := range list.Items {
				obj := &list.Items[i]
				if err := d.backupObject("secrets", obj, v1.SchemeGroupVersion); err != nil {
					return err
				}
			}
		}

		{
			list := &v1.LimitRangeList{}
			if err := d.getAll("limitranges", namespace, list); err != nil {
				return err
			}
			for i := range list.Items {
				obj := &list.Items[i]
				if err := d.backupObject("limitranges", obj, v1.SchemeGroupVersion); err != nil {
					return err
				}
			}
		}

		{
			list := &apisextensions.DeploymentList{}
			if err := d.getAll("deployments", namespace, list); err != nil {
				return err
			}
			for i := range list.Items {
				obj := &list.Items[i]
				if err := d.backupObject("deployments", obj, apisextensions.SchemeGroupVersion); err != nil {
					return err
				}
			}
		}

		{
			list := &v1.ServiceAccountList{}
			if err := d.getAll("serviceaccounts", namespace, list); err != nil {
				return err
			}
			for i := range list.Items {
				obj := &list.Items[i]
				if err := d.backupObject("serviceaccounts", obj, v1.SchemeGroupVersion); err != nil {
					return err
				}
			}
		}

		{
			list := &apisrbac.ClusterRoleBindingList{}
			if err := d.getAll("clusterrolebindings", namespace, list); err != nil {
				return err
			}
			for i := range list.Items {
				obj := &list.Items[i]
				if err := d.backupObject("clusterrolebindings", obj, apisrbac.SchemeGroupVersion); err != nil {
					return err
				}
			}
		}

		{
			list := &v1.ConfigMapList{}
			if err := d.getAll("configmaps", namespace, list); err != nil {
				return err
			}
			for i := range list.Items {
				obj := &list.Items[i]
				if err := d.backupObject("configmaps", obj, v1.SchemeGroupVersion); err != nil {
					return err
				}
			}
		}

		{
			list := &apisextensions.ReplicaSetList{}
			if err := d.getAll("replicasets", namespace, list); err != nil {
				return err
			}
			for i := range list.Items {
				obj := &list.Items[i]
				if err := d.backupObject("replicasets", obj, apisextensions.SchemeGroupVersion); err != nil {
					return err
				}
			}
		}
	}

	// TODO: LOTS MORE TYPES!

	return nil
}

func (d *Dumper) getAll(resource string, namespace string, list runtime.Object) error {
	ctx := context.Background()
	resourceVersion := ""
	selectionPredicate := storage.Everything

	glog.Infof("Backing up all %q", resource)

	k := list.GetObjectKind()
	gvk := k.GroupVersionKind()

	groupResource := schema.GroupResource{
		Group:    gvk.Group,
		Resource: resource,
	}

	store, err := d.getStore(groupResource)
	if err != nil {
		return err
	}

	key, err := d.getListKey(groupResource, namespace)
	if err != nil {
		return err
	}

	if err := store.GetToList(ctx, key, resourceVersion, selectionPredicate, list); err != nil {
		return fmt.Errorf("error getting objects of type %v: %v", key, err)
	}

	return nil
}

func (d *Dumper) backupObject(resource string, obj runtime.Object, gv runtime.GroupVersioner) error {
	encoder := api.Codecs.EncoderForVersion(d.encoder, gv)

	var w bytes.Buffer
	err := encoder.Encode(obj, &w)
	if err != nil {
		return fmt.Errorf("error encoding %T: %v", obj, err)
	}

	accessor, err := meta.Accessor(obj)
	if err != nil {
		return fmt.Errorf("error getting accessor: %v", err)
	}

	namespace := accessor.GetNamespace()
	name := accessor.GetName()

	path := resource
	if namespace != "" {
		path += "/" + namespace
	}
	path += "/" + name

	if d.zipfile != nil {
		f, err := d.zipfile.Create(path)
		if err != nil {
			return fmt.Errorf("error creating zipfile entry %q: %v", path, err)
		}
		_, err = f.Write(w.Bytes())
		if err != nil {
			return fmt.Errorf("error writing zipfile entry %q: %v", path, err)
		}
	} else {
		fmt.Printf("#%s:\n%s\n---\n\n", path, w.String())
	}

	return nil
}

func (d *Dumper) Restore(z *zip.ReadCloser) error {
	for _, f := range z.File {
		glog.Infof("processing %s", f.Name)

		r, err := f.Open()
		if err != nil {
			return fmt.Errorf("error opening %s: %v", f.Name, err)
		}
		data, err := ioutil.ReadAll(r)
		r.Close()
		if err != nil {
			return fmt.Errorf("error reading %s: %v", f.Name, err)
		}

		if err := d.restoreObject(f.Name, data); err != nil {
			return fmt.Errorf("error restoring %s: %v", f.Name, err)
		}
	}
	return nil
}

func (d *Dumper) getStore(groupResource schema.GroupResource) (storage.Interface, error) {
	// TODO: CACHE ... MUST CACHE!
	storageConfig, err := d.storageFactory.NewConfig(groupResource)
	if err != nil {
		return nil, fmt.Errorf("error getting storage config for %v: %v", groupResource, err)
	}
	store, _, err := factory.Create(*storageConfig)
	if err != nil {
		return nil, err
	}
	return store, nil
}

func (d *Dumper) getObjectKey(resource string, obj runtime.Object) (string, error) {
	k := obj.GetObjectKind()
	gvk := k.GroupVersionKind()

	groupResource := schema.GroupResource{
		Group:    gvk.Group,
		Resource: resource,
	}

	accessor, err := meta.Accessor(obj)
	if err != nil {
		return "", fmt.Errorf("error getting accessor for object: %v", err)
	}

	prefix := d.storageFactory.ResourcePrefix(groupResource)
	namespace := accessor.GetNamespace()
	name := accessor.GetName()

	key := prefix + "/" + namespace + "/" + name
	if namespace == "" {
		key = prefix + "/" + name
	}

	return key, nil
}

func (d *Dumper) getListKey(groupResource schema.GroupResource, namespace string) (string, error) {
	prefix := d.storageFactory.ResourcePrefix(groupResource)

	key := prefix + "/" + namespace
	if namespace == "" {
		key = prefix
	}

	return key, nil
}

func (d *Dumper) restoreObject(zipPath string, data []byte) error {
	decoder := api.Codecs.UniversalDeserializer()
	obj, gvk, err := decoder.Decode(data, nil, nil)
	if err != nil {
		return fmt.Errorf("error decoding %s: %v", zipPath, err)
	}
	glog.V(8).Infof("GVK %v", gvk)

	tokens := strings.Split(zipPath, "/")
	resource := tokens[0]

	groupResource := schema.GroupResource{
		Group:    gvk.Group,
		Resource: resource,
	}
	store, err := d.getStore(groupResource)
	if err != nil {
		return err
	}

	accessor, err := meta.Accessor(obj)
	if err != nil {
		return fmt.Errorf("error getting accessor for %v: %v", gvk, err)
	}

	key, err := d.getObjectKey(resource, obj)
	if err != nil {
		return err
	}

	accessor.SetResourceVersion("")

	//typeAccessor, err := meta.TypeAccessor(obj)
	////typeAccessor.SetAPIVersion()
	//typeAccessor.SetKind(gvk.Kind)

	ctx := context.Background()

	glog.Infof("Restoring object %q", key)

	//ttl := uint64(0) // forever
	//err = d.store.Create(ctx, key, obj, obj, ttl)
	//if err == nil {
	//	return nil
	//}
	//if !storage.IsNodeExist(err) {
	//	return fmt.Errorf("error creating objects %q: %v", key, err)
	//}

	//typeObj := accessor.
	ignoreNotFound := true
	err = store.GuaranteedUpdate(
		ctx, key, obj, ignoreNotFound, nil,
		func(input runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
			return obj, nil, nil
		})
	if err != nil {
		return fmt.Errorf("error creating object %q: %v", key, err)
	}

	return nil
}
