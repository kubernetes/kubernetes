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
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/storagebackend/factory"
	"k8s.io/kubernetes/cmd/kubebackup/app/options"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions"
	apisrbac "k8s.io/kubernetes/pkg/apis/rbac"
	apisstorage "k8s.io/kubernetes/pkg/apis/storage"

	"archive/zip"
	_ "k8s.io/kubernetes/pkg/apis/rbac/install"
	_ "k8s.io/kubernetes/pkg/apis/storage/install"
	"os"
	"path/filepath"
)

// Run runs the specified APIServer.  This should never exit.
func Run(s *options.ServerRunOptions) error {
	config, _, err := BuildMasterConfig(s)
	if err != nil {
		return err
	}

	groupResource := api.Resource("registry")
	storageConfig, err := config.StorageFactory.NewConfig(groupResource)
	if err != nil {
		return fmt.Errorf("error getting storage config for %v: %v", groupResource, err)
	}
	store, _, err := factory.Create(*storageConfig)
	if err != nil {
		return err
	}

	yaml, ok := runtime.SerializerInfoForMediaType(api.Codecs.SupportedMediaTypes(), "application/yaml")
	if !ok {
		glog.Fatalf("no YAML serializer registered")
	}

	dumper := &Dumper{
		store:   store,
		encoder: yaml.Serializer,
	}

	if s.DestFile != "" {
		f, err := os.Create(s.DestFile)
		if err != nil {
			return fmt.Errorf("error opening zipfile %q: %v", s.DestFile, err)
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
	namespaces []string
	store      storage.Interface
	encoder    runtime.Encoder
	zipfile    *zip.Writer
}

func (d *Dumper) Dump() error {
	var namespaces []string
	{
		list := &v1.NamespaceList{}
		if err := d.getAll("namespaces", list); err != nil {
			return err
		}
		for i := range list.Items {
			obj := &list.Items[i]
			if err := d.write("namespaces/"+obj.Name, obj, v1.SchemeGroupVersion); err != nil {
				return err
			}
			namespaces = append(namespaces, obj.Name)
		}
	}
	d.namespaces = namespaces

	// TODO: Any way to eliminate the copy pasta here?

	{
		list := &apisstorage.StorageClassList{}
		if err := d.getAll("storageclasses", list); err != nil {
			return err
		}
		for i := range list.Items {
			obj := &list.Items[i]
			if err := d.write("storageclasses/"+obj.Name, obj, apisstorage.SchemeGroupVersion); err != nil {
				return err
			}
		}
	}

	{
		list := &apisrbac.ClusterRoleList{}
		if err := d.getAll("clusterroles", list); err != nil {
			return err
		}
		for i := range list.Items {
			obj := &list.Items[i]
			if err := d.write("clusterroles/"+obj.Name, obj, apisrbac.SchemeGroupVersion); err != nil {
				return err
			}
		}
	}

	{
		list := &v1.NodeList{}
		if err := d.getAll("minions", list); err != nil {
			return err
		}
		for i := range list.Items {
			obj := &list.Items[i]
			if err := d.write("minions/"+obj.Name, obj, v1.SchemeGroupVersion); err != nil {
				return err
			}
		}
	}

	// TODO: Will ranges be repaired?

	for _, namespace := range namespaces {
		{
			list := &v1.PodList{}
			if err := d.getAll("pods/"+namespace, list); err != nil {
				return err
			}
			for i := range list.Items {
				obj := &list.Items[i]
				if err := d.write("pods/"+namespace+"/"+obj.Name, obj, v1.SchemeGroupVersion); err != nil {
					return err
				}
			}
		}

		{
			list := &v1.ServiceList{}
			if err := d.getAll("services/"+namespace, list); err != nil {
				return err
			}
			for i := range list.Items {
				obj := &list.Items[i]
				if err := d.write("services/"+namespace+"/"+obj.Name, obj, v1.SchemeGroupVersion); err != nil {
					return err
				}
			}
		}

		{
			list := &v1.SecretList{}
			if err := d.getAll("secrets/"+namespace, list); err != nil {
				return err
			}
			for i := range list.Items {
				obj := &list.Items[i]
				if err := d.write("secrets/"+namespace+"/"+obj.Name, obj, v1.SchemeGroupVersion); err != nil {
					return err
				}
			}
		}
		{
			list := &v1.PodList{}
			if err := d.getAll("pods/"+namespace, list); err != nil {
				return err
			}
			for i := range list.Items {
				obj := &list.Items[i]
				if err := d.write("pods/"+namespace+"/"+obj.Name, obj, v1.SchemeGroupVersion); err != nil {
					return err
				}
			}
		}

		{
			list := &v1.LimitRangeList{}
			if err := d.getAll("limitranges/"+namespace, list); err != nil {
				return err
			}
			for i := range list.Items {
				obj := &list.Items[i]
				if err := d.write("limitranges/"+namespace+"/"+obj.Name, obj, v1.SchemeGroupVersion); err != nil {
					return err
				}
			}
		}

		{
			list := &extensions.DeploymentList{}
			if err := d.getAll("deployments/"+namespace, list); err != nil {
				return err
			}
			for i := range list.Items {
				obj := &list.Items[i]
				if err := d.write("deployments/"+namespace+"/"+obj.Name, obj, extensions.SchemeGroupVersion); err != nil {
					return err
				}
			}
		}

		{
			list := &v1.ServiceAccountList{}
			if err := d.getAll("serviceaccounts/"+namespace, list); err != nil {
				return err
			}
			for i := range list.Items {
				obj := &list.Items[i]
				if err := d.write("serviceaccounts/"+namespace+"/"+obj.Name, obj, v1.SchemeGroupVersion); err != nil {
					return err
				}
			}
		}

		{
			list := &apisrbac.ClusterRoleBindingList{}
			if err := d.getAll("clusterrolebindings/"+namespace, list); err != nil {
				return err
			}
			for i := range list.Items {
				obj := &list.Items[i]
				if err := d.write("clusterrolebindings/"+namespace+"/"+obj.Name, obj, apisrbac.SchemeGroupVersion); err != nil {
					return err
				}
			}
		}

		{
			list := &v1.ConfigMapList{}
			if err := d.getAll("configmaps/"+namespace, list); err != nil {
				return err
			}
			for i := range list.Items {
				obj := &list.Items[i]
				if err := d.write("configmaps/"+namespace+"/"+obj.Name, obj, v1.SchemeGroupVersion); err != nil {
					return err
				}
			}
		}

		{
			list := &extensions.ReplicaSetList{}
			if err := d.getAll("replicasets/"+namespace, list); err != nil {
				return err
			}
			for i := range list.Items {
				obj := &list.Items[i]
				if err := d.write("replicasets/"+namespace+"/"+obj.Name, obj, extensions.SchemeGroupVersion); err != nil {
					return err
				}
			}
		}
	}

	// TODO: LOTS MORE TYPES!

	return nil
}

func (d *Dumper) getAll(key string, list runtime.Object) error {
	ctx := context.Background()
	resourceVersion := ""
	selectionPredicate := storage.Everything

	glog.Infof("Listing all %q", key)

	if err := d.store.GetToList(ctx, key, resourceVersion, selectionPredicate, list); err != nil {
		return fmt.Errorf("error getting objects of type %v: %v", key, err)
	}

	return nil
}

func (d *Dumper) write(path string, obj runtime.Object, gv runtime.GroupVersioner) error {
	encoder := api.Codecs.EncoderForVersion(d.encoder, gv)

	var w bytes.Buffer
	err := encoder.Encode(obj, &w)
	if err != nil {
		return fmt.Errorf("error encoding %T: %v", obj, err)
	}

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
