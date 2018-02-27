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

package bulk

import (
	"net/url"
	"path"
	"reflect"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/runtime"
)

func generateSelfLink(obj runtime.Object, linker runtime.SelfLinker) (uri string, err error) {
	gvk := obj.GetObjectKind().GroupVersionKind()
	name, err := linker.Name(obj)
	if err != nil {
		return
	}
	namespace, err := linker.Namespace(obj)
	if err != nil {
		return
	}
	uri = "/" + path.Join("apis", gvk.Group, gvk.Version)
	if namespace != "" {
		uri += "/namespaces/" + namespace
	}
	uri += "/" + url.QueryEscape(gvk.Kind)
	if name != "" {
		uri += "/" + url.QueryEscape(name)
	}
	return
}

func fixupObjectSelfLink(obj runtime.Object, linker runtime.SelfLinker) runtime.Object {
	selfLink, err := generateSelfLink(obj, linker)
	if err == nil {
		err = linker.SetSelfLink(obj, selfLink)
	}
	if err != nil {
		glog.V(5).Infof("failed to set link for object %v: %v", reflect.TypeOf(obj), err)
	}
	return obj
}
