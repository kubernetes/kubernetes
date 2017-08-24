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
	bulkapi "k8s.io/apiserver/pkg/apis/bulk"
)

func generateSelfLink(s bulkapi.ResourceSelector) (uri string) {
	uri = "/" + path.Join("apis", s.Group, s.Version)
	if s.Namespace != "" {
		uri += "/namespaces/" + s.Namespace
	}
	uri += "/" + url.QueryEscape(s.Resource)
	if s.Name != "" {
		uri += "/" + url.QueryEscape(s.Name)
	}
	return
}

func fixupObjectSelfLink(s bulkapi.ResourceSelector, obj runtime.Object, linker runtime.SelfLinker) runtime.Object {
	selfLink := generateSelfLink(s)
	if err := linker.SetSelfLink(obj, selfLink); err != nil {
		glog.V(5).Infof("failed to set link for object %v: %v", reflect.TypeOf(obj), err)
	}
	return obj
}
