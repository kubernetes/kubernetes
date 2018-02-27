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
	"fmt"

	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	bulkapi "k8s.io/apiserver/pkg/apis/bulk"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
)

func performSingleGet(ctx request.Context, bc bulkConnection, r *bulkapi.ClientMessage, gi *LocalAPIGroupInfo) {
	rs := &r.Get.ItemSelector
	storage, ok := gi.Storage[rs.GroupVersionResource.Resource]
	if !ok {
		bc.SendResponse(r, errorResponse(fmt.Errorf("unsupported resource %v", rs.GroupVersionResource.Resource)))
		return
	}
	getter, ok := storage.(rest.Getter)
	if !ok {
		bc.SendResponse(r, errorResponse(fmt.Errorf("storage doesn't support getting")))
		return
	}

	ctx = request.WithNamespace(ctx, rs.Namespace)
	err := authorizationCheckerFactory{
		GroupInfo: gi,
		Resource:  rs.GroupVersionResource,
		Name:      rs.Name,
		Context:   ctx,
		Verb:      "get"}.checkAuthorization()
	if err != nil {
		bc.SendResponse(r, errorResponse(err))
		return
	}

	gv := schema.GroupVersion{Group: rs.GroupVersionResource.Group, Version: rs.GroupVersionResource.Version}
	embeddedEncoder := gi.Serializer.EncoderForVersion(bc.SerializerInfo().Serializer, gv)

	var opts metav1.GetOptions
	if rs.Options != nil {
		opts = *rs.Options
	}
	obj, err := getter.Get(ctx, rs.Name, &opts)
	if err != nil {
		bc.SendResponse(r, errorResponse(err))
		return
	}
	item, err := serializeEmbeddedObject(obj, embeddedEncoder)
	if err != nil {
		bc.SendResponse(r, errorResponse(err))
		return
	}
	bc.SendResponse(r, &bulkapi.ServerMessage{GetResult: &bulkapi.GetResult{Item: item}})
}

func performSingleList(ctx request.Context, bc bulkConnection, r *bulkapi.ClientMessage, gi *LocalAPIGroupInfo) {
	rs := &r.List.ListSelector
	storage, ok := gi.Storage[rs.GroupVersionResource.Resource]
	if !ok {
		bc.SendResponse(r, errorResponse(fmt.Errorf("unsupported resource %v", rs.GroupVersionResource.Resource)))
		return
	}
	lister, ok := storage.(rest.Lister)
	if !ok {
		bc.SendResponse(r, errorResponse(fmt.Errorf("storage doesn't support listing")))
		return
	}

	ctx = request.WithNamespace(ctx, rs.Namespace)
	err := authorizationCheckerFactory{
		GroupInfo: gi,
		Resource:  rs.GroupVersionResource,
		Context:   ctx,
		Verb:      "list"}.checkAuthorization()
	if err != nil {
		bc.SendResponse(r, errorResponse(err))
		return
	}

	var opts metainternalversion.ListOptions
	if rs.Options != nil {
		if err = metainternalversion.Convert_v1_ListOptions_To_internalversion_ListOptions(rs.Options, &opts, nil); err != nil {
			bc.SendResponse(r, errorResponse(err))
			return
		}
	}

	gv := schema.GroupVersion{Group: rs.GroupVersionResource.Group, Version: rs.GroupVersionResource.Version}
	embeddedEncoder := gi.Serializer.EncoderForVersion(bc.SerializerInfo().Serializer, gv)

	listRaw, err := lister.List(ctx, &opts)
	if err != nil {
		bc.SendResponse(r, errorResponse(err))
		return
	}
	list, err := serializeEmbeddedObject(listRaw, embeddedEncoder)
	if err != nil {
		bc.SendResponse(r, errorResponse(err))
		return
	}
	bc.SendResponse(r, &bulkapi.ServerMessage{ListResult: &bulkapi.ListResult{List: list}})
}
