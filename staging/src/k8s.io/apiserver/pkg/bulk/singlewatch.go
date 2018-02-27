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
	"bytes"
	"fmt"
	"sync"

	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/watch"
	bulkapi "k8s.io/apiserver/pkg/apis/bulk"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
)

type singleWatch interface {
	StopWatch(stopReq *bulkapi.BulkRequest)
	RunWatchLoop()
	WatchID() string
}

// Holds state for single watch.
type singleWatchImpl struct {

	// Initial watch request (used to match with responses)
	startWatchRequest *bulkapi.BulkRequest

	// Identifies watched resource
	gvr bulkapi.GroupVersionResource

	// Root connection
	bulkConnection

	// Context
	context request.Context

	// Recheck authorization/admission
	permChecker permCheckFunc

	// User provided id (unique per connection)
	id string

	// Underlying watcher
	watcher watch.Interface

	// Encoder for embedded RawExtension/Object
	encoder runtime.Encoder

	// SelfLinker of watched resource.
	linker runtime.SelfLinker

	// Non-nil when watch was stopped by explicit request.
	stopWatchRequest *bulkapi.BulkRequest

	stopper sync.Once
}

func makeSingleWatch(ctx request.Context, bc bulkConnection, r *bulkapi.BulkRequest, groupInfo *LocalAPIGroupInfo) (singleWatch, error) {
	rs := &r.Watch.ItemSelector
	storage, ok := groupInfo.Storage[rs.GroupVersionResource.Resource]
	if !ok {
		return nil, fmt.Errorf("unsupported resource %v", rs.GroupVersionResource)
	}
	watcher, ok := storage.(rest.Watcher)
	if !ok {
		return nil, fmt.Errorf("storage doesn't support watching")
	}

	ctx = request.WithNamespace(ctx, rs.Namespace)
	permChecker := authorizationCheckerFactory{
		GroupInfo: groupInfo,
		Resource:  rs.GroupVersionResource,
		Name:      rs.Name,
		Namespace: rs.Namespace,
		Context:   ctx,
		Verb:      "watch"}.makeAuthorizationChecker()
	if err := permChecker(); err != nil {
		return nil, err
	}

	// TODO: implement more efficient way to watch multiple single resources (reuse watch.Interface).
	nameSelector := fields.OneTermEqualSelector("metadata.name", rs.Name)
	opts := &metainternalversion.ListOptions{FieldSelector: nameSelector}
	wifc, err := watcher.Watch(ctx, opts)
	if err != nil {
		return nil, fmt.Errorf("unable to watch: %v", err)
	}

	// FIXME: What serializer should we use here?
	gv := schema.GroupVersion{Group: rs.GroupVersionResource.Group, Version: rs.GroupVersionResource.Version}
	embeddedEncoder := groupInfo.Serializer.EncoderForVersion(bc.SerializerInfo().Serializer, gv)

	return &singleWatchImpl{
		startWatchRequest: r,
		bulkConnection:    bc,
		id:                r.Watch.WatchID,
		encoder:           embeddedEncoder,
		watcher:           wifc,
		permChecker:       permChecker,
		linker:            groupInfo.Linker,
		context:           ctx,
	}, nil
}

func makeSingleWatchList(ctx request.Context, bc bulkConnection, r *bulkapi.BulkRequest, groupInfo *LocalAPIGroupInfo) (singleWatch, error) {
	rs := &r.WatchList.ListSelector
	storage, ok := groupInfo.Storage[rs.GroupVersionResource.Resource]
	if !ok {
		return nil, fmt.Errorf("unsupported resource %v", rs.GroupVersionResource)
	}
	watcher, ok := storage.(rest.Watcher)
	if !ok {
		return nil, fmt.Errorf("storage doesn't support watching")
	}

	ctx = request.WithNamespace(ctx, rs.Namespace)
	permChecker := authorizationCheckerFactory{
		GroupInfo: groupInfo,
		Resource:  rs.GroupVersionResource,
		Namespace: rs.Namespace,
		Context:   ctx,
		Verb:      "watch"}.makeAuthorizationChecker()
	if err := permChecker(); err != nil {
		return nil, err
	}

	var opts *metainternalversion.ListOptions
	if rs.Options != nil {
		if err := metainternalversion.Convert_v1_ListOptions_To_internalversion_ListOptions(rs.Options, opts, nil); err != nil {
			return nil, err
		}
	}

	wifc, err := watcher.Watch(ctx, opts)
	if err != nil {
		return nil, fmt.Errorf("unable to watch: %v", err)
	}

	// FIXME: What serializer should we use here?
	gv := schema.GroupVersion{Group: rs.GroupVersionResource.Group, Version: rs.GroupVersionResource.Version}
	embeddedEncoder := groupInfo.Serializer.EncoderForVersion(bc.SerializerInfo().Serializer, gv)

	return &singleWatchImpl{
		startWatchRequest: r,
		bulkConnection:    bc,
		id:                r.WatchList.WatchID,
		encoder:           embeddedEncoder,
		watcher:           wifc,
		permChecker:       permChecker,
		linker:            groupInfo.Linker,
		context:           ctx,
	}, nil
}

func serializeEmbeddedObject(obj runtime.Object, e runtime.Encoder) (runtime.Object, error) {
	buf := &bytes.Buffer{}
	if err := e.Encode(obj, buf); err != nil {
		return nil, fmt.Errorf("unable to encode object: %v", err)
	}
	// ContentType is not required here because Raw already contains correctly encoded data
	return &runtime.Unknown{Raw: buf.Bytes()}, nil
}

func (w *singleWatchImpl) StopWatch(req *bulkapi.BulkRequest) {
	if req != nil {
		w.stopWatchRequest = req
	}
	w.stopper.Do(w.watcher.Stop)
}

func (w *singleWatchImpl) WatchID() string {
	return w.id
}

func (w *singleWatchImpl) RunWatchLoop() {
	w.bulkConnection.SendResponse(
		w.startWatchRequest,
		&bulkapi.BulkResponse{WatchStarted: &bulkapi.WatchStarted{WatchID: w.id}})

	defer func() {
		w.bulkConnection.SendResponse(
			w.stopWatchRequest,
			&bulkapi.BulkResponse{WatchStopped: &bulkapi.WatchStopped{WatchID: w.id}})
	}()
	ch := w.watcher.ResultChan()
	for {
		select {
		case <-w.context.Done():
			return
		case event, ok := <-ch:
			if !ok {
				return
			}
			err := w.permChecker()
			if err == nil {
				obj := fixupObjectSelfLink(w.gvr, event.Object, w.linker)
				event.Object, err = serializeEmbeddedObject(obj, w.encoder)
			}
			if err != nil {
				// Something is wrong - send 'WatchEvent' with error details & break.
				resp := &bulkapi.BulkResponse{
					WatchEvent: &bulkapi.BulkWatchEvent{
						WatchID: w.id,
						WatchEvent: watch.Event{
							Type:   watch.Error,
							Object: responsewriters.ErrorToAPIStatus(err)},
					}}
				if !w.bulkConnection.SendResponse(nil, resp) {
					utilruntime.HandleError(err)
				}
				return
			}
			resp := &bulkapi.BulkResponse{
				WatchEvent: &bulkapi.BulkWatchEvent{
					WatchID:    w.id,
					WatchEvent: event,
				}}
			w.bulkConnection.SendResponse(nil, resp)
		}
	}
}
