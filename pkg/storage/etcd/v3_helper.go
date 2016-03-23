/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package etcd

import (
	"errors"
	"path"
	"reflect"
	"strings"

	"github.com/coreos/etcd/clientv3"
	"github.com/golang/glog"
	"golang.org/x/net/context"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/watch"
)

type v3Helper struct {
	// etcd interfaces
	client *clientv3.Client

	codec      runtime.Codec
	versioner  storage.Versioner
	pathPrefix string
}

func newV3Helper(c *clientv3.Client, codec runtime.Codec, prefix string) *v3Helper {
	return &v3Helper{
		client:     c,
		versioner:  APIObjectVersioner{},
		codec:      codec,
		pathPrefix: prefix,
	}
}

func (h *v3Helper) Backends(ctx context.Context) []string {
	if ctx == nil {
		glog.Errorf("Context is nil")
		ctx = context.TODO()
	}
	resp, err := h.client.MemberList(ctx)
	if err != nil {
		glog.Errorf("Error obtaining etcd members list: %q", err)
		return nil
	}
	var mlist []string
	for _, member := range resp.Members {
		mlist = append(mlist, member.ClientURLs...)
	}
	return mlist
}

func (h *v3Helper) Codec() runtime.Codec {
	return h.codec
}

func (h *v3Helper) Versioner() storage.Versioner {
	return h.versioner
}

func (h *v3Helper) Set(ctx context.Context, key string, obj, out runtime.Object, ttl uint64) error {
	if ctx == nil {
		glog.Errorf("Context is nil")
		ctx = context.TODO()
	}

	rev, err := extractRevision(h.versioner, obj)
	if err != nil {
		return err
	}

	data, err := runtime.Encode(h.codec, obj)
	if err != nil {
		return err
	}
	key = keyWithPrefix(h.pathPrefix, key)

	// Note: if rev = 0, it is equivalent to create.
	txnResp, err := h.client.KV.Txn(ctx).If(
		clientv3.Compare(clientv3.ModifiedRevision(key), "=", rev),
	).Then(
		clientv3.OpPut(key, string(data)),
	).Commit()
	if err != nil {
		return err
	}
	if !txnResp.Succeeded {
		return storage.NewResourceVersionConflictsError(key, rev)
	}

	if out != nil {
		// get the response to the put request
		putResp := txnResp.Responses[0].GetResponsePut()
		if _, err := conversion.EnforcePtr(out); err != nil {
			panic("unable to convert output object to pointer")
		}
		return decode(h.codec, h.versioner, data, out, putResp.Header.Revision)
	}

	return nil
}

func (h *v3Helper) Get(ctx context.Context, key string, out runtime.Object, ignoreNotFound bool) error {
	if ctx == nil {
		glog.Errorf("Context is nil")
		ctx = context.TODO()
	}
	key = keyWithPrefix(h.pathPrefix, key)
	getResp, err := h.client.KV.Get(ctx, key)
	if err != nil {
		return err
	}

	// if not found:
	//   if ignoreNotFound: set objPtr to be zero value
	//   else: return not found error
	if len(getResp.Kvs) == 0 {
		if ignoreNotFound {
			return setZeroValue(out)
		} else {
			return storage.NewKeyNotFoundError(key, 0)
		}
	}

	kv := getResp.Kvs[0]
	return decode(h.codec, h.versioner, kv.Value, out, kv.ModRevision)
}

func (h *v3Helper) Create(ctx context.Context, key string, obj, out runtime.Object, ttl uint64) error {
	panic("unimplemented")
}

func (h *v3Helper) Delete(ctx context.Context, key string, out runtime.Object) error {
	panic("unimplemented")
}

func (h *v3Helper) GetToList(ctx context.Context, key string, filter storage.FilterFunc, listObj runtime.Object) error {
	panic("unimplemented")
}

func (h *v3Helper) List(ctx context.Context, key, resourceVersion string, filter storage.FilterFunc, listObj runtime.Object) error {
	panic("unimplemented")
}

func (h *v3Helper) GuaranteedUpdate(ctx context.Context, key string, ptrToType runtime.Object, ignoreNotFound bool, tryUpdate storage.UpdateFunc) error {
	panic("unimplemented")
}

func (h *v3Helper) Watch(ctx context.Context, key string, resourceVersion string, filter storage.FilterFunc) (watch.Interface, error) {
	panic("unimplemented")
}

func (h *v3Helper) WatchList(ctx context.Context, key string, resourceVersion string, filter storage.FilterFunc) (watch.Interface, error) {
	panic("unimplemented")
}

func keyWithPrefix(prefix, key string) string {
	if strings.HasPrefix(key, prefix) {
		return key
	}
	return path.Join(prefix, key)
}

func extractRevision(versioner storage.Versioner, obj runtime.Object) (int64, error) {
	rev, err := versioner.ObjectResourceVersion(obj)
	if err != nil {
		return 0, errors.New("couldn't get resourceVersion from object")
	}
	if rev != 0 {
		// We cannot store object with resourceVersion in etcd, we need to clear it here.
		if err := versioner.UpdateObject(obj, nil, 0); err != nil {
			return 0, errors.New("resourceVersion cannot be set on objects store in etcd")
		}
	}
	return int64(rev), nil
}

func decode(codec runtime.Codec, versioner storage.Versioner, body []byte, objPtr runtime.Object, rev int64) error {
	_, _, err := codec.Decode(body, nil, objPtr)
	if err != nil {
		return err
	}
	versioner.UpdateObject(objPtr, nil, uint64(rev))
	return nil
}

func setZeroValue(objPtr runtime.Object) error {
	// TODO: We shouldn't use "reflect" but consider supporting ZeroValue() method in Object.
	v, err := conversion.EnforcePtr(objPtr)
	if err != nil {
		return err
	}
	v.Set(reflect.Zero(v.Type()))
	return nil
}
