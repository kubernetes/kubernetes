/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"net"
	"net/http"
	"path"
	"reflect"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/storage/etcd/metrics"
	etcdutil "k8s.io/kubernetes/pkg/storage/etcd/util"
	"k8s.io/kubernetes/pkg/util"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/pkg/watch"

	etcd "github.com/coreos/etcd/client"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/golang/glog"
	"golang.org/x/net/context"
)

// storage.Config object for etcd.
type EtcdStorageConfig struct {
	Config EtcdConfig
	Codec  runtime.Codec
}

// implements storage.Config
func (c *EtcdStorageConfig) GetType() string {
	return "etcd"
}

// implements storage.Config
func (c *EtcdStorageConfig) NewStorage() (storage.Interface, error) {
	etcdClient, err := c.Config.newEtcdClient()
	if err != nil {
		return nil, err
	}
	return NewEtcdStorage(etcdClient, c.Codec, c.Config.Prefix, c.Config.Quorum), nil
}

// Configuration object for constructing etcd.Config
type EtcdConfig struct {
	Prefix     string
	ServerList []string
	KeyFile    string
	CertFile   string
	CAFile     string
	Quorum     bool
}

func (c *EtcdConfig) newEtcdClient() (etcd.Client, error) {
	t, err := c.newHttpTransport()
	if err != nil {
		return nil, err
	}

	cli, err := etcd.New(etcd.Config{
		Endpoints: c.ServerList,
		Transport: t,
	})
	if err != nil {
		return nil, err
	}

	return cli, nil
}

func (c *EtcdConfig) newHttpTransport() (*http.Transport, error) {
	info := transport.TLSInfo{
		CertFile: c.CertFile,
		KeyFile:  c.KeyFile,
		CAFile:   c.CAFile,
	}
	cfg, err := info.ClientConfig()
	if err != nil {
		return nil, err
	}

	// Copied from etcd.DefaultTransport declaration.
	// TODO: Determine if transport needs optimization
	tr := utilnet.SetTransportDefaults(&http.Transport{
		Proxy: http.ProxyFromEnvironment,
		Dial: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).Dial,
		TLSHandshakeTimeout: 10 * time.Second,
		MaxIdleConnsPerHost: 500,
		TLSClientConfig:     cfg,
	})

	return tr, nil
}

// Creates a new storage interface from the client
// TODO: deprecate in favor of storage.Config abstraction over time
func NewEtcdStorage(client etcd.Client, codec runtime.Codec, prefix string, quorum bool) storage.Interface {
	return &etcdHelper{
		etcdMembersAPI: etcd.NewMembersAPI(client),
		etcdKeysAPI:    etcd.NewKeysAPI(client),
		codec:          codec,
		versioner:      APIObjectVersioner{},
		copier:         api.Scheme,
		pathPrefix:     path.Join("/", prefix),
		quorum:         quorum,
		cache:          util.NewCache(maxEtcdCacheEntries),
	}
}

// etcdHelper is the reference implementation of storage.Interface.
type etcdHelper struct {
	etcdMembersAPI etcd.MembersAPI
	etcdKeysAPI    etcd.KeysAPI
	codec          runtime.Codec
	copier         runtime.ObjectCopier
	// Note that versioner is required for etcdHelper to work correctly.
	// The public constructors (NewStorage & NewEtcdStorage) are setting it
	// correctly, so be careful when manipulating with it manually.
	// optional, has to be set to perform any atomic operations
	versioner storage.Versioner
	// prefix for all etcd keys
	pathPrefix string
	// if true,  perform quorum read
	quorum bool

	// We cache objects stored in etcd. For keys we use Node.ModifiedIndex which is equivalent
	// to resourceVersion.
	// This depends on etcd's indexes being globally unique across all objects/types. This will
	// have to revisited if we decide to do things like multiple etcd clusters, or etcd will
	// support multi-object transaction that will result in many objects with the same index.
	// Number of entries stored in the cache is controlled by maxEtcdCacheEntries constant.
	// TODO: Measure how much this cache helps after the conversion code is optimized.
	cache util.Cache
}

func init() {
	metrics.Register()
}

// Codec provides access to the underlying codec being used by the implementation.
func (h *etcdHelper) Codec() runtime.Codec {
	return h.codec
}

// Implements storage.Interface.
func (h *etcdHelper) Backends(ctx context.Context) []string {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}
	members, err := h.etcdMembersAPI.List(ctx)
	if err != nil {
		glog.Errorf("Error obtaining etcd members list: %q", err)
		return nil
	}
	mlist := []string{}
	for _, member := range members {
		mlist = append(mlist, member.ClientURLs...)
	}
	return mlist
}

// Implements storage.Interface.
func (h *etcdHelper) Versioner() storage.Versioner {
	return h.versioner
}

// Implements storage.Interface.
func (h *etcdHelper) Create(ctx context.Context, key string, obj, out runtime.Object, ttl uint64) error {
	trace := util.NewTrace("etcdHelper::Create " + getTypeName(obj))
	defer trace.LogIfLong(250 * time.Millisecond)
	if ctx == nil {
		glog.Errorf("Context is nil")
	}
	key = h.prefixEtcdKey(key)
	data, err := runtime.Encode(h.codec, obj)
	trace.Step("Object encoded")
	if err != nil {
		return err
	}
	if version, err := h.versioner.ObjectResourceVersion(obj); err == nil && version != 0 {
		return errors.New("resourceVersion may not be set on objects to be created")
	}
	trace.Step("Version checked")

	startTime := time.Now()
	opts := etcd.SetOptions{
		TTL:       time.Duration(ttl) * time.Second,
		PrevExist: etcd.PrevNoExist,
	}
	response, err := h.etcdKeysAPI.Set(ctx, key, string(data), &opts)
	metrics.RecordEtcdRequestLatency("create", getTypeName(obj), startTime)
	trace.Step("Object created")
	if err != nil {
		return err
	}
	if out != nil {
		if _, err := conversion.EnforcePtr(out); err != nil {
			panic("unable to convert output object to pointer")
		}
		_, _, err = h.extractObj(response, err, out, false, false)
	}
	return err
}

// Implements storage.Interface.
func (h *etcdHelper) Set(ctx context.Context, key string, obj, out runtime.Object, ttl uint64) error {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}

	version := uint64(0)
	var err error
	if version, err = h.versioner.ObjectResourceVersion(obj); err != nil {
		return errors.New("couldn't get resourceVersion from object")
	}
	if version != 0 {
		// We cannot store object with resourceVersion in etcd, we need to clear it here.
		if err := h.versioner.UpdateObject(obj, nil, 0); err != nil {
			return errors.New("resourceVersion cannot be set on objects store in etcd")
		}
	}

	var response *etcd.Response
	data, err := runtime.Encode(h.codec, obj)
	if err != nil {
		return err
	}
	key = h.prefixEtcdKey(key)

	create := true
	if version != 0 {
		create = false
		startTime := time.Now()
		opts := etcd.SetOptions{
			TTL:       time.Duration(ttl) * time.Second,
			PrevIndex: version,
		}
		response, err = h.etcdKeysAPI.Set(ctx, key, string(data), &opts)
		metrics.RecordEtcdRequestLatency("compareAndSwap", getTypeName(obj), startTime)
		if err != nil {
			return err
		}
	}
	if create {
		// Create will fail if a key already exists.
		startTime := time.Now()
		opts := etcd.SetOptions{
			TTL:       time.Duration(ttl) * time.Second,
			PrevExist: etcd.PrevNoExist,
		}
		response, err = h.etcdKeysAPI.Set(ctx, key, string(data), &opts)
		if err != nil {
			return err
		}
		metrics.RecordEtcdRequestLatency("create", getTypeName(obj), startTime)
	}

	if out != nil {
		if _, err := conversion.EnforcePtr(out); err != nil {
			panic("unable to convert output object to pointer")
		}
		_, _, err = h.extractObj(response, err, out, false, false)
	}

	return err
}

// Implements storage.Interface.
func (h *etcdHelper) Delete(ctx context.Context, key string, out runtime.Object) error {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}
	key = h.prefixEtcdKey(key)
	if _, err := conversion.EnforcePtr(out); err != nil {
		panic("unable to convert output object to pointer")
	}

	startTime := time.Now()
	response, err := h.etcdKeysAPI.Delete(ctx, key, nil)
	metrics.RecordEtcdRequestLatency("delete", getTypeName(out), startTime)
	if !etcdutil.IsEtcdNotFound(err) {
		// if the object that existed prior to the delete is returned by etcd, update out.
		if err != nil || response.PrevNode != nil {
			_, _, err = h.extractObj(response, err, out, false, true)
		}
	}
	return err
}

// Implements storage.Interface.
func (h *etcdHelper) Watch(ctx context.Context, key string, resourceVersion string, filter storage.FilterFunc) (watch.Interface, error) {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}
	watchRV, err := storage.ParseWatchResourceVersion(resourceVersion)
	if err != nil {
		return nil, err
	}
	key = h.prefixEtcdKey(key)
	w := newEtcdWatcher(false, h.quorum, nil, filter, h.codec, h.versioner, nil, h)
	go w.etcdWatch(ctx, h.etcdKeysAPI, key, watchRV)
	return w, nil
}

// Implements storage.Interface.
func (h *etcdHelper) WatchList(ctx context.Context, key string, resourceVersion string, filter storage.FilterFunc) (watch.Interface, error) {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}
	watchRV, err := storage.ParseWatchResourceVersion(resourceVersion)
	if err != nil {
		return nil, err
	}
	key = h.prefixEtcdKey(key)
	w := newEtcdWatcher(true, h.quorum, exceptKey(key), filter, h.codec, h.versioner, nil, h)
	go w.etcdWatch(ctx, h.etcdKeysAPI, key, watchRV)
	return w, nil
}

// Implements storage.Interface.
func (h *etcdHelper) Get(ctx context.Context, key string, objPtr runtime.Object, ignoreNotFound bool) error {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}
	key = h.prefixEtcdKey(key)
	_, _, _, err := h.bodyAndExtractObj(ctx, key, objPtr, ignoreNotFound)
	return err
}

// bodyAndExtractObj performs the normal Get path to etcd, returning the parsed node and response for additional information
// about the response, like the current etcd index and the ttl.
func (h *etcdHelper) bodyAndExtractObj(ctx context.Context, key string, objPtr runtime.Object, ignoreNotFound bool) (body string, node *etcd.Node, res *etcd.Response, err error) {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}
	startTime := time.Now()

	opts := &etcd.GetOptions{
		Quorum: h.quorum,
	}

	response, err := h.etcdKeysAPI.Get(ctx, key, opts)
	metrics.RecordEtcdRequestLatency("get", getTypeName(objPtr), startTime)

	if err != nil && !etcdutil.IsEtcdNotFound(err) {
		return "", nil, nil, err
	}
	body, node, err = h.extractObj(response, err, objPtr, ignoreNotFound, false)
	return body, node, response, err
}

func (h *etcdHelper) extractObj(response *etcd.Response, inErr error, objPtr runtime.Object, ignoreNotFound, prevNode bool) (body string, node *etcd.Node, err error) {
	if response != nil {
		if prevNode {
			node = response.PrevNode
		} else {
			node = response.Node
		}
	}
	if inErr != nil || node == nil || len(node.Value) == 0 {
		if ignoreNotFound {
			v, err := conversion.EnforcePtr(objPtr)
			if err != nil {
				return "", nil, err
			}
			v.Set(reflect.Zero(v.Type()))
			return "", nil, nil
		} else if inErr != nil {
			return "", nil, inErr
		}
		return "", nil, fmt.Errorf("unable to locate a value on the response: %#v", response)
	}
	body = node.Value
	out, gvk, err := h.codec.Decode([]byte(body), nil, objPtr)
	if err != nil {
		return body, nil, err
	}
	if out != objPtr {
		return body, nil, fmt.Errorf("unable to decode object %s into %v", gvk.String(), reflect.TypeOf(objPtr))
	}
	// being unable to set the version does not prevent the object from being extracted
	_ = h.versioner.UpdateObject(objPtr, node.Expiration, node.ModifiedIndex)
	return body, node, err
}

// Implements storage.Interface.
func (h *etcdHelper) GetToList(ctx context.Context, key string, filter storage.FilterFunc, listObj runtime.Object) error {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}
	trace := util.NewTrace("GetToList " + getTypeName(listObj))
	listPtr, err := meta.GetItemsPtr(listObj)
	if err != nil {
		return err
	}
	key = h.prefixEtcdKey(key)
	startTime := time.Now()
	trace.Step("About to read etcd node")

	opts := &etcd.GetOptions{
		Quorum: h.quorum,
	}
	response, err := h.etcdKeysAPI.Get(ctx, key, opts)

	metrics.RecordEtcdRequestLatency("get", getTypeName(listPtr), startTime)
	trace.Step("Etcd node read")
	if err != nil {
		if etcdutil.IsEtcdNotFound(err) {
			return nil
		}
		return err
	}

	nodes := make([]*etcd.Node, 0)
	nodes = append(nodes, response.Node)

	if err := h.decodeNodeList(nodes, filter, listPtr); err != nil {
		return err
	}
	trace.Step("Object decoded")
	if err := h.versioner.UpdateList(listObj, response.Index); err != nil {
		return err
	}
	return nil
}

// decodeNodeList walks the tree of each node in the list and decodes into the specified object
func (h *etcdHelper) decodeNodeList(nodes []*etcd.Node, filter storage.FilterFunc, slicePtr interface{}) error {
	trace := util.NewTrace("decodeNodeList " + getTypeName(slicePtr))
	defer trace.LogIfLong(500 * time.Millisecond)
	v, err := conversion.EnforcePtr(slicePtr)
	if err != nil || v.Kind() != reflect.Slice {
		// This should not happen at runtime.
		panic("need ptr to slice")
	}
	for _, node := range nodes {
		if node.Dir {
			trace.Step("Decoding dir " + node.Key + " START")
			if err := h.decodeNodeList(node.Nodes, filter, slicePtr); err != nil {
				return err
			}
			trace.Step("Decoding dir " + node.Key + " END")
			continue
		}
		if obj, found := h.getFromCache(node.ModifiedIndex, filter); found {
			// obj != nil iff it matches the filter function.
			if obj != nil {
				v.Set(reflect.Append(v, reflect.ValueOf(obj).Elem()))
			}
		} else {
			obj, _, err := h.codec.Decode([]byte(node.Value), nil, reflect.New(v.Type().Elem()).Interface().(runtime.Object))
			if err != nil {
				return err
			}
			// being unable to set the version does not prevent the object from being extracted
			_ = h.versioner.UpdateObject(obj, node.Expiration, node.ModifiedIndex)
			if filter(obj) {
				v.Set(reflect.Append(v, reflect.ValueOf(obj).Elem()))
			}
			if node.ModifiedIndex != 0 {
				h.addToCache(node.ModifiedIndex, obj)
			}
		}
	}
	trace.Step(fmt.Sprintf("Decoded %v nodes", len(nodes)))
	return nil
}

// Implements storage.Interface.
func (h *etcdHelper) List(ctx context.Context, key string, resourceVersion string, filter storage.FilterFunc, listObj runtime.Object) error {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}
	trace := util.NewTrace("List " + getTypeName(listObj))
	defer trace.LogIfLong(time.Second)
	listPtr, err := meta.GetItemsPtr(listObj)
	if err != nil {
		return err
	}
	key = h.prefixEtcdKey(key)
	startTime := time.Now()
	trace.Step("About to list etcd node")
	nodes, index, err := h.listEtcdNode(ctx, key)
	metrics.RecordEtcdRequestLatency("list", getTypeName(listPtr), startTime)
	trace.Step("Etcd node listed")
	if err != nil {
		return err
	}
	if err := h.decodeNodeList(nodes, filter, listPtr); err != nil {
		return err
	}
	trace.Step("Node list decoded")
	if err := h.versioner.UpdateList(listObj, index); err != nil {
		return err
	}
	return nil
}

func (h *etcdHelper) listEtcdNode(ctx context.Context, key string) ([]*etcd.Node, uint64, error) {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}
	opts := etcd.GetOptions{
		Recursive: true,
		Sort:      true,
		Quorum:    h.quorum,
	}
	result, err := h.etcdKeysAPI.Get(ctx, key, &opts)
	if err != nil {
		var index uint64
		if etcdError, ok := err.(etcd.Error); ok {
			index = etcdError.Index
		}
		nodes := make([]*etcd.Node, 0)
		if etcdutil.IsEtcdNotFound(err) {
			return nodes, index, nil
		} else {
			return nodes, index, err
		}
	}
	return result.Node.Nodes, result.Index, nil
}

// Implements storage.Interface.
func (h *etcdHelper) GuaranteedUpdate(ctx context.Context, key string, ptrToType runtime.Object, ignoreNotFound bool, tryUpdate storage.UpdateFunc) error {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}
	v, err := conversion.EnforcePtr(ptrToType)
	if err != nil {
		// Panic is appropriate, because this is a programming error.
		panic("need ptr to type")
	}
	key = h.prefixEtcdKey(key)
	for {
		obj := reflect.New(v.Type()).Interface().(runtime.Object)
		origBody, node, res, err := h.bodyAndExtractObj(ctx, key, obj, ignoreNotFound)
		if err != nil {
			return err
		}
		meta := storage.ResponseMeta{}
		if node != nil {
			meta.TTL = node.TTL
			if node.Expiration != nil {
				meta.Expiration = node.Expiration
			}
			meta.ResourceVersion = node.ModifiedIndex
		}
		// Get the object to be written by calling tryUpdate.
		ret, newTTL, err := tryUpdate(obj, meta)
		if err != nil {
			return err
		}

		index := uint64(0)
		ttl := uint64(0)
		if node != nil {
			index = node.ModifiedIndex
			if node.TTL != 0 {
				ttl = uint64(node.TTL)
			}
			if node.Expiration != nil && ttl == 0 {
				ttl = 1
			}
		} else if res != nil {
			index = res.Index
		}

		if newTTL != nil {
			if ttl != 0 && *newTTL == 0 {
				// TODO: remove this after we have verified this is no longer an issue
				glog.V(4).Infof("GuaranteedUpdate is clearing TTL for %q, may not be intentional", key)
			}
			ttl = *newTTL
		}

		// Since update object may have a resourceVersion set, we need to clear it here.
		if err := h.versioner.UpdateObject(ret, meta.Expiration, 0); err != nil {
			return errors.New("resourceVersion cannot be set on objects store in etcd")
		}

		data, err := runtime.Encode(h.codec, ret)
		if err != nil {
			return err
		}

		// First time this key has been used, try creating new value.
		if index == 0 {
			startTime := time.Now()
			opts := etcd.SetOptions{
				TTL:       time.Duration(ttl) * time.Second,
				PrevExist: etcd.PrevNoExist,
			}
			response, err := h.etcdKeysAPI.Set(ctx, key, string(data), &opts)
			metrics.RecordEtcdRequestLatency("create", getTypeName(ptrToType), startTime)
			if etcdutil.IsEtcdNodeExist(err) {
				continue
			}
			_, _, err = h.extractObj(response, err, ptrToType, false, false)
			return err
		}

		if string(data) == origBody {
			// If we don't send an update, we simply return the currently existing
			// version of the object.
			_, _, err := h.extractObj(res, nil, ptrToType, ignoreNotFound, false)
			return err
		}

		startTime := time.Now()
		// Swap origBody with data, if origBody is the latest etcd data.
		opts := etcd.SetOptions{
			PrevValue: origBody,
			PrevIndex: index,
			TTL:       time.Duration(ttl) * time.Second,
		}
		response, err := h.etcdKeysAPI.Set(ctx, key, string(data), &opts)
		metrics.RecordEtcdRequestLatency("compareAndSwap", getTypeName(ptrToType), startTime)
		if etcdutil.IsEtcdTestFailed(err) {
			// Try again.
			continue
		}
		_, _, err = h.extractObj(response, err, ptrToType, false, false)
		return err
	}
}

func (h *etcdHelper) prefixEtcdKey(key string) string {
	if strings.HasPrefix(key, h.pathPrefix) {
		return key
	}
	return path.Join(h.pathPrefix, key)
}

// etcdCache defines interface used for caching objects stored in etcd. Objects are keyed by
// their Node.ModifiedIndex, which is unique across all types.
// All implementations must be thread-safe.
type etcdCache interface {
	getFromCache(index uint64, filter storage.FilterFunc) (runtime.Object, bool)
	addToCache(index uint64, obj runtime.Object)
}

const maxEtcdCacheEntries int = 50000

func getTypeName(obj interface{}) string {
	return reflect.TypeOf(obj).String()
}

func (h *etcdHelper) getFromCache(index uint64, filter storage.FilterFunc) (runtime.Object, bool) {
	startTime := time.Now()
	defer func() {
		metrics.ObserveGetCache(startTime)
	}()
	obj, found := h.cache.Get(index)
	if found {
		if !filter(obj.(runtime.Object)) {
			return nil, true
		}
		// We should not return the object itself to avoid polluting the cache if someone
		// modifies returned values.
		objCopy, err := h.copier.Copy(obj.(runtime.Object))
		if err != nil {
			glog.Errorf("Error during DeepCopy of cached object: %q", err)
			// We can't return a copy, thus we report the object as not found.
			return nil, false
		}
		metrics.ObserveCacheHit()
		return objCopy.(runtime.Object), true
	}
	metrics.ObserveCacheMiss()
	return nil, false
}

func (h *etcdHelper) addToCache(index uint64, obj runtime.Object) {
	startTime := time.Now()
	defer func() {
		metrics.ObserveAddCache(startTime)
	}()
	objCopy, err := h.copier.Copy(obj)
	if err != nil {
		glog.Errorf("Error during DeepCopy of cached object: %q", err)
		return
	}
	isOverwrite := h.cache.Add(index, objCopy)
	if !isOverwrite {
		metrics.ObserveNewEntry()
	}
}
