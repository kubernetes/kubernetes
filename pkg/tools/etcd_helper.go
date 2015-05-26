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

package tools

import (
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"os/exec"
	"path"
	"reflect"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
	"github.com/prometheus/client_golang/prometheus"

	"github.com/golang/glog"
)

var (
	cacheHitCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "etcd_helper_cache_hit_count",
			Help: "Counter of etcd helper cache hits.",
		},
	)
	cacheMissCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "etcd_helper_cache_miss_count",
			Help: "Counter of etcd helper cache miss.",
		},
	)
	cacheEntryCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "etcd_helper_cache_entry_count",
			Help: "Counter of etcd helper cache entries. This can be different from etcd_helper_cache_miss_count " +
				"because two concurrent threads can miss the cache and generate the same entry twice.",
		},
	)
	cacheGetLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Name: "etcd_request_cache_get_latencies_summary",
			Help: "Latency in microseconds of getting an object from etcd cache",
		},
	)
	cacheAddLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Name: "etcd_request_cache_add_latencies_summary",
			Help: "Latency in microseconds of adding an object to etcd cache",
		},
	)
	etcdRequestLatenciesSummary = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Name: "etcd_request_latencies_summary",
			Help: "Etcd request latency summary in microseconds for each operation and object type.",
		},
		[]string{"operation", "type"},
	)
)

const maxEtcdCacheEntries int = 50000

func init() {
	prometheus.MustRegister(cacheHitCounter)
	prometheus.MustRegister(cacheMissCounter)
	prometheus.MustRegister(cacheEntryCounter)
	prometheus.MustRegister(cacheAddLatency)
	prometheus.MustRegister(cacheGetLatency)
	prometheus.MustRegister(etcdRequestLatenciesSummary)
}

func getTypeName(obj interface{}) string {
	return reflect.TypeOf(obj).String()
}

func recordEtcdRequestLatency(verb, resource string, startTime time.Time) {
	etcdRequestLatenciesSummary.WithLabelValues(verb, resource).Observe(float64(time.Since(startTime) / time.Microsecond))
}

// EtcdHelper offers common object marshalling/unmarshalling operations on an etcd client.
type EtcdHelper struct {
	Client EtcdGetSet
	Codec  runtime.Codec
	// optional, no atomic operations can be performed without this interface
	Versioner EtcdVersioner
	// prefix for all etcd keys
	PathPrefix string

	// We cache objects stored in etcd. For keys we use Node.ModifiedIndex which is equivalent
	// to resourceVersion.
	// This depends on etcd's indexes being globally unique across all objects/types. This will
	// have to revisited if we decide to do things like multiple etcd clusters, or etcd will
	// support multi-object transaction that will result in many objects with the same index.
	// Number of entries stored in the cache is controlled by maxEtcdCacheEntries constant.
	// TODO: Measure how much this cache helps after the conversion code is optimized.
	cache util.Cache
}

// NewEtcdHelper creates a helper that works against objects that use the internal
// Kubernetes API objects.
func NewEtcdHelper(client EtcdGetSet, codec runtime.Codec, prefix string) EtcdHelper {
	return EtcdHelper{
		Client:     client,
		Codec:      codec,
		Versioner:  APIObjectVersioner{},
		PathPrefix: prefix,
		cache:      util.NewCache(maxEtcdCacheEntries),
	}
}

// IsEtcdNotFound returns true iff err is an etcd not found error.
func IsEtcdNotFound(err error) bool {
	return isEtcdErrorNum(err, EtcdErrorCodeNotFound)
}

// IsEtcdNodeExist returns true iff err is an etcd node aleady exist error.
func IsEtcdNodeExist(err error) bool {
	return isEtcdErrorNum(err, EtcdErrorCodeNodeExist)
}

// IsEtcdTestFailed returns true iff err is an etcd write conflict.
func IsEtcdTestFailed(err error) bool {
	return isEtcdErrorNum(err, EtcdErrorCodeTestFailed)
}

// IsEtcdWatchStoppedByUser returns true iff err is a client triggered stop.
func IsEtcdWatchStoppedByUser(err error) bool {
	return etcd.ErrWatchStoppedByUser == err
}

// isEtcdErrorNum returns true iff err is an etcd error, whose errorCode matches errorCode
func isEtcdErrorNum(err error, errorCode int) bool {
	etcdError, ok := err.(*etcd.EtcdError)
	return ok && etcdError != nil && etcdError.ErrorCode == errorCode
}

// etcdErrorIndex returns the index associated with the error message and whether the
// index was available.
func etcdErrorIndex(err error) (uint64, bool) {
	if etcdError, ok := err.(*etcd.EtcdError); ok {
		return etcdError.Index, true
	}
	return 0, false
}

func (h *EtcdHelper) listEtcdNode(key string) ([]*etcd.Node, uint64, error) {
	result, err := h.Client.Get(key, true, true)
	if err != nil {
		index, ok := etcdErrorIndex(err)
		if !ok {
			index = 0
		}
		nodes := make([]*etcd.Node, 0)
		if IsEtcdNotFound(err) {
			return nodes, index, nil
		} else {
			return nodes, index, err
		}
	}
	return result.Node.Nodes, result.EtcdIndex, nil
}

// decodeNodeList walks the tree of each node in the list and decodes into the specified object
func (h *EtcdHelper) decodeNodeList(nodes []*etcd.Node, slicePtr interface{}) error {
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
			if err := h.decodeNodeList(node.Nodes, slicePtr); err != nil {
				return err
			}
			trace.Step("Decoding dir " + node.Key + " END")
			continue
		}
		if obj, found := h.getFromCache(node.ModifiedIndex); found {
			v.Set(reflect.Append(v, reflect.ValueOf(obj).Elem()))
		} else {
			obj := reflect.New(v.Type().Elem())
			if err := h.Codec.DecodeInto([]byte(node.Value), obj.Interface().(runtime.Object)); err != nil {
				return err
			}
			if h.Versioner != nil {
				// being unable to set the version does not prevent the object from being extracted
				_ = h.Versioner.UpdateObject(obj.Interface().(runtime.Object), node)
			}
			v.Set(reflect.Append(v, obj.Elem()))
			if node.ModifiedIndex != 0 {
				h.addToCache(node.ModifiedIndex, obj.Interface().(runtime.Object))
			}
		}
	}
	trace.Step(fmt.Sprintf("Decoded %v nodes", len(nodes)))
	return nil
}

// etcdCache defines interface used for caching objects stored in etcd. Objects are keyed by
// their Node.ModifiedIndex, which is unique across all types.
// All implementations must be thread-safe.
type etcdCache interface {
	getFromCache(index uint64) (runtime.Object, bool)
	addToCache(index uint64, obj runtime.Object)
}

func (h *EtcdHelper) getFromCache(index uint64) (runtime.Object, bool) {
	trace := util.NewTrace("getFromCache")
	defer trace.LogIfLong(200 * time.Microsecond)
	startTime := time.Now()
	defer func() {
		cacheGetLatency.Observe(float64(time.Since(startTime) / time.Microsecond))
	}()
	obj, found := h.cache.Get(index)
	trace.Step("Raw get done")
	if found {
		// We should not return the object itself to avoid poluting the cache if someone
		// modifies returned values.
		objCopy, err := conversion.DeepCopy(obj)
		trace.Step("Deep copied")
		if err != nil {
			glog.Errorf("Error during DeepCopy of cached object: %q", err)
			return nil, false
		}
		cacheHitCounter.Inc()
		return objCopy.(runtime.Object), true
	}
	cacheMissCounter.Inc()
	return nil, false
}

func (h *EtcdHelper) addToCache(index uint64, obj runtime.Object) {
	startTime := time.Now()
	defer func() {
		cacheAddLatency.Observe(float64(time.Since(startTime) / time.Microsecond))
	}()
	objCopy, err := conversion.DeepCopy(obj)
	if err != nil {
		glog.Errorf("Error during DeepCopy of cached object: %q", err)
		return
	}
	isOverwrite := h.cache.Add(index, objCopy)
	if !isOverwrite {
		cacheEntryCounter.Inc()
	}
}

// ExtractToList works on a *List api object (an object that satisfies the runtime.IsList
// definition) and extracts a go object per etcd node into a slice with the resource version.
func (h *EtcdHelper) ExtractToList(key string, listObj runtime.Object) error {
	trace := util.NewTrace("ExtractToList " + getTypeName(listObj))
	defer trace.LogIfLong(time.Second)
	listPtr, err := runtime.GetItemsPtr(listObj)
	if err != nil {
		return err
	}
	key = h.PrefixEtcdKey(key)
	startTime := time.Now()
	trace.Step("About to list etcd node")
	nodes, index, err := h.listEtcdNode(key)
	recordEtcdRequestLatency("list", getTypeName(listPtr), startTime)
	trace.Step("Etcd node listed")
	if err != nil {
		return err
	}
	if err := h.decodeNodeList(nodes, listPtr); err != nil {
		return err
	}
	trace.Step("Node list decoded")
	if h.Versioner != nil {
		if err := h.Versioner.UpdateList(listObj, index); err != nil {
			return err
		}
	}
	return nil
}

// ExtractObjToList unmarshals json found at key and opaques it into a *List api object
// (an object that satisfies the runtime.IsList definition).
func (h *EtcdHelper) ExtractObjToList(key string, listObj runtime.Object) error {
	trace := util.NewTrace("ExtractObjToList " + getTypeName(listObj))
	listPtr, err := runtime.GetItemsPtr(listObj)
	if err != nil {
		return err
	}
	key = h.PrefixEtcdKey(key)
	startTime := time.Now()
	trace.Step("About to read etcd node")
	response, err := h.Client.Get(key, false, false)
	recordEtcdRequestLatency("get", getTypeName(listPtr), startTime)
	trace.Step("Etcd node read")
	if err != nil {
		if IsEtcdNotFound(err) {
			return nil
		}
		return err
	}

	nodes := make([]*etcd.Node, 0)
	nodes = append(nodes, response.Node)

	if err := h.decodeNodeList(nodes, listPtr); err != nil {
		return err
	}
	trace.Step("Object decoded")
	if h.Versioner != nil {
		if err := h.Versioner.UpdateList(listObj, response.EtcdIndex); err != nil {
			return err
		}
	}
	return nil
}

// ExtractObj unmarshals json found at key into objPtr. On a not found error, will either return
// a zero object of the requested type, or an error, depending on ignoreNotFound. Treats
// empty responses and nil response nodes exactly like a not found error.
func (h *EtcdHelper) ExtractObj(key string, objPtr runtime.Object, ignoreNotFound bool) error {
	key = h.PrefixEtcdKey(key)
	_, _, err := h.bodyAndExtractObj(key, objPtr, ignoreNotFound)
	return err
}

func (h *EtcdHelper) bodyAndExtractObj(key string, objPtr runtime.Object, ignoreNotFound bool) (body string, modifiedIndex uint64, err error) {
	startTime := time.Now()
	response, err := h.Client.Get(key, false, false)
	recordEtcdRequestLatency("get", getTypeName(objPtr), startTime)

	if err != nil && !IsEtcdNotFound(err) {
		return "", 0, err
	}
	return h.extractObj(response, err, objPtr, ignoreNotFound, false)
}

func (h *EtcdHelper) extractObj(response *etcd.Response, inErr error, objPtr runtime.Object, ignoreNotFound, prevNode bool) (body string, modifiedIndex uint64, err error) {
	var node *etcd.Node
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
				return "", 0, err
			}
			v.Set(reflect.Zero(v.Type()))
			return "", 0, nil
		} else if inErr != nil {
			return "", 0, inErr
		}
		return "", 0, fmt.Errorf("unable to locate a value on the response: %#v", response)
	}
	body = node.Value
	err = h.Codec.DecodeInto([]byte(body), objPtr)
	if h.Versioner != nil {
		_ = h.Versioner.UpdateObject(objPtr, node)
		// being unable to set the version does not prevent the object from being extracted
	}
	return body, node.ModifiedIndex, err
}

// CreateObj adds a new object at a key unless it already exists. 'ttl' is time-to-live in seconds,
// and 0 means forever. If no error is returned and out is not nil, out will be set to the read value
// from etcd.
func (h *EtcdHelper) CreateObj(key string, obj, out runtime.Object, ttl uint64) error {
	key = h.PrefixEtcdKey(key)
	data, err := h.Codec.Encode(obj)
	if err != nil {
		return err
	}
	if h.Versioner != nil {
		if version, err := h.Versioner.ObjectResourceVersion(obj); err == nil && version != 0 {
			return errors.New("resourceVersion may not be set on objects to be created")
		}
	}

	startTime := time.Now()
	response, err := h.Client.Create(key, string(data), ttl)
	recordEtcdRequestLatency("create", getTypeName(obj), startTime)
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

// Delete removes the specified key.
func (h *EtcdHelper) Delete(key string, recursive bool) error {
	key = h.PrefixEtcdKey(key)
	startTime := time.Now()
	_, err := h.Client.Delete(key, recursive)
	recordEtcdRequestLatency("delete", "UNKNOWN", startTime)
	return err
}

// DeleteObj removes the specified key and returns the value that existed at that spot.
func (h *EtcdHelper) DeleteObj(key string, out runtime.Object) error {
	key = h.PrefixEtcdKey(key)
	if _, err := conversion.EnforcePtr(out); err != nil {
		panic("unable to convert output object to pointer")
	}

	startTime := time.Now()
	response, err := h.Client.Delete(key, false)
	recordEtcdRequestLatency("delete", getTypeName(out), startTime)
	if !IsEtcdNotFound(err) {
		// if the object that existed prior to the delete is returned by etcd, update out.
		if err != nil || response.PrevNode != nil {
			_, _, err = h.extractObj(response, err, out, false, true)
		}
	}
	return err
}

// SetObj marshals obj via json, and stores under key. Will do an atomic update if obj's ResourceVersion
// field is set. 'ttl' is time-to-live in seconds, and 0 means forever. If no error is returned and out is
// not nil, out will be set to the read value from etcd.
func (h *EtcdHelper) SetObj(key string, obj, out runtime.Object, ttl uint64) error {
	var response *etcd.Response
	data, err := h.Codec.Encode(obj)
	if err != nil {
		return err
	}
	key = h.PrefixEtcdKey(key)

	create := true
	if h.Versioner != nil {
		if version, err := h.Versioner.ObjectResourceVersion(obj); err == nil && version != 0 {
			create = false
			startTime := time.Now()
			response, err = h.Client.CompareAndSwap(key, string(data), ttl, "", version)
			recordEtcdRequestLatency("compareAndSwap", getTypeName(obj), startTime)
			if err != nil {
				return err
			}
		}
	}
	if create {
		// Create will fail if a key already exists.
		startTime := time.Now()
		response, err = h.Client.Create(key, string(data), ttl)
		recordEtcdRequestLatency("create", getTypeName(obj), startTime)
	}

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

// Pass an EtcdUpdateFunc to EtcdHelper.GuaranteedUpdate to make an etcd update that is guaranteed to succeed.
// See the comment for GuaranteedUpdate for more detail.
type EtcdUpdateFunc func(input runtime.Object) (output runtime.Object, ttl uint64, err error)

// GuaranteedUpdate calls "tryUpdate()" to update key "key" that is of type "ptrToType". It keeps
// calling tryUpdate() and retrying the update until success if there is etcd index conflict. Note that object
// passed to tryUpdate() may change across invocations of tryUpdate() if other writers are simultaneously
// updating it, so tryUpdate() needs to take into account the current contents of the object when
// deciding how the updated object (that it returns) should look.
//
// Example:
//
// h := &util.EtcdHelper{client, encoding, versioning}
// err := h.GuaranteedUpdate("myKey", &MyType{}, true, func(input runtime.Object) (runtime.Object, uint64, error) {
//	// Before each invocation of the user-defined function, "input" is reset to etcd's current contents for "myKey".
//
//	cur := input.(*MyType) // Guaranteed to succeed.
//
//	// Make a *modification*.
//	cur.Counter++
//
//	// Return the modified object. Return an error to stop iterating. Return a non-zero uint64 to set
//      // the TTL on the object.
//	return cur, 0, nil
// })
//
func (h *EtcdHelper) GuaranteedUpdate(key string, ptrToType runtime.Object, ignoreNotFound bool, tryUpdate EtcdUpdateFunc) error {
	v, err := conversion.EnforcePtr(ptrToType)
	if err != nil {
		// Panic is appropriate, because this is a programming error.
		panic("need ptr to type")
	}
	key = h.PrefixEtcdKey(key)
	for {
		obj := reflect.New(v.Type()).Interface().(runtime.Object)
		origBody, index, err := h.bodyAndExtractObj(key, obj, ignoreNotFound)
		if err != nil {
			return err
		}

		ret, ttl, err := tryUpdate(obj)
		if err != nil {
			return err
		}

		data, err := h.Codec.Encode(ret)
		if err != nil {
			return err
		}

		// First time this key has been used, try creating new value.
		if index == 0 {
			startTime := time.Now()
			response, err := h.Client.Create(key, string(data), ttl)
			recordEtcdRequestLatency("create", getTypeName(ptrToType), startTime)
			if IsEtcdNodeExist(err) {
				continue
			}
			_, _, err = h.extractObj(response, err, ptrToType, false, false)
			return err
		}

		if string(data) == origBody {
			return nil
		}

		startTime := time.Now()
		response, err := h.Client.CompareAndSwap(key, string(data), ttl, origBody, index)
		recordEtcdRequestLatency("compareAndSwap", getTypeName(ptrToType), startTime)
		if IsEtcdTestFailed(err) {
			continue
		}
		_, _, err = h.extractObj(response, err, ptrToType, false, false)
		return err
	}
}

func (h *EtcdHelper) PrefixEtcdKey(key string) string {
	if strings.HasPrefix(key, path.Join("/", h.PathPrefix)) {
		return key
	}
	return path.Join("/", h.PathPrefix, key)
}

// Copies the key-value pairs from their old location to a new location based
// on this helper's etcd prefix. All old keys without the prefix are then deleted.
func (h *EtcdHelper) MigrateKeys(oldPathPrefix string) error {
	// Check to see if a migration is necessary, i.e. is the oldPrefix different
	// from the newPrefix?
	if h.PathPrefix == oldPathPrefix {
		return nil
	}

	// Get the root node
	response, err := h.Client.Get(oldPathPrefix, false, true)
	if err != nil {
		glog.Infof("Couldn't get the existing etcd root node.")
		return err
	}

	// Perform the migration
	if err = h.migrateChildren(response.Node, oldPathPrefix); err != nil {
		glog.Infof("Error performing the migration.")
		return err
	}

	// Delete the old top-level entry recursively
	// Quick sanity check: Did the process at least create a new top-level entry?
	if _, err = h.Client.Get(h.PathPrefix, false, false); err != nil {
		glog.Infof("Couldn't get the new etcd root node.")
		return err
	} else {
		if _, err = h.Client.Delete(oldPathPrefix, true); err != nil {
			glog.Infof("Couldn't delete the old etcd root node.")
			return err
		}
	}
	return nil
}

// This recurses through the etcd registry. Each key-value pair is copied with
// to a new pair with a prefixed key.
func (h *EtcdHelper) migrateChildren(parent *etcd.Node, oldPathPrefix string) error {
	for _, child := range parent.Nodes {
		if child.Dir && len(child.Nodes) > 0 {
			// Descend into this directory
			h.migrateChildren(child, oldPathPrefix)

			// All children have been migrated, so this directory has
			// already been automatically added.
			continue
		}

		// Check if already prefixed (maybe we got interrupted in last attempt)
		if strings.HasPrefix(child.Key, h.PathPrefix) {
			// Skip this iteration
			continue
		}

		// Create new entry
		newKey := path.Join("/", h.PathPrefix, strings.TrimPrefix(child.Key, oldPathPrefix))
		if _, err := h.Client.Create(newKey, child.Value, 0); err != nil {
			// Assuming etcd is still available, this is due to the key
			// already existing, in which case we can skip.
			continue
		}
	}
	return nil
}

// GetEtcdVersion performs a version check against the provided Etcd server,
// returning the string response, and error (if any).
func GetEtcdVersion(host string) (string, error) {
	response, err := http.Get(host + "/version")
	if err != nil {
		return "", err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unsuccessful response from etcd server %q: %v", host, err)
	}
	versionBytes, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return "", err
	}
	return string(versionBytes), nil
}

func startEtcd() (*exec.Cmd, error) {
	cmd := exec.Command("etcd")
	err := cmd.Start()
	if err != nil {
		return nil, err
	}
	return cmd, nil
}

func NewEtcdClientStartServerIfNecessary(server string) (EtcdClient, error) {
	_, err := GetEtcdVersion(server)
	if err != nil {
		glog.Infof("Failed to find etcd, attempting to start.")
		_, err := startEtcd()
		if err != nil {
			return nil, err
		}
	}

	servers := []string{server}
	return etcd.NewClient(servers), nil
}
