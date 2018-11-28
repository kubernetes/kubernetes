package manager

import (
	"fmt"
	"strings"
	"sync"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/kubelet/configmap"
)

type configMapKey struct {
	namespace string
	name      string
}

func (cmk *configMapKey) String() string {
	return fmt.Sprintf("%s/%s", cmk.namespace, cmk.name)
}

func (cmk *configMapKey) Parse(keyStr string) error {
	token := strings.Split(keyStr, "/")
	if len(token) != 2 {
		err := fmt.Errorf("invalid key string: %s", keyStr)
		return err
	}
	cmk.namespace = token[0]
	cmk.name = token[1]
	return nil
}

type configMapCacheObject struct {
	key             *configMapKey
	resourceVersion string
}

type callbackFunc func(configMap *v1.ConfigMap)

type ConfigMapWatcher struct {
	mutex              sync.RWMutex
	cache              map[string]*configMapCacheObject
	cacheKeySet        sets.String
	configMapManager   configmap.Manager
	updateCallbackFunc callbackFunc
}

func NewConfigMapWatcher(
	configMapManager configmap.Manager,
	updateCallbackFunc callbackFunc,
) *ConfigMapWatcher {
	w := &ConfigMapWatcher{
		cache:              make(map[string]*configMapCacheObject),
		cacheKeySet:        sets.NewString(),
		configMapManager:   configMapManager,
		updateCallbackFunc: updateCallbackFunc,
	}
	return w
}

// add new, delete old, update exists
func (w *ConfigMapWatcher) Sync(keyStrs sets.String) {
	w.mutex.RLock()
	added := keyStrs.Difference(w.cacheKeySet)
	deleted := w.cacheKeySet.Difference(keyStrs)
	updated := keyStrs.Intersection(w.cacheKeySet)
	w.mutex.RUnlock()

	w.mutex.Lock()
	for keyStr := range deleted {
		delete(w.cache, keyStr)
		w.cacheKeySet.Delete(keyStr)
	}
	w.mutex.Unlock()

	for keyStr := range added.Union(updated) {
		key := &configMapKey{}
		err := key.Parse(keyStr)
		if err != nil {
			glog.Errorf("parse configmap key error, %v, key: %s", err, keyStr)
			continue
		}
		configMap, err := w.configMapManager.GetConfigMap(key.namespace, key.name)
		if err != nil {
			glog.Errorf("get configmap error, %v, namespace: %s, name: %s", err, key.namespace, key.name)
			continue
		}

		var cacheObj *configMapCacheObject
		w.mutex.RLock()
		if w.cacheKeySet.Has(keyStr) {
			cacheObj = w.cache[keyStr]
		}
		w.mutex.RUnlock()

		// add/update cache obj
		if (cacheObj == nil) || (cacheObj.resourceVersion != configMap.ResourceVersion) {
			newCacheObj := &configMapCacheObject{
				key:             key,
				resourceVersion: configMap.ResourceVersion,
			}
			w.mutex.Lock()
			w.cacheKeySet.Insert(keyStr)
			w.cache[keyStr] = newCacheObj
			w.mutex.Unlock()
			// invoke callback function
			w.updateCallbackFunc(configMap)
		}
	}
}
