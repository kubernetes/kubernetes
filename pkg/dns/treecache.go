/*
Copyright 2016 The Kubernetes Authors.

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

package dns

import (
	"encoding/json"
	"strings"

	skymsg "github.com/skynetservices/skydns/msg"
)

type TreeCache struct {
	ChildNodes map[string]*TreeCache
	Entries    map[string]interface{}
}

func NewTreeCache() *TreeCache {
	return &TreeCache{
		ChildNodes: make(map[string]*TreeCache),
		Entries:    make(map[string]interface{}),
	}
}

func (cache *TreeCache) Serialize() (string, error) {
	prettyJSON, err := json.MarshalIndent(cache, "", "\t")
	if err != nil {
		return "", err
	}
	return string(prettyJSON), nil
}

// setEntry creates the entire path if it doesn't already exist in the cache,
// then sets the given service record under the given key. The path this entry
// would have occupied in an etcd datastore is computed from the given fqdn and
// stored as the "Key" of the skydns service; this is only required because
// skydns expects the service record to contain a key in a specific format
// (presumably for legacy compatibility). Note that the fqnd string typically
// contains both the key and all elements in the path.
func (cache *TreeCache) setEntry(key string, val *skymsg.Service, fqdn string, path ...string) {
	// TODO: Consolidate setEntry and setSubCache into a single method with a
	// type switch.
	// TODO: Instead of passing the fqdn as an argument, we can reconstruct
	// it from the path, provided callers always pass the full path to the
	// object. This is currently *not* the case, since callers first create
	// a new, empty node, populate it, then parent it under the right path.
	// So we don't know the full key till the final parenting operation.
	node := cache.ensureChildNode(path...)

	// This key is used to construct the "target" for SRV record lookups.
	// For normal service/endpoint lookups, this will result in a key like:
	// /skydns/local/cluster/svc/svcNS/svcName/record-hash
	// but for headless services that govern pods requesting a specific
	// hostname (as used by petset), this will end up being:
	// /skydns/local/cluster/svc/svcNS/svcName/pod-hostname
	val.Key = skymsg.Path(fqdn)
	node.Entries[key] = val
}

func (cache *TreeCache) getSubCache(path ...string) *TreeCache {
	childCache := cache
	for _, subpath := range path {
		childCache = childCache.ChildNodes[subpath]
		if childCache == nil {
			return nil
		}
	}
	return childCache
}

// setSubCache inserts the given subtree under the given path:key. Usually the
// key is the name of a Kubernetes Service, and the path maps to the cluster
// subdomains matching the Service.
func (cache *TreeCache) setSubCache(key string, subCache *TreeCache, path ...string) {
	node := cache.ensureChildNode(path...)
	node.ChildNodes[key] = subCache
}

func (cache *TreeCache) getEntry(key string, path ...string) (interface{}, bool) {
	childNode := cache.getSubCache(path...)
	if childNode == nil {
		return nil, false
	}
	val, ok := childNode.Entries[key]
	return val, ok
}

func (cache *TreeCache) getValuesForPathWithWildcards(path ...string) []*skymsg.Service {
	retval := []*skymsg.Service{}
	nodesToExplore := []*TreeCache{cache}
	for idx, subpath := range path {
		nextNodesToExplore := []*TreeCache{}
		if idx == len(path)-1 {
			// if path ends on an entry, instead of a child node, add the entry
			for _, node := range nodesToExplore {
				if subpath == "*" {
					nextNodesToExplore = append(nextNodesToExplore, node)
				} else {
					if val, ok := node.Entries[subpath]; ok {
						retval = append(retval, val.(*skymsg.Service))
					} else {
						childNode := node.ChildNodes[subpath]
						if childNode != nil {
							nextNodesToExplore = append(nextNodesToExplore, childNode)
						}
					}
				}
			}
			nodesToExplore = nextNodesToExplore
			break
		}

		if subpath == "*" {
			for _, node := range nodesToExplore {
				for subkey, subnode := range node.ChildNodes {
					if !strings.HasPrefix(subkey, "_") {
						nextNodesToExplore = append(nextNodesToExplore, subnode)
					}
				}
			}
		} else {
			for _, node := range nodesToExplore {
				childNode := node.ChildNodes[subpath]
				if childNode != nil {
					nextNodesToExplore = append(nextNodesToExplore, childNode)
				}
			}
		}
		nodesToExplore = nextNodesToExplore
	}

	for _, node := range nodesToExplore {
		for _, val := range node.Entries {
			retval = append(retval, val.(*skymsg.Service))
		}
	}
	return retval
}

func (cache *TreeCache) deletePath(path ...string) bool {
	if len(path) == 0 {
		return false
	}
	if parentNode := cache.getSubCache(path[:len(path)-1]...); parentNode != nil {
		name := path[len(path)-1]
		if _, ok := parentNode.ChildNodes[name]; ok {
			delete(parentNode.ChildNodes, name)
			return true
		}
		// ExternalName services are stored with their name as the leaf key
		if _, ok := parentNode.Entries[name]; ok {
			delete(parentNode.Entries, name)
			return true
		}
	}
	return false
}

func (cache *TreeCache) deleteEntry(key string, path ...string) bool {
	childNode := cache.getSubCache(path...)
	if childNode == nil {
		return false
	}
	if _, ok := childNode.Entries[key]; ok {
		delete(childNode.Entries, key)
		return true
	}
	return false
}

func (cache *TreeCache) appendValues(recursive bool, ref [][]interface{}) {
	for _, value := range cache.Entries {
		ref[0] = append(ref[0], value)
	}
	if recursive {
		for _, node := range cache.ChildNodes {
			node.appendValues(recursive, ref)
		}
	}
}

func (cache *TreeCache) ensureChildNode(path ...string) *TreeCache {
	childNode := cache
	for _, subpath := range path {
		newNode, ok := childNode.ChildNodes[subpath]
		if !ok {
			newNode = NewTreeCache()
			childNode.ChildNodes[subpath] = newNode
		}
		childNode = newNode
	}
	return childNode
}

// unused function. keeping it around in commented-fashion
// in the future, we might need some form of this function so that
// we can serialize to a file in a mounted empty dir..
//const (
//	dataFile = "data.dat"
//	crcFile  = "data.crc"
//)
//func (cache *TreeCache) Serialize(dir string) (string, error) {
//	cache.m.RLock()
//	defer cache.m.RUnlock()
//	b, err := json.Marshal(cache)
//	if err != nil {
//		return "", err
//	}
//
//	if err := ensureDir(dir, os.FileMode(0755)); err != nil {
//		return "", err
//	}
//	if err := ioutil.WriteFile(path.Join(dir, dataFile), b, 0644); err != nil {
//		return "", err
//	}
//	if err := ioutil.WriteFile(path.Join(dir, crcFile), getMD5(b), 0644); err != nil {
//		return "", err
//	}
//	return string(b), nil
//}

//func ensureDir(path string, perm os.FileMode) error {
//	s, err := os.Stat(path)
//	if err != nil || !s.IsDir() {
//		return os.Mkdir(path, perm)
//	}
//	return nil
//}

//func getMD5(b []byte) []byte {
//	h := md5.New()
//	h.Write(b)
//	return []byte(fmt.Sprintf("%x", h.Sum(nil)))
//}

// unused function. keeping it around in commented-fashion
// in the future, we might need some form of this function so that
// we can restart kube-dns, deserialize the tree and have a cache
// without having to wait for kube-dns to reach out to API server.
//func Deserialize(dir string) (*TreeCache, error) {
//	b, err := ioutil.ReadFile(path.Join(dir, dataFile))
//	if err != nil {
//		return nil, err
//	}
//
//	hash, err := ioutil.ReadFile(path.Join(dir, crcFile))
//	if err != nil {
//		return nil, err
//	}
//	if !reflect.DeepEqual(hash, getMD5(b)) {
//		return nil, fmt.Errorf("Checksum failed")
//	}
//
//	var cache TreeCache
//	err = json.Unmarshal(b, &cache)
//	if err != nil {
//		return nil, err
//	}
//	cache.m = &sync.RWMutex{}
//	return &cache, nil
//}
