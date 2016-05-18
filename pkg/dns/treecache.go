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

package dns

import (
	"bytes"
	"encoding/json"
	"strings"
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
	b, err := json.Marshal(cache)
	if err != nil {
		return "", err
	}

	var prettyJSON bytes.Buffer
	err = json.Indent(&prettyJSON, b, "", "\t")

	if err != nil {
		return "", err
	}
	return string(prettyJSON.Bytes()), nil
}

func (cache *TreeCache) setEntry(key string, val interface{}, path ...string) {
	node := cache.ensureChildNode(path...)
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

func (cache *TreeCache) setSubCache(key string, subCache *TreeCache, path ...string) {
	node := cache.ensureChildNode(path...)
	node.ChildNodes[key] = subCache
}

func (cache *TreeCache) getEntry(key string, path ...string) (interface{}, bool) {
	childNode := cache.getSubCache(path...)
	val, ok := childNode.Entries[key]
	return val, ok
}

func (cache *TreeCache) getValuesForPathWithWildcards(path ...string) []interface{} {
	retval := []interface{}{}
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
						retval = append(retval, val)
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
			retval = append(retval, val)
		}
	}

	return retval
}

func (cache *TreeCache) deletePath(path ...string) bool {
	if len(path) == 0 {
		return false
	}
	if parentNode := cache.getSubCache(path[:len(path)-1]...); parentNode != nil {
		if _, ok := parentNode.ChildNodes[path[len(path)-1]]; ok {
			delete(parentNode.ChildNodes, path[len(path)-1])
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
