/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"crypto/md5"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"reflect"
	"strings"
	"sync"
)

const (
	dataFile = "data.dat"
	crcFile  = "data.crc"
)

type object interface{}

type TreeCache struct {
	ChildNodes map[string]*TreeCache
	Entries    map[string]interface{}
	m          *sync.RWMutex
}

func NewTreeCache() *TreeCache {
	return &TreeCache{
		ChildNodes: make(map[string]*TreeCache),
		Entries:    make(map[string]interface{}),
		m:          &sync.RWMutex{},
	}
}

func Deserialize(dir string) (*TreeCache, error) {
	b, err := ioutil.ReadFile(path.Join(dir, dataFile))
	if err != nil {
		return nil, err
	}
	var hash []byte
	hash, err = ioutil.ReadFile(path.Join(dir, crcFile))
	if err != nil {
		return nil, err
	}
	if !reflect.DeepEqual(hash, getMD5(b)) {
		return nil, fmt.Errorf("Checksum failed")
	}

	var cache TreeCache
	err = json.Unmarshal(b, &cache)
	if err != nil {
		return nil, err
	}
	cache.m = &sync.RWMutex{}
	return &cache, nil
}

func (cache *TreeCache) Serialize(dir string) (string, error) {
	cache.m.RLock()
	defer cache.m.RUnlock()
	b, err := json.Marshal(cache)
	if err != nil {
		return "", err
	}

	if len(dir) == 0 {
		var prettyJSON bytes.Buffer
		err = json.Indent(&prettyJSON, b, "", "\t")

		if err != nil {
			return "", err
		}
		return string(prettyJSON.Bytes()), nil
	}
	if err := ensureDir(dir, os.FileMode(0755)); err != nil {
		return "", err
	}
	if err := ioutil.WriteFile(path.Join(dir, dataFile), b, 0644); err != nil {
		return "", err
	}
	if err := ioutil.WriteFile(path.Join(dir, crcFile), getMD5(b), 0644); err != nil {
		return "", err
	}
	return string(b), nil
}

func (cache *TreeCache) SetEntry(key string, val interface{}, path ...string) {
	cache.m.Lock()
	defer cache.m.Unlock()
	node := cache.ensureChildNode(path...)
	node.Entries[key] = val
}

func (cache *TreeCache) ReplaceEntries(entries map[string]interface{}, path ...string) {
	cache.m.Lock()
	defer cache.m.Unlock()
	node := cache.ensureChildNode(path...)
	node.Entries = make(map[string]interface{})
	for key, val := range entries {
		node.Entries[key] = val
	}
}

func (cache *TreeCache) GetSubCache(path ...string) *TreeCache {
	childCache := cache
	for _, subpath := range path {
		childCache = childCache.ChildNodes[subpath]
		if childCache == nil {
			return childCache
		}
	}
	return childCache
}

func (cache *TreeCache) SetSubCache(key string, subCache *TreeCache, path ...string) {
	cache.m.Lock()
	defer cache.m.Unlock()
	node := cache.ensureChildNode(path...)
	node.ChildNodes[key] = subCache
}

func (cache *TreeCache) GetEntry(key string, path ...string) (interface{}, bool) {
	cache.m.RLock()
	defer cache.m.RUnlock()
	childNode := cache.GetSubCache(path...)
	val, ok := childNode.Entries[key]
	return val, ok
}

func (cache *TreeCache) GetValuesForPathWithRegex(path ...string) []interface{} {
	cache.m.RLock()
	defer cache.m.RUnlock()
	retval := []interface{}{}
	nodesToExplore := []*TreeCache{cache}
	for idx, subpath := range path {
		nextNodesToExplore := []*TreeCache{}
		if idx == len(path)-1 {
			// if path ends on an entry, instead of a child node, add the entry
			for _, node := range nodesToExplore {
				if subpath == "*" || subpath == "any" {
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

		if subpath == "*" || subpath == "any" {
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

func (cache *TreeCache) GetEntries(recursive bool, path ...string) []interface{} {
	cache.m.RLock()
	defer cache.m.RUnlock()
	childNode := cache.GetSubCache(path...)
	if childNode == nil {
		return nil
	}

	retval := [][]interface{}{{}}
	childNode.appendValues(recursive, retval)
	return retval[0]
}

func (cache *TreeCache) DeletePath(path ...string) bool {
	if len(path) == 0 {
		return false
	}
	cache.m.Lock()
	defer cache.m.Unlock()
	if parentNode := cache.GetSubCache(path[:len(path)-1]...); parentNode != nil {
		if _, ok := parentNode.ChildNodes[path[len(path)-1]]; ok {
			delete(parentNode.ChildNodes, path[len(path)-1])
			return true
		}
	}
	return false
}

func (tn *TreeCache) DeleteEntry(key string, path ...string) bool {
	tn.m.Lock()
	defer tn.m.Unlock()
	childNode := tn.GetSubCache(path...)
	if childNode == nil {
		return false
	}
	if _, ok := childNode.Entries[key]; ok {
		delete(childNode.Entries, key)
		return true
	}
	return false
}

func (tn *TreeCache) appendValues(recursive bool, ref [][]interface{}) {
	for _, value := range tn.Entries {
		ref[0] = append(ref[0], value)
	}
	if recursive {
		for _, node := range tn.ChildNodes {
			node.appendValues(recursive, ref)
		}
	}
}

func (tn *TreeCache) ensureChildNode(path ...string) *TreeCache {
	childNode := tn
	for _, subpath := range path {
		newNode := childNode.ChildNodes[subpath]
		if newNode == nil {
			newNode = NewTreeCache()
			childNode.ChildNodes[subpath] = newNode
		}
		childNode = newNode
	}
	return childNode
}

func ensureDir(path string, perm os.FileMode) error {
	s, err := os.Stat(path)
	if err != nil || !s.IsDir() {
		return os.Mkdir(path, perm)
	}
	return nil
}

func getMD5(b []byte) []byte {
	h := md5.New()
	h.Write(b)
	return []byte(fmt.Sprintf("%x", h.Sum(nil)))
}

func main() {
	root := NewTreeCache()
	fmt.Println("Adding Entries")
	root.SetEntry("k", "v")
	root.SetEntry("foo", "bar", "local")
	root.SetEntry("foo1", "bar1", "local", "cluster")

	fmt.Println("Fetching Entries")
	for _, entry := range root.GetEntries(true, "local") {
		fmt.Printf("%s\n", entry)
	}

	fmt.Println("Serializing")
	if _, err := root.Serialize("./foo"); err != nil {
		fmt.Printf("Serialization Error:  %v,\n", err)
		return
	}

	fmt.Println("Deserializing")
	tn, err := Deserialize("./foo")
	if err != nil {
		fmt.Printf("Deserialization Error: %v\n", err)
		return
	}

	fmt.Println("Fetching Entries")
	for _, entry := range tn.GetEntries(true, "local") {
		fmt.Printf("%s\n", entry)
	}
}
