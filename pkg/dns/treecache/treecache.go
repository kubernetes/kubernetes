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

package treecache

import (
	"encoding/json"
	"strings"

	skymsg "github.com/skynetservices/skydns/msg"
)

type TreeCache interface {
	// GetEntry with the given key for the given path.
	GetEntry(key string, path ...string) (interface{}, bool)

	// Get a list of values including wildcards labels (e.g. "*").
	GetValuesForPathWithWildcards(path ...string) []*skymsg.Service

	// SetEntry creates the entire path if it doesn't already exist in
	// the cache, then sets the given service record under the given
	// key. The path this entry would have occupied in an etcd datastore
	// is computed from the given fqdn and stored as the "Key" of the
	// skydns service; this is only required because skydns expects the
	// service record to contain a key in a specific format (presumably
	// for legacy compatibility). Note that the fqnd string typically
	// contains both the key and all elements in the path.
	SetEntry(key string, val *skymsg.Service, fqdn string, path ...string)

	// SetSubCache inserts the given subtree under the given
	// path:key. Usually the key is the name of a Kubernetes Service,
	// and the path maps to the cluster subdomains matching the Service.
	SetSubCache(key string, subCache TreeCache, path ...string)

	// DeletePath removes all entries associated with a given path.
	DeletePath(path ...string) bool

	// Serialize dumps a JSON representation of the cache.
	Serialize() (string, error)
}

type treeCache struct {
	ChildNodes map[string]*treeCache
	Entries    map[string]interface{}
}

func NewTreeCache() TreeCache {
	return &treeCache{
		ChildNodes: make(map[string]*treeCache),
		Entries:    make(map[string]interface{}),
	}
}

func (cache *treeCache) Serialize() (string, error) {
	prettyJSON, err := json.MarshalIndent(cache, "", "\t")
	if err != nil {
		return "", err
	}
	return string(prettyJSON), nil
}

func (cache *treeCache) SetEntry(key string, val *skymsg.Service, fqdn string, path ...string) {
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

func (cache *treeCache) getSubCache(path ...string) *treeCache {
	childCache := cache
	for _, subpath := range path {
		childCache = childCache.ChildNodes[subpath]
		if childCache == nil {
			return nil
		}
	}
	return childCache
}

func (cache *treeCache) SetSubCache(key string, subCache TreeCache, path ...string) {
	node := cache.ensureChildNode(path...)
	node.ChildNodes[key] = subCache.(*treeCache)
}

func (cache *treeCache) GetEntry(key string, path ...string) (interface{}, bool) {
	childNode := cache.getSubCache(path...)
	if childNode == nil {
		return nil, false
	}
	val, ok := childNode.Entries[key]
	return val, ok
}

func (cache *treeCache) GetValuesForPathWithWildcards(path ...string) []*skymsg.Service {
	retval := []*skymsg.Service{}
	nodesToExplore := []*treeCache{cache}
	for idx, subpath := range path {
		nextNodesToExplore := []*treeCache{}
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

func (cache *treeCache) DeletePath(path ...string) bool {
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

func (cache *treeCache) appendValues(recursive bool, ref [][]interface{}) {
	for _, value := range cache.Entries {
		ref[0] = append(ref[0], value)
	}
	if recursive {
		for _, node := range cache.ChildNodes {
			node.appendValues(recursive, ref)
		}
	}
}

func (cache *treeCache) ensureChildNode(path ...string) *treeCache {
	childNode := cache
	for _, subpath := range path {
		newNode, ok := childNode.ChildNodes[subpath]
		if !ok {
			newNode = NewTreeCache().(*treeCache)
			childNode.ChildNodes[subpath] = newNode
		}
		childNode = newNode
	}
	return childNode
}
