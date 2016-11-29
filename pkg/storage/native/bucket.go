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

package native

import (
	"strings"
)

type bucket struct {
	parent   *bucket
	path     string
	children map[string]*bucket

	items map[string]itemData
}

func newBucket(parent *bucket, key string) *bucket {
	b := &bucket{
		parent:   parent,
		children: make(map[string]*bucket),
		items:    make(map[string]itemData),
	}
	if parent != nil {
		b.path = parent.path + key + "/"
	} else {
		b.path = key + "/"
	}
	return b

}

// We assume lock is held!
func (s *bucket) resolveBucket(path string, create bool) *bucket {
	// TODO: Easy to optimize (a lot!)
	path = strings.Trim(path, "/")
	if path == "" {
		return s
	}

	tokens := strings.Split(path, "/")
	b := s
	for _, t := range tokens {
		if t == "" {
			continue
		}
		child := b.children[t]
		if child == nil {
			if create {
				child = newBucket(b, t)
				b.children[t] = child
			} else {
				return nil
			}
		}
		b = child
	}
	return b
}

func splitPath(path string) (string, string) {
	// TODO: Easy to optimize (a lot!)
	path = strings.Trim(path, "/")

	lastSlash := strings.LastIndexByte(path, '/')
	if lastSlash == -1 {
		return "", path
	}
	item := path[lastSlash+1:]
	bucket := path[:lastSlash]
	return bucket, item
}

//func normalizePath(path string) string {
//	// TODO: Easy to optimize (a lot)
//	path = strings.Trim(path, "/")
//
//	var b bytes.Buffer
//
//	for _, t := range strings.Split(path, "/") {
//		if t == "" {
//			continue
//		}
//		if b.Len() != 0 {
//			b.WriteString("/")
//		}
//		b.WriteString(t)
//	}
//	return b.String()
//}
