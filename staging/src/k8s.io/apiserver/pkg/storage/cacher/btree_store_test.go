/*
Copyright 2022 The Kubernetes Authors.

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

package cacher

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/google/btree"
)

type item struct {
	Key string
}

func (i *item) Less(than btree.Item) bool {
	return i.Key < than.(*item).Key
}

type btreeTest struct {
	tree btree.BTree
}

func newBTreeTest(degree int) btreeTest {
	return btreeTest{tree: *btree.New(degree)}
}

func (t btreeTest) getByKey(key string) (itemRet interface{}, exists bool, err error) {
	t.tree.Ascend(func(i btree.Item) bool {
		if key == i.(*item).Key {
			itemRet = i
			exists = true
			return false
		}
		return true
	})

	return itemRet, exists, nil
}

func Benchmark_BTreeGetByKey(b *testing.B) {
	tree := newBTreeTest(9)
	prefixes := []string{"/auth/", "/mutate/", "/universe/"}
	for _, p := range prefixes {
		for i := 0; i < 10000; i++ {
			key := fmt.Sprintf("%s%d", p, i)
			tree.tree.ReplaceOrInsert(&item{Key: key})
		}
	}

	rand.Seed(100)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		prefixIndex := rand.Intn(3)
		keyIndex := rand.Intn(10000)
		key := fmt.Sprintf("%s%d", prefixes[prefixIndex], keyIndex)
		b.StartTimer()
		tree.getByKey(key)
	}
}
