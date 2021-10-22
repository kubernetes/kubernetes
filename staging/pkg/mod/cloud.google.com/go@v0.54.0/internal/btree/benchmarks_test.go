// Copyright 2014 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package btree

import (
	"fmt"
	"sort"
	"testing"
)

const benchmarkTreeSize = 10000

var degrees = []int{2, 8, 32, 64}

func BenchmarkInsert(b *testing.B) {
	insertP := perm(benchmarkTreeSize)
	for _, d := range degrees {
		b.Run(fmt.Sprintf("degree=%d", d), func(b *testing.B) {
			i := 0
			for i < b.N {
				tr := New(d, less)
				for _, m := range insertP {
					tr.Set(m.Key, m.Value)
					i++
					if i >= b.N {
						return
					}
				}
			}
		})
	}
}

func BenchmarkDeleteInsert(b *testing.B) {
	insertP := perm(benchmarkTreeSize)
	for _, d := range degrees {
		b.Run(fmt.Sprintf("degree=%d", d), func(b *testing.B) {
			tr := New(d, less)
			for _, m := range insertP {
				tr.Set(m.Key, m.Value)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m := insertP[i%benchmarkTreeSize]
				tr.Delete(m.Key)
				tr.Set(m.Key, m.Value)
			}
		})
	}
}

func BenchmarkDeleteInsertCloneOnce(b *testing.B) {
	insertP := perm(benchmarkTreeSize)
	for _, d := range degrees {
		b.Run(fmt.Sprintf("degree=%d", d), func(b *testing.B) {
			tr := New(d, less)
			for _, m := range insertP {
				tr.Set(m.Key, m.Value)
			}
			tr = tr.Clone()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m := insertP[i%benchmarkTreeSize]
				tr.Delete(m.Key)
				tr.Set(m.Key, m.Value)
			}
		})
	}
}

func BenchmarkDeleteInsertCloneEachTime(b *testing.B) {
	insertP := perm(benchmarkTreeSize)
	for _, d := range degrees {
		b.Run(fmt.Sprintf("degree=%d", d), func(b *testing.B) {
			tr := New(d, less)
			for _, m := range insertP {
				tr.Set(m.Key, m.Value)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				tr = tr.Clone()
				m := insertP[i%benchmarkTreeSize]
				tr.Delete(m.Key)
				tr.Set(m.Key, m.Value)
			}
		})
	}
}

func BenchmarkDelete(b *testing.B) {
	insertP := perm(benchmarkTreeSize)
	removeP := perm(benchmarkTreeSize)
	for _, d := range degrees {
		b.Run(fmt.Sprintf("degree=%d", d), func(b *testing.B) {
			i := 0
			for i < b.N {
				b.StopTimer()
				tr := New(d, less)
				for _, v := range insertP {
					tr.Set(v.Key, v.Value)
				}
				b.StartTimer()
				for _, m := range removeP {
					tr.Delete(m.Key)
					i++
					if i >= b.N {
						return
					}
				}
				if tr.Len() > 0 {
					panic(tr.Len())
				}
			}
		})
	}
}

func BenchmarkGet(b *testing.B) {
	insertP := perm(benchmarkTreeSize)
	getP := perm(benchmarkTreeSize)
	for _, d := range degrees {
		b.Run(fmt.Sprintf("degree=%d", d), func(b *testing.B) {
			i := 0
			for i < b.N {
				b.StopTimer()
				tr := New(d, less)
				for _, v := range insertP {
					tr.Set(v.Key, v.Value)
				}
				b.StartTimer()
				for _, m := range getP {
					tr.Get(m.Key)
					i++
					if i >= b.N {
						return
					}
				}
			}
		})
	}
}

func BenchmarkGetWithIndex(b *testing.B) {
	insertP := perm(benchmarkTreeSize)
	getP := perm(benchmarkTreeSize)
	for _, d := range degrees {
		b.Run(fmt.Sprintf("degree=%d", d), func(b *testing.B) {
			i := 0
			for i < b.N {
				b.StopTimer()
				tr := New(d, less)
				for _, v := range insertP {
					tr.Set(v.Key, v.Value)
				}
				b.StartTimer()
				for _, m := range getP {
					tr.GetWithIndex(m.Key)
					i++
					if i >= b.N {
						return
					}
				}
			}
		})
	}
}

func BenchmarkGetCloneEachTime(b *testing.B) {
	insertP := perm(benchmarkTreeSize)
	getP := perm(benchmarkTreeSize)
	for _, d := range degrees {
		b.Run(fmt.Sprintf("degree=%d", d), func(b *testing.B) {
			i := 0
			for i < b.N {
				b.StopTimer()
				tr := New(d, less)
				for _, m := range insertP {
					tr.Set(m.Key, m.Value)
				}
				b.StartTimer()
				for _, m := range getP {
					tr = tr.Clone()
					tr.Get(m.Key)
					i++
					if i >= b.N {
						return
					}
				}
			}
		})
	}
}

func BenchmarkFind(b *testing.B) {
	for _, d := range degrees {
		var items []item
		for i := 0; i < 2*d; i++ {
			items = append(items, item{i, i})
		}
		b.Run(fmt.Sprintf("size=%d", len(items)), func(b *testing.B) {
			for _, alg := range []struct {
				name string
				fun  func(Key, []item) (int, bool)
			}{
				{"binary", findBinary},
				{"linear", findLinear},
			} {
				b.Run(alg.name, func(b *testing.B) {
					for i := 0; i < b.N; i++ {
						for j := 0; j < len(items); j++ {
							alg.fun(items[j].key, items)
						}
					}
				})
			}
		})
	}
}

func findBinary(k Key, s []item) (int, bool) {
	i := sort.Search(len(s), func(i int) bool { return less(k, s[i].key) })
	// i is the smallest index of s for which key.Less(s[i].Key), or len(s).
	if i > 0 && !less(s[i-1], k) {
		return i - 1, true
	}
	return i, false
}

func findLinear(k Key, s []item) (int, bool) {
	var i int
	for i = 0; i < len(s); i++ {
		if less(k, s[i].key) {
			break
		}
	}
	if i > 0 && !less(s[i-1].key, k) {
		return i - 1, true
	}
	return i, false
}
