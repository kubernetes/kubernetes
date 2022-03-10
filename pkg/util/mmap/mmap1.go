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

package mmap

import (
	"k8s.io/klog/v2"
)

type LEVEL int

const (
	LEVEL_0 LEVEL = iota
	LEVEL_1
	LEVEL_2
	LEVEL_3
	LEVEL_MAX
)

type VisitItemFunc func(v interface{}, k ...interface{}) (LEVEL, error)

type AcceptValueFunc func(v interface{})

type Mmaper interface {
	GetLeafValues(f AcceptValueFunc, k ...interface{}) []interface{}
	GetNextLevelKeys(k ...interface{}) []interface{}
	GetLevelKeyNum(level LEVEL) int
	Iterate(level LEVEL, f VisitItemFunc, k ...interface{}) error
	Insert(v interface{}, k ...interface{}) bool
	Delete(k ...interface{}) bool
	Exist(k ...interface{}) bool
	Num(k ...interface{}) int

	getAll(f AcceptValueFunc) []interface{}
	inner_iterate(level LEVEL, f VisitItemFunc, kbase ...interface{}) (LEVEL, error)
}

type Mmap1 struct {
	m1 map[interface{}]interface{}
}

func NewMmap1() Mmaper {
	return &Mmap1{make(map[interface{}]interface{})}
}

func (m *Mmap1) GetLeafValues(f AcceptValueFunc, k ...interface{}) []interface{} {
	keynum := len(k)
	if keynum > 1 {
		klog.InfoS("Mmap1 GetLeafValues keynum should le 1", "keynum", keynum)
		return []interface{}{}
	}

	locGet := false
	if keynum == 0 {
		locGet = true
	}

	if locGet {
		return m.getAll(f)
	}

	if v, ok := m.m1[k[0]]; ok {
		if f != nil {
			f(v)
		} else {
			return []interface{}{v}
		}
	}

	return []interface{}{}
}

func (m *Mmap1) GetNextLevelKeys(k ...interface{}) []interface{} {
	keynum := len(k)
	if keynum != 0 {
		klog.InfoS("Mmap1 GetNextLevelKeys keynum should be 0", "keynum", keynum)
		return []interface{}{}
	}

	rslt := make([]interface{}, 0, len(m.m1))
	for k := range m.m1 {
		rslt = append(rslt, k)
	}
	return rslt
}

func (m *Mmap1) GetLevelKeyNum(level LEVEL) int {
	if level != LEVEL_1 {
		klog.InfoS("Mmap1 GetLevelKeyNum level should eq LEVEL_1")
		return 0
	}

	return len(m.m1)
}

func (m *Mmap1) Iterate(level LEVEL, f VisitItemFunc, k ...interface{}) error {
	keynum := len(k)
	if keynum > 1 {
		klog.InfoS("Mmap1 Iterate keynum should le 1", "keynum", keynum)
		return nil
	}

	if level != LEVEL_1 {
		klog.InfoS("Mmap1 Iterate level should eq 1", "level", level)
		return nil
	}

	locIterate := false
	if keynum == 0 {
		locIterate = true
	}

	if locIterate {
		level, err := m.inner_iterate(level, f)
		if level > LEVEL_1 {
			return err
		}
		return nil
	}

	if v, ok := m.m1[k[0]]; ok {
		continueLevel, err := f(v)
		if continueLevel > LEVEL_1 {
			return err
		}
	}

	return nil
}

func (m *Mmap1) inner_iterate(level LEVEL, f VisitItemFunc, kbase ...interface{}) (LEVEL, error) {
	for k, v := range m.m1 {
		keys := append(kbase, k)
		continueLevel, err := f(v, keys...)

		if continueLevel > LEVEL_1 {
			return continueLevel, err
		}
	}

	return LEVEL_0, nil
}

func (m *Mmap1) getAll(f AcceptValueFunc) []interface{} {
	rslt := []interface{}{}
	for _, v := range m.m1 {
		if f != nil {
			f(v)
		} else {
			rslt = append(rslt, v)
		}
	}

	return rslt
}

func (m *Mmap1) Insert(v interface{}, k ...interface{}) bool {
	if len(k) != 1 {
		klog.InfoS("Mmap1 Insert keynum should be 1", "keynum", len(k))
		return false
	}

	m.m1[k[0]] = v
	return true
}

func (m *Mmap1) Delete(k ...interface{}) bool {
	if len(k) != 1 {
		klog.InfoS("Mmap1 Delete keynum should be 1", "keynum", len(k))
		return false
	}

	delete(m.m1, k[0])

	return true
}

func (m *Mmap1) Exist(k ...interface{}) bool {
	if len(k) != 1 {
		klog.InfoS("Mmap1 Exist keynum should be 1", "keynum", len(k))
		return false
	}

	_, ok := m.m1[k[0]]

	return ok
}

func (m *Mmap1) Num(k ...interface{}) int {
	keynum := len(k)
	if keynum != 0 {
		klog.InfoS("Mmap1 Num keynum should be 0", "keynum", keynum)
		return 0
	}

	return len(m.m1)
}
