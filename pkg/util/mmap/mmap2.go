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

type Mmap2 struct {
	m2  map[interface{}]Mmaper
	num int // leaf num
}

func NewMmap2() Mmaper {
	return &Mmap2{
		m2:  make(map[interface{}]Mmaper),
		num: 0,
	}
}

func (m *Mmap2) GetLeafValues(f AcceptValueFunc, k ...interface{}) []interface{} {
	keynum := len(k)
	if keynum > 2 {
		klog.InfoS("Mmap2 GetLeafValues keynum should le 2", "keynum", keynum)
		return []interface{}{}
	}

	locGet := false
	if keynum == 0 {
		locGet = true
	}

	if locGet {
		return m.getAll(f)
	}

	lockey, keys := k[0], k[1:]
	if m1, ok := m.m2[lockey]; ok {
		return m1.GetLeafValues(f, keys...)
	}

	return []interface{}{}
}

func (m *Mmap2) GetNextLevelKeys(k ...interface{}) []interface{} {
	keynum := len(k)
	if keynum > 1 {
		klog.InfoS("Mmap2 GetNextLevelKeys keynum should le 1", "keynum", keynum)
		return []interface{}{}
	}

	locGet := false
	if keynum == 0 {
		locGet = true
	}

	if locGet {
		rslt := make([]interface{}, 0, len(m.m2))
		for k := range m.m2 {
			rslt = append(rslt, k)
		}
		return rslt
	}

	lockey, keys := k[0], k[1:]
	if m1, ok := m.m2[lockey]; ok {
		return m1.GetNextLevelKeys(keys...)
	}

	return []interface{}{}
}

func (m *Mmap2) GetLevelKeyNum(level LEVEL) int {
	if level > LEVEL_2 || level == LEVEL_0 {
		klog.InfoS("Mmap2 GetLevelKeyNum level should le LEVEL_2")
		return 0
	}

	locGet := false
	if level == LEVEL_2 {
		locGet = true
	}

	if locGet {
		return len(m.m2)
	}

	rslt := 0
	for _, m1 := range m.m2 {
		rslt = rslt + m1.GetLevelKeyNum(level)
	}
	return rslt
}

func (m *Mmap2) Iterate(level LEVEL, f VisitItemFunc, k ...interface{}) error {
	keynum := len(k)
	if keynum > 2 {
		klog.InfoS("Mmap2 Iterate keynum should le 2", "keynum", keynum)
		return nil
	}

	if level > LEVEL_2 || level == LEVEL_0 {
		klog.InfoS("Mmap2 Iterate level should le LEVEL_2")
		return nil
	}

	locIterate := false
	if keynum == 0 {
		locIterate = true
	}

	if locIterate {
		level, err := m.inner_iterate(level, f)
		if level > LEVEL_2 {
			return err
		}
		return nil
	}

	lockey, keys := k[0], k[1:]
	if m1, ok := m.m2[lockey]; ok {
		return m1.Iterate(level, f, keys...)
	}

	return nil
}

func (m *Mmap2) inner_iterate(level LEVEL, f VisitItemFunc, kbase ...interface{}) (LEVEL, error) {
	locGet := false
	if level == LEVEL_2 {
		locGet = true
	}

	for k, m1 := range m.m2 {
		keys := append(kbase, k)
		var err error
		var continueLevel LEVEL
		if locGet {
			continueLevel, err = f(m1, keys...)
		} else {
			continueLevel, err = m1.inner_iterate(level, f, keys...)
		}

		if continueLevel > LEVEL_2 {
			return continueLevel, err
		}
	}

	return LEVEL_0, nil
}

func (m *Mmap2) getAll(f AcceptValueFunc) []interface{} {
	rslt := []interface{}{}
	for _, m1 := range m.m2 {
		vlist := m1.getAll(f)
		if f == nil {
			rslt = append(rslt, vlist...)
		}
	}

	return rslt
}

func (m *Mmap2) Insert(v interface{}, k ...interface{}) bool {
	keynum := len(k)
	if keynum > 2 || keynum < 1 {
		klog.InfoS("Mmap2 Insert keynum should le 2 and ge 1", "keynum", keynum)
		return false
	}

	locInsert := false

	if keynum == 1 { //1个参数时,v必须是*Mmap1,一个以上参数交给下级map检验
		if !func(v interface{}) bool {
			if _, ok := v.(*Mmap1); !ok {
				return false
			}
			return true
		}(v) {
			klog.InfoS("Mmap2 Insert value must be *Mmap1, when has 1 arg")
			return false
		}

		locInsert = true //本级操作
	}

	opok := false
	lockey, keys := k[0], k[1:]
	if m1, ok := m.m2[lockey]; ok {
		m.num = m.num - m1.Num()
		if locInsert {
			m.m2[lockey] = v.(*Mmap1)
			opok = true
		} else {
			opok = m1.Insert(v, keys...)
		}
		m.num = m.num + m.m2[lockey].Num()
	} else {
		if locInsert {
			m.m2[lockey] = v.(*Mmap1)
			opok = true
		} else {
			m1 := NewMmap1()
			opok = m1.Insert(v, keys...)
			m.m2[lockey] = m1
		}
		m.num = m.num + m.m2[lockey].Num()
	}

	return opok
}

func (m *Mmap2) Delete(k ...interface{}) bool {
	keynum := len(k)
	if keynum > 2 || keynum < 1 {
		klog.InfoS("Mmap2 Delete keynum should le 2 and ge 1", "keynum", keynum)
		return false
	}

	locDelete := false
	if keynum == 1 {
		locDelete = true
	}

	opok := true
	lockey, keys := k[0], k[1:]
	if m1, ok := m.m2[lockey]; ok {
		m.num = m.num - m1.Num()
		if locDelete {
			delete(m.m2, lockey)
			return true
		}

		opok = m1.Delete(keys...)
		if m1.Num() == 0 {
			delete(m.m2, lockey)
		}
		m.num = m.num + m1.Num()
	}

	return opok
}

func (m *Mmap2) Exist(k ...interface{}) bool {
	keynum := len(k)
	if keynum > 2 || keynum < 1 {
		klog.InfoS("Mmap2 Exist keynum should le 2 and ge 1", "keynum", keynum)
		return false
	}

	locExist := false
	if keynum == 1 {
		locExist = true
	}

	lockey, keys := k[0], k[1:]
	if m1, ok := m.m2[lockey]; ok {
		if locExist {
			return ok
		}
		return m1.Exist(keys...)
	}

	return false
}

func (m *Mmap2) Num(k ...interface{}) int {
	keynum := len(k)
	if keynum > 1 {
		klog.InfoS("Mmap2 Num keynum should le 1", "keynum", keynum)
		return 0
	}

	locGet := false
	if keynum == 0 {
		locGet = true
	}

	if locGet {
		return m.num
	}

	lockey, keys := k[0], k[1:]
	if m1, ok := m.m2[lockey]; ok {
		return m1.Num(keys...)
	}

	return 0
}
