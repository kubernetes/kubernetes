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

type Mmap3 struct {
	m3  map[interface{}]Mmaper
	num int // leaf num
}

func NewMmap3() Mmaper {
	return &Mmap3{
		m3:  make(map[interface{}]Mmaper),
		num: 0,
	}
}

func (m *Mmap3) GetLeafValues(f AcceptValueFunc, k ...interface{}) []interface{} {
	keynum := len(k)
	if keynum > 3 {
		klog.InfoS("Mmap3 GetLeafValues keynum should le 3", "keynum", keynum)
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
	if m2, ok := m.m3[lockey]; ok {
		return m2.GetLeafValues(f, keys...)
	}

	return []interface{}{}
}

func (m *Mmap3) GetNextLevelKeys(k ...interface{}) []interface{} {
	keynum := len(k)
	if keynum > 2 {
		klog.InfoS("Mmap3 GetNextLevelKeys keynum should le 2", "keynum", keynum)
		return []interface{}{}
	}

	locGet := false
	if keynum == 0 {
		locGet = true
	}

	if locGet {
		rslt := make([]interface{}, 0, len(m.m3))
		for k := range m.m3 {
			rslt = append(rslt, k)
		}
		return rslt
	}

	lockey, keys := k[0], k[1:]
	if m2, ok := m.m3[lockey]; ok {
		return m2.GetNextLevelKeys(keys...)
	}

	return []interface{}{}
}

func (m *Mmap3) GetLevelKeyNum(level LEVEL) int {
	if level > LEVEL_3 || level == LEVEL_0 {
		klog.InfoS("Mmap3 GetLevelKeyNum level should le LEVEL_3 and not LEVEL_0")
		return 0
	}

	locGet := false
	if level == LEVEL_3 {
		locGet = true
	}

	if locGet {
		return len(m.m3)
	}

	rslt := 0
	for _, m2 := range m.m3 {
		rslt = rslt + m2.GetLevelKeyNum(level)
	}
	return rslt
}

func (m *Mmap3) Iterate(level LEVEL, f VisitItemFunc, k ...interface{}) error {
	keynum := len(k)
	if keynum > 3 {
		klog.InfoS("Mmap3 Iterate keynum should le 3", "keynum", keynum)
		return nil
	}

	if level > LEVEL_3 || level == LEVEL_0 {
		klog.InfoS("Mmap3 Iterate level should le LEVEL_3 and not LEVEL_0")
		return nil
	}

	locIterate := false
	if keynum == 0 {
		locIterate = true
	}

	if locIterate {
		level, err := m.inner_iterate(level, f)
		if level > LEVEL_3 {
			return err
		}
		return nil
	}

	lockey, keys := k[0], k[1:]
	if m2, ok := m.m3[lockey]; ok {
		return m2.Iterate(level, f, keys...)
	}

	return nil
}

func (m *Mmap3) inner_iterate(level LEVEL, f VisitItemFunc, kbase ...interface{}) (LEVEL, error) {
	locGet := false
	if level == LEVEL_3 {
		locGet = true
	}

	for k, m2 := range m.m3 {
		keys := append(kbase, k)
		var err error
		var continueLevel LEVEL
		if locGet {
			continueLevel, err = f(m2, keys...)
		} else {
			continueLevel, err = m2.inner_iterate(level, f, keys...)
		}

		if continueLevel > LEVEL_3 {
			return continueLevel, err
		}
	}

	return LEVEL_0, nil
}

func (m *Mmap3) getAll(f AcceptValueFunc) []interface{} {
	rslt := []interface{}{}
	for _, m2 := range m.m3 {
		vlist := m2.getAll(f)
		if f == nil {
			rslt = append(rslt, vlist...)
		}
	}

	return rslt
}

func (m *Mmap3) Insert(v interface{}, k ...interface{}) bool {
	keynum := len(k)
	if keynum > 3 || keynum < 1 {
		klog.InfoS("Mmap3 Insert keynum should le 3 and ge 1", "keynum", keynum)
		return false
	}

	locInsert := false

	if keynum == 1 { //1个参数时,v必须是*Mmap2,一个以上参数交给下级map检验
		if !func(v interface{}) bool {
			if _, ok := v.(*Mmap2); !ok {
				return false
			}
			return true
		}(v) {
			klog.InfoS("Mmap3 Insert value must be *Mmap2, when has 1 arg")
			return false
		}

		locInsert = true //本级操作
	}

	opok := false
	lockey, keys := k[0], k[1:]
	if m2, ok := m.m3[lockey]; ok {
		m.num = m.num - m2.Num()
		if locInsert {
			m.m3[lockey] = v.(*Mmap2)
			opok = true
		} else {
			opok = m2.Insert(v, keys...)
		}
		m.num = m.num + m.m3[lockey].Num()
	} else {
		if locInsert {
			m.m3[lockey] = v.(*Mmap2)
			opok = true
		} else {
			m2 := NewMmap2()
			opok = m2.Insert(v, keys...)
			m.m3[lockey] = m2
		}
		m.num = m.num + m.m3[lockey].Num()
	}

	return opok
}

func (m *Mmap3) Delete(k ...interface{}) bool {
	keynum := len(k)
	if keynum > 3 || keynum < 1 {
		klog.InfoS("Mmap3 Delete keynum should le 3 and ge 1", "keynum", keynum)
		return false
	}

	locDelete := false
	if keynum == 1 {
		locDelete = true
	}

	opok := true
	lockey, keys := k[0], k[1:]
	if m2, ok := m.m3[lockey]; ok {
		m.num = m.num - m2.Num()
		if locDelete {
			delete(m.m3, lockey)
			return true
		}

		opok = m2.Delete(keys...)
		if m2.Num() == 0 {
			delete(m.m3, lockey)
		}
		m.num = m.num + m2.Num()
	}

	return opok
}

func (m *Mmap3) Exist(k ...interface{}) bool {
	keynum := len(k)
	if keynum > 3 || keynum < 1 {
		klog.InfoS("Mmap3 Exist keynum should le 3 and ge 1", "keynum", keynum)
		return false
	}

	locExist := false
	if keynum == 1 {
		locExist = true
	}

	lockey, keys := k[0], k[1:]
	if m2, ok := m.m3[lockey]; ok {
		if locExist {
			return ok
		}
		return m2.Exist(keys...)
	}

	return false
}

func (m *Mmap3) Num(k ...interface{}) int {
	keynum := len(k)
	if keynum > 2 {
		klog.InfoS("Mmap3 Num keynum should le 2", "keynum", keynum)
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
	if m2, ok := m.m3[lockey]; ok {
		return m2.Num(keys...)
	}

	return 0
}
