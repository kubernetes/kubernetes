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
	"fmt"
	"testing"
)

func TestMmap3(t *testing.T) {
	m3 := NewMmap3()
	m3.Insert("a", "1", "1.1", "1.1.1")
	m3.Insert("b", "1", "1.1", "1.1.2")
	m3.Insert("c", "1", "1.2", "1.2.1")
	m3.Insert("d", "1", "1.2", "1.2.2")
	m3.Insert("e", "2", "2.1", "2.1.1")
	m3.Insert("f", "2", "2.1", "2.1.2")
	m3.Insert("g", "2", "2.2", "2.2.1")
	m3.Insert("h", "2", "2.2", "2.2.2")

	if m3.Num() != 8 {
		t.Errorf("m3.Num %v not accepted", m3.Num())
	}

	if !m3.Exist("1") {
		t.Errorf("%#v not accepted", m3.Exist("1"))
	}
	if !m3.Exist("1", "1.1") {
		t.Errorf("%#v not accepted", m3.Exist("1", "1.1"))
	}
	if !m3.Exist("2", "2.1", "2.1.2") {
		t.Errorf("%#v not accepted", m3.Exist("2", "2.1", "2.1.2"))
	}
	if m3.Exist("2", "2.1", "2.1") {
		t.Errorf("%#v not accepted", m3.Exist("2", "2.1", "2.1"))
	}

	if 2 != m3.GetLevelKeyNum(LEVEL_3) {
		t.Errorf("%#v not accepted", m3.GetLevelKeyNum(LEVEL_3))
	}
	if 4 != m3.GetLevelKeyNum(LEVEL_2) {
		t.Errorf("%#v not accepted", m3.GetLevelKeyNum(LEVEL_2))
	}
	if 8 != m3.GetLevelKeyNum(LEVEL_1) {
		t.Errorf("%#v not accepted", m3.GetLevelKeyNum(LEVEL_1))
	}
	if 0 != m3.GetLevelKeyNum(LEVEL_0) {
		t.Errorf("%#v not accepted", m3.GetLevelKeyNum(LEVEL_0))
	}
	if 0 != m3.GetLevelKeyNum(LEVEL(4)) {
		t.Errorf("%#v not accepted", m3.GetLevelKeyNum(LEVEL(4)))
	}

	if !sliceEqual([]interface{}{"1", "2"}, m3.GetNextLevelKeys()) {
		t.Errorf("%#v not accepted", m3.GetNextLevelKeys())
	}

	if !sliceEqual([]interface{}{"1.1", "1.2"}, m3.GetNextLevelKeys("1")) {
		t.Errorf("%#v not accepted", m3.GetNextLevelKeys())
	}

	if !sliceEqual([]interface{}{"1.1.1", "1.1.2"}, m3.GetNextLevelKeys("1", "1.1")) {
		t.Errorf("%#v not accepted", m3.GetNextLevelKeys())
	}

	if !sliceEqual([]interface{}{}, m3.GetNextLevelKeys("1", "1.1", "1.1.1")) {
		t.Errorf("%#v not accepted", m3.GetNextLevelKeys())
	}

	var iterkeys []interface{}
	f := func(v interface{}, k ...interface{}) (LEVEL, error) {
		iterkeys = append(iterkeys, k[len(k)-1])
		return LEVEL_0, nil
	}

	m3.Iterate(LEVEL_3, f)
	if !sliceEqual([]interface{}{"1", "2"}, iterkeys) {
		t.Errorf("%#v not accepted", iterkeys)
	}

	iterkeys = []interface{}{}
	m3.Iterate(LEVEL_2, f)
	if !sliceEqual([]interface{}{"1.1", "1.2", "2.1", "2.2"}, iterkeys) {
		t.Errorf("%#v not accepted", iterkeys)
	}

	iterkeys = []interface{}{}
	m3.Iterate(LEVEL_1, f)
	if !sliceEqual([]interface{}{"1.1.1", "1.1.2", "1.2.1", "1.2.2", "2.1.1", "2.1.2", "2.2.1", "2.2.2"}, iterkeys) {
		t.Errorf("%#v not accepted", iterkeys)
	}

	f1 := func(v interface{}, k ...interface{}) (LEVEL, error) {
		str := k[0].(string)
		if str == "1" {
			return LEVEL_3, nil
		}
		iterkeys = append(iterkeys, k[len(k)-1])
		return LEVEL_0, nil
	}

	iterkeys = []interface{}{}
	if m3.Iterate(LEVEL_2, f1) != nil {
		t.Errorf("%#v not accepted", m3.Iterate(LEVEL_2, f1))
	}
	if !sliceEqual([]interface{}{"2.1", "2.2"}, iterkeys) {
		t.Errorf("%#v not accepted", iterkeys)
	}

	f2 := func(v interface{}, k ...interface{}) (LEVEL, error) {
		str := k[0].(string)
		if str == "1" {
			return LEVEL_MAX, fmt.Errorf("f2 throw err")
		}
		iterkeys = append(iterkeys, k[len(k)-1])
		return LEVEL_0, nil
	}

	iterkeys = []interface{}{}
	if m3.Iterate(LEVEL_2, f2) == nil {
		t.Errorf("nil not accepted")
	}

	f3 := func(v interface{}, k ...interface{}) (LEVEL, error) {
		iterkeys = append(iterkeys, v)
		return LEVEL_0, nil
	}

	iterkeys = []interface{}{}
	if m3.Iterate(LEVEL_1, f3, "1") != nil {
		t.Errorf("err %#v not accepted", m3.Iterate(LEVEL_1, f3, "1"))
	}

	if !sliceEqual([]interface{}{"a", "b", "c", "d"}, iterkeys) {
		t.Errorf("%#v not accepted", iterkeys)
	}

	iterkeys = []interface{}{}
	if m3.Iterate(LEVEL_1, f3, "1", "1.1") != nil {
		t.Errorf("err %#v not accepted", m3.Iterate(LEVEL_1, f3, "1"))
	}

	if !sliceEqual([]interface{}{"a", "b"}, iterkeys) {
		t.Errorf("%#v not accepted", iterkeys)
	}

	iterkeys = []interface{}{}
	if m3.Iterate(LEVEL_1, f3, "1", "1.1", "1.1.1") != nil {
		t.Errorf("err %#v not accepted", m3.Iterate(LEVEL_1, f3, "1"))
	}

	if !sliceEqual([]interface{}{"a"}, iterkeys) {
		t.Errorf("%#v not accepted", iterkeys)
	}

	if m3.Iterate(LEVEL_1, f3, "1", "1.1", "1.1.3") != nil {
		t.Errorf("err %#v not accepted", m3.Iterate(LEVEL_1, f3, "1"))
	}

	if m3.Iterate(LEVEL_1, f3, "1", "1.3") != nil {
		t.Errorf("err %#v not accepted", m3.Iterate(LEVEL_1, f3, "1"))
	}

	if m3.Iterate(LEVEL_1, f3, "3") != nil {
		t.Errorf("err %#v not accepted", m3.Iterate(LEVEL_1, f3, "1"))
	}

	if !sliceEqual([]interface{}{}, m3.GetLeafValues(nil, "1", "1.1", "1.1.1", "1.1.1.1")) {
		t.Errorf("%#v not accepted", m3.GetLeafValues(nil, "1", "1.1", "1.1.1", "1.1.1.1"))
	}

	if !sliceEqual([]interface{}{"a"}, m3.GetLeafValues(nil, "1", "1.1", "1.1.1")) {
		t.Errorf("%#v not accepted", m3.GetLeafValues(nil, "1", "1.1", "1.1.1"))
	}

	if !sliceEqual([]interface{}{"e", "f"}, m3.GetLeafValues(nil, "2", "2.1")) {
		t.Errorf("%#v not accepted", m3.GetLeafValues(nil, "2", "2.1"))
	}

	if !sliceEqual([]interface{}{"a", "b", "c", "d"}, m3.GetLeafValues(nil, "1")) {
		t.Errorf("%#v not accepted", m3.GetLeafValues(nil, "1"))
	}

	if !sliceEqual([]interface{}{"a", "b", "c", "d", "e", "f", "g", "h"}, m3.GetLeafValues(nil)) {
		t.Errorf("%#v not accepted", m3.GetLeafValues(nil))
	}

	valueList := []string{}
	valueFunc := func(v interface{}) {
		valueList = append(valueList, v.(string))
	}

	if !sliceEqual([]interface{}{}, m3.GetLeafValues(valueFunc, "1")) {
		t.Errorf("%#v not accepted", m3.GetLeafValues(valueFunc, "1"))
	}
	if !strSliceEqual([]string{"a", "b", "c", "d"}, valueList) {
		t.Errorf("%#v not accepted", valueList)
	}

	//测试插入下级
	m2 := NewMmap2()
	m2.Insert("j", "3.1", "3.1.1")
	m2.Insert("k", "3.1", "3.1.2")
	m3.Insert(m2, "3")
	if !sliceEqual([]interface{}{"j", "k"}, m3.GetLeafValues(nil, "3")) {
		t.Errorf("%#v not accepted", m3.GetLeafValues(nil, "3"))
	}

	//测试插入下级,覆盖
	m2_1 := NewMmap2()
	m2_1.Insert("l", "3.2", "3.2.1")
	m2_1.Insert("m", "3.2", "3.2.2")
	m3.Insert(m2_1, "3")
	if !sliceEqual([]interface{}{"l", "m"}, m3.GetLeafValues(nil, "3")) {
		t.Errorf("%#v not accepted", m3.GetLeafValues(nil, "3"))
	}

	//测试插入下级,失败
	m1 := NewMmap1()
	if m3.Insert(m1, "4") {
		t.Errorf("Mmap3 not insert Mmap1")
	}

	//测试删除下级
	if !m3.Delete("3", "3.2", "3.2.3") {
		t.Errorf("delete not exist,should true")
	}
	if !sliceEqual([]interface{}{"l", "m"}, m3.GetLeafValues(nil, "3")) {
		t.Errorf("%#v not accepted", m3.GetLeafValues(nil, "3"))
	}
	if !m3.Delete("3", "3.2", "3.2.1") {
		t.Errorf("delete exist,should true")
	}
	if !sliceEqual([]interface{}{"m"}, m3.GetLeafValues(nil, "3")) {
		t.Errorf("%#v not accepted", m3.GetLeafValues(nil, "3"))
	}
	if !m3.Delete("3", "3.2") {
		t.Errorf("delete exist,should true")
	}
	if !sliceEqual([]interface{}{}, m3.GetLeafValues(nil, "3")) {
		t.Errorf("%#v not accepted", m3.GetLeafValues(nil, "3"))
	}

}

func sliceEqual(s1, s2 []interface{}) bool {
	for _, a := range s1 {
		bfind := false
		for _, b := range s2 {
			if a == b {
				bfind = true
				break
			}
		}
		if !bfind {
			return false
		}
	}

	for _, a := range s2 {
		bfind := false
		for _, b := range s1 {
			if a == b {
				bfind = true
				break
			}
		}
		if !bfind {
			return false
		}
	}

	return true
}

func strSliceEqual(s1, s2 []string) bool {
	for _, a := range s1 {
		bfind := false
		for _, b := range s2 {
			if a == b {
				bfind = true
				break
			}
		}
		if !bfind {
			return false
		}
	}

	for _, a := range s2 {
		bfind := false
		for _, b := range s1 {
			if a == b {
				bfind = true
				break
			}
		}
		if !bfind {
			return false
		}
	}

	return true
}
