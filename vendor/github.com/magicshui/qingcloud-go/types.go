package qingcloud

import (
	"strings"
)

type String struct {
	value string
	enums map[string]bool
	write bool
}

// TODO:
func (s *String) Set(t string) {
	if len(s.enums) != 0 {
		if _, ok := s.enums[t]; ok {
			s.value = t
			s.write = true
		}
	} else {
		s.value = t
		s.write = true
	}
}

// 对设置的内容进行修改
func (s *String) Enum(t ...string) {
	if s.enums == nil {
		s.enums = make(map[string]bool)
		for _, o := range t {
			s.enums[o] = true
		}
	}
	if _, ok := s.enums[s.value]; !ok {
		s.value = ""
		s.write = false
	}
}

func (s *String) String() string {
	return s.value
}

// TODO 实现这个类型
type NumberedInteger struct {
	values map[int64]bool
	enums  map[int64]bool
	write  bool
}

func (s *NumberedInteger) Add(t ...int64) {
	if s.enums == nil {
		s.enums = make(map[int64]bool)
	}
	if s.values == nil {
		s.values = make(map[int64]bool)
	}
	for _, o := range t {
		if len(s.enums) != 0 {
			if _, ok := s.enums[o]; ok {
				s.values[o] = true
			}
		} else {
			s.values[o] = true
		}
	}
}
func (s *NumberedInteger) Enum(e ...int64) {
	if s.enums == nil {
		s.enums = make(map[int64]bool)
	}
	for k, _ := range s.values {
		if _, ok := s.enums[k]; !ok {
			delete(s.values, k)
		}
	}
	if len(s.values) == 0 {
		s.write = false
	}
}

type NumberedString struct {
	values map[string]bool
	enums  map[string]bool
	write  bool
}

func (s *NumberedString) String() string {
	var a = []string{}
	for i, _ := range s.values {
		a = append(a, i)
	}
	return strings.Join(a, ",")
}
func (s *NumberedString) Add(t ...string) {
	if s.enums == nil {
		s.enums = make(map[string]bool)
	}
	if s.values == nil {
		s.values = make(map[string]bool)
	}

	for _, o := range t {
		if len(s.enums) != 0 {
			if _, ok := s.enums[o]; ok {
				s.values[o] = true
			}
		} else {
			s.values[o] = true
		}
	}
}
func (s *NumberedString) Enum(e ...string) {
	if s.enums == nil {
		s.enums = make(map[string]bool)
	}
	
	for _, o := range e {
		s.enums[o] = true
	}
	
	for k, _ := range s.values {
		if _, ok := s.enums[k]; !ok {
			delete(s.values, k)
		}
	}
	if len(s.values) == 0 {
		s.write = false
	}
}

type Integer struct {
	value int
	enums map[int]bool
	write bool
}

func (s *Integer) Set(t int) {
	if len(s.enums) != 0 {
		if _, ok := s.enums[t]; ok {
			s.value = t
			s.write = true
		}
	} else {
		s.value = t
		s.write = true
	}
}

func (s *Integer) Enum(e ...int) {
	if s.enums == nil {
		s.enums = make(map[int]bool)
	}
	for _, o := range e {
		s.enums[o] = true
	}
}

type Dict struct {
	values []map[string]interface{}
}

func (c *Dict) Add(t map[string]interface{}) {
	if len(c.values) == 0 {
		c.values = make([]map[string]interface{}, 0)
	}
	c.values = append(c.values, t)
}

// TODO 实现这个类型

type Array struct{}
