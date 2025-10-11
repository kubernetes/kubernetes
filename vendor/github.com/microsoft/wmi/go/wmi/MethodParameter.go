package wmi

import (
	"errors"
	"reflect"
)

// MethodParameter
type MethodParameter struct {
	Name  string
	Value interface{}
	Type  WmiType
}

// MethodParameterCollection
type MethodParameterCollection []MethodParameter

// GetValue
func (c MethodParameterCollection) GetValue(paramName string, value interface{}) error {
	mval, err := c.Get(paramName)
	if err != nil {
		return errors.New("Not Found")
	}
	value = mval.Value
	return nil
}

// GetValueArray
func (c MethodParameterCollection) GetValueArray(paramName string, value interface{}) error {
	mval, err := c.Get(paramName)
	if err != nil {
		return errors.New("Not Found")
	}
	if mval.Value != nil {
		v := reflect.ValueOf(mval.Value)

		tmpValue := make([]interface{}, v.Len())
		for i := 0; i < v.Len(); i++ {
			tmpValue[i] = v.Index(i).Interface()
			value = tmpValue
		}
	} else {
		value = make([]interface{}, 0)
	}
	return nil
}

// Contains
func (c MethodParameterCollection) Contains(paramName string) bool {
	for _, a := range c {
		if a.Name == paramName {
			return true
		}
	}
	return false
}

// Contains
func (c MethodParameterCollection) Get(paramName string) (val *MethodParameter, err error) {
	for _, a := range c {
		if a.Name == paramName {
			return &a, nil
		}
	}
	return nil, errors.New("Not Found")
}
