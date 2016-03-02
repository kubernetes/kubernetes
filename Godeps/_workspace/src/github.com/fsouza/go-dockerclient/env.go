// Copyright 2014 Docker authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the DOCKER-LICENSE file.

package docker

import (
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// Env represents a list of key-pair represented in the form KEY=VALUE.
type Env []string

// Get returns the string value of the given key.
func (env *Env) Get(key string) (value string) {
	return env.Map()[key]
}

// Exists checks whether the given key is defined in the internal Env
// representation.
func (env *Env) Exists(key string) bool {
	_, exists := env.Map()[key]
	return exists
}

// GetBool returns a boolean representation of the given key. The key is false
// whenever its value if 0, no, false, none or an empty string. Any other value
// will be interpreted as true.
func (env *Env) GetBool(key string) (value bool) {
	s := strings.ToLower(strings.Trim(env.Get(key), " \t"))
	if s == "" || s == "0" || s == "no" || s == "false" || s == "none" {
		return false
	}
	return true
}

// SetBool defines a boolean value to the given key.
func (env *Env) SetBool(key string, value bool) {
	if value {
		env.Set(key, "1")
	} else {
		env.Set(key, "0")
	}
}

// GetInt returns the value of the provided key, converted to int.
//
// It the value cannot be represented as an integer, it returns -1.
func (env *Env) GetInt(key string) int {
	return int(env.GetInt64(key))
}

// SetInt defines an integer value to the given key.
func (env *Env) SetInt(key string, value int) {
	env.Set(key, strconv.Itoa(value))
}

// GetInt64 returns the value of the provided key, converted to int64.
//
// It the value cannot be represented as an integer, it returns -1.
func (env *Env) GetInt64(key string) int64 {
	s := strings.Trim(env.Get(key), " \t")
	val, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return -1
	}
	return val
}

// SetInt64 defines an integer (64-bit wide) value to the given key.
func (env *Env) SetInt64(key string, value int64) {
	env.Set(key, strconv.FormatInt(value, 10))
}

// GetJSON unmarshals the value of the provided key in the provided iface.
//
// iface is a value that can be provided to the json.Unmarshal function.
func (env *Env) GetJSON(key string, iface interface{}) error {
	sval := env.Get(key)
	if sval == "" {
		return nil
	}
	return json.Unmarshal([]byte(sval), iface)
}

// SetJSON marshals the given value to JSON format and stores it using the
// provided key.
func (env *Env) SetJSON(key string, value interface{}) error {
	sval, err := json.Marshal(value)
	if err != nil {
		return err
	}
	env.Set(key, string(sval))
	return nil
}

// GetList returns a list of strings matching the provided key. It handles the
// list as a JSON representation of a list of strings.
//
// If the given key matches to a single string, it will return a list
// containing only the value that matches the key.
func (env *Env) GetList(key string) []string {
	sval := env.Get(key)
	if sval == "" {
		return nil
	}
	var l []string
	if err := json.Unmarshal([]byte(sval), &l); err != nil {
		l = append(l, sval)
	}
	return l
}

// SetList stores the given list in the provided key, after serializing it to
// JSON format.
func (env *Env) SetList(key string, value []string) error {
	return env.SetJSON(key, value)
}

// Set defines the value of a key to the given string.
func (env *Env) Set(key, value string) {
	*env = append(*env, key+"="+value)
}

// Decode decodes `src` as a json dictionary, and adds each decoded key-value
// pair to the environment.
//
// If `src` cannot be decoded as a json dictionary, an error is returned.
func (env *Env) Decode(src io.Reader) error {
	m := make(map[string]interface{})
	if err := json.NewDecoder(src).Decode(&m); err != nil {
		return err
	}
	for k, v := range m {
		env.SetAuto(k, v)
	}
	return nil
}

// SetAuto will try to define the Set* method to call based on the given value.
func (env *Env) SetAuto(key string, value interface{}) {
	if fval, ok := value.(float64); ok {
		env.SetInt64(key, int64(fval))
	} else if sval, ok := value.(string); ok {
		env.Set(key, sval)
	} else if val, err := json.Marshal(value); err == nil {
		env.Set(key, string(val))
	} else {
		env.Set(key, fmt.Sprintf("%v", value))
	}
}

// Map returns the map representation of the env.
func (env *Env) Map() map[string]string {
	if len(*env) == 0 {
		return nil
	}
	m := make(map[string]string)
	for _, kv := range *env {
		parts := strings.SplitN(kv, "=", 2)
		m[parts[0]] = parts[1]
	}
	return m
}
