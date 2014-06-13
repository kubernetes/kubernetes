// Copyright 2014 Docker authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the DOCKER-LICENSE file.

package engine

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
)

type Env []string

func (env *Env) Get(key string) (value string) {
	// FIXME: use Map()
	for _, kv := range *env {
		if strings.Index(kv, "=") == -1 {
			continue
		}
		parts := strings.SplitN(kv, "=", 2)
		if parts[0] != key {
			continue
		}
		if len(parts) < 2 {
			value = ""
		} else {
			value = parts[1]
		}
	}
	return
}

func (env *Env) Exists(key string) bool {
	_, exists := env.Map()[key]
	return exists
}

func (env *Env) GetBool(key string) (value bool) {
	s := strings.ToLower(strings.Trim(env.Get(key), " \t"))
	if s == "" || s == "0" || s == "no" || s == "false" || s == "none" {
		return false
	}
	return true
}

func (env *Env) SetBool(key string, value bool) {
	if value {
		env.Set(key, "1")
	} else {
		env.Set(key, "0")
	}
}

func (env *Env) GetInt(key string) int {
	return int(env.GetInt64(key))
}

func (env *Env) GetInt64(key string) int64 {
	s := strings.Trim(env.Get(key), " \t")
	val, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return -1
	}
	return val
}

func (env *Env) SetInt(key string, value int) {
	env.Set(key, fmt.Sprintf("%d", value))
}

func (env *Env) SetInt64(key string, value int64) {
	env.Set(key, fmt.Sprintf("%d", value))
}

// Returns nil if key not found
func (env *Env) GetList(key string) []string {
	sval := env.Get(key)
	if sval == "" {
		return nil
	}
	l := make([]string, 0, 1)
	if err := json.Unmarshal([]byte(sval), &l); err != nil {
		l = append(l, sval)
	}
	return l
}

func (env *Env) GetJson(key string, iface interface{}) error {
	sval := env.Get(key)
	if sval == "" {
		return nil
	}
	return json.Unmarshal([]byte(sval), iface)
}

func (env *Env) SetJson(key string, value interface{}) error {
	sval, err := json.Marshal(value)
	if err != nil {
		return err
	}
	env.Set(key, string(sval))
	return nil
}

func (env *Env) SetList(key string, value []string) error {
	return env.SetJson(key, value)
}

func (env *Env) Set(key, value string) {
	*env = append(*env, key+"="+value)
}

func NewDecoder(src io.Reader) *Decoder {
	return &Decoder{
		json.NewDecoder(src),
	}
}

type Decoder struct {
	*json.Decoder
}

func (decoder *Decoder) Decode() (*Env, error) {
	m := make(map[string]interface{})
	if err := decoder.Decoder.Decode(&m); err != nil {
		return nil, err
	}
	env := &Env{}
	for key, value := range m {
		env.SetAuto(key, value)
	}
	return env, nil
}

// DecodeEnv decodes `src` as a json dictionary, and adds
// each decoded key-value pair to the environment.
//
// If `src` cannot be decoded as a json dictionary, an error
// is returned.
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

func (env *Env) SetAuto(k string, v interface{}) {
	// FIXME: we fix-convert float values to int, because
	// encoding/json decodes integers to float64, but cannot encode them back.
	// (See http://golang.org/src/pkg/encoding/json/decode.go#L46)
	if fval, ok := v.(float64); ok {
		env.SetInt64(k, int64(fval))
	} else if sval, ok := v.(string); ok {
		env.Set(k, sval)
	} else if val, err := json.Marshal(v); err == nil {
		env.Set(k, string(val))
	} else {
		env.Set(k, fmt.Sprintf("%v", v))
	}
}

func (env *Env) Encode(dst io.Writer) error {
	m := make(map[string]interface{})
	for k, v := range env.Map() {
		var val interface{}
		if err := json.Unmarshal([]byte(v), &val); err == nil {
			// FIXME: we fix-convert float values to int, because
			// encoding/json decodes integers to float64, but cannot encode them back.
			// (See http://golang.org/src/pkg/encoding/json/decode.go#L46)
			if fval, isFloat := val.(float64); isFloat {
				val = int(fval)
			}
			m[k] = val
		} else {
			m[k] = v
		}
	}
	if err := json.NewEncoder(dst).Encode(&m); err != nil {
		return err
	}
	return nil
}

func (env *Env) WriteTo(dst io.Writer) (n int64, err error) {
	// FIXME: return the number of bytes written to respect io.WriterTo
	return 0, env.Encode(dst)
}

func (env *Env) Export(dst interface{}) (err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("ExportEnv %s", err)
		}
	}()
	var buf bytes.Buffer
	// step 1: encode/marshal the env to an intermediary json representation
	if err := env.Encode(&buf); err != nil {
		return err
	}
	// step 2: decode/unmarshal the intermediary json into the destination object
	if err := json.NewDecoder(&buf).Decode(dst); err != nil {
		return err
	}
	return nil
}

func (env *Env) Import(src interface{}) (err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("ImportEnv: %s", err)
		}
	}()
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(src); err != nil {
		return err
	}
	if err := env.Decode(&buf); err != nil {
		return err
	}
	return nil
}

func (env *Env) Map() map[string]string {
	m := make(map[string]string)
	for _, kv := range *env {
		parts := strings.SplitN(kv, "=", 2)
		m[parts[0]] = parts[1]
	}
	return m
}
