// Copyright 2013 Dario Castañé. All rights reserved.
// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Based on src/pkg/reflect/deepequal.go from official
// golang's stdlib.

package mergo

import "reflect"

func hasExportedField(dst reflect.Value) (exported bool) {
	for i, n := 0, dst.NumField(); i < n; i++ {
		field := dst.Type().Field(i)
		if field.Anonymous && dst.Field(i).Kind() == reflect.Struct {
			exported = exported || hasExportedField(dst.Field(i))
		} else {
			exported = exported || len(field.PkgPath) == 0
		}
	}
	return
}

type config struct {
	overwrite    bool
	transformers transformers
}

type transformers interface {
	Transformer(reflect.Type) func(dst, src reflect.Value) error
}

// Traverses recursively both values, assigning src's fields values to dst.
// The map argument tracks comparisons that have already been seen, which allows
// short circuiting on recursive types.
func deepMerge(dst, src reflect.Value, visited map[uintptr]*visit, depth int, config *config) (err error) {
	overwrite := config.overwrite

	if !src.IsValid() {
		return
	}
	if dst.CanAddr() {
		addr := dst.UnsafeAddr()
		h := 17 * addr
		seen := visited[h]
		typ := dst.Type()
		for p := seen; p != nil; p = p.next {
			if p.ptr == addr && p.typ == typ {
				return nil
			}
		}
		// Remember, remember...
		visited[h] = &visit{addr, typ, seen}
	}

	if config.transformers != nil && !isEmptyValue(dst) {
		if fn := config.transformers.Transformer(dst.Type()); fn != nil {
			err = fn(dst, src)
			return
		}
	}

	switch dst.Kind() {
	case reflect.Struct:
		if hasExportedField(dst) {
			for i, n := 0, dst.NumField(); i < n; i++ {
				if err = deepMerge(dst.Field(i), src.Field(i), visited, depth+1, config); err != nil {
					return
				}
			}
		} else {
			if dst.CanSet() && !isEmptyValue(src) && (overwrite || isEmptyValue(dst)) {
				dst.Set(src)
			}
		}
	case reflect.Map:
		if len(src.MapKeys()) == 0 && !src.IsNil() && len(dst.MapKeys()) == 0 {
			dst.Set(reflect.MakeMap(dst.Type()))
			return
		}
		for _, key := range src.MapKeys() {
			srcElement := src.MapIndex(key)
			if !srcElement.IsValid() {
				continue
			}
			dstElement := dst.MapIndex(key)
			switch srcElement.Kind() {
			case reflect.Chan, reflect.Func, reflect.Map, reflect.Ptr, reflect.Interface, reflect.Slice:
				if srcElement.IsNil() {
					continue
				}
				fallthrough
			default:
				if !srcElement.CanInterface() {
					continue
				}
				switch reflect.TypeOf(srcElement.Interface()).Kind() {
				case reflect.Struct:
					fallthrough
				case reflect.Ptr:
					fallthrough
				case reflect.Map:
					if err = deepMerge(dstElement, srcElement, visited, depth+1, config); err != nil {
						return
					}
				case reflect.Slice:
					srcSlice := reflect.ValueOf(srcElement.Interface())

					var dstSlice reflect.Value
					if !dstElement.IsValid() || dstElement.IsNil() {
						dstSlice = reflect.MakeSlice(srcSlice.Type(), 0, srcSlice.Len())
					} else {
						dstSlice = reflect.ValueOf(dstElement.Interface())
					}

					dstSlice = reflect.AppendSlice(dstSlice, srcSlice)
					dst.SetMapIndex(key, dstSlice)
				}
			}
			if dstElement.IsValid() && reflect.TypeOf(srcElement.Interface()).Kind() == reflect.Map {
				continue
			}

			if !isEmptyValue(srcElement) && (overwrite || (!dstElement.IsValid() || isEmptyValue(dst))) {
				if dst.IsNil() {
					dst.Set(reflect.MakeMap(dst.Type()))
				}
				dst.SetMapIndex(key, srcElement)
			}
		}
	case reflect.Slice:
		dst.Set(reflect.AppendSlice(dst, src))
	case reflect.Ptr:
		fallthrough
	case reflect.Interface:
		if src.IsNil() {
			break
		}
		if src.Kind() != reflect.Interface {
			if dst.IsNil() || overwrite {
				if dst.CanSet() && (overwrite || isEmptyValue(dst)) {
					dst.Set(src)
				}
			} else if src.Kind() == reflect.Ptr {
				if err = deepMerge(dst.Elem(), src.Elem(), visited, depth+1, config); err != nil {
					return
				}
			} else if dst.Elem().Type() == src.Type() {
				if err = deepMerge(dst.Elem(), src, visited, depth+1, config); err != nil {
					return
				}
			} else {
				return ErrDifferentArgumentsTypes
			}
			break
		}
		if dst.IsNil() || overwrite {
			if dst.CanSet() && (overwrite || isEmptyValue(dst)) {
				dst.Set(src)
			}
		} else if err = deepMerge(dst.Elem(), src.Elem(), visited, depth+1, config); err != nil {
			return
		}
	default:
		if dst.CanSet() && !isEmptyValue(src) && (overwrite || isEmptyValue(dst)) {
			dst.Set(src)
		}
	}
	return
}

// Merge will fill any empty for value type attributes on the dst struct using corresponding
// src attributes if they themselves are not empty. dst and src must be valid same-type structs
// and dst must be a pointer to struct.
// It won't merge unexported (private) fields and will do recursively any exported field.
func Merge(dst, src interface{}, opts ...func(*config)) error {
	return merge(dst, src, opts...)
}

// MergeWithOverwrite will do the same as Merge except that non-empty dst attributes will be overriden by
// non-empty src attribute values.
// Deprecated: use Merge(…) with WithOverride
func MergeWithOverwrite(dst, src interface{}, opts ...func(*config)) error {
	return merge(dst, src, append(opts, WithOverride)...)
}

// WithTransformers adds transformers to merge, allowing to customize the merging of some types.
func WithTransformers(transformers transformers) func(*config) {
	return func(config *config) {
		config.transformers = transformers
	}
}

// WithOverride will make merge override non-empty dst attributes with non-empty src attributes values.
func WithOverride(config *config) {
	config.overwrite = true
}

func merge(dst, src interface{}, opts ...func(*config)) error {
	var (
		vDst, vSrc reflect.Value
		err        error
	)

	config := &config{}

	for _, opt := range opts {
		opt(config)
	}

	if vDst, vSrc, err = resolveValues(dst, src); err != nil {
		return err
	}
	if vDst.Type() != vSrc.Type() {
		return ErrDifferentArgumentsTypes
	}
	return deepMerge(vDst, vSrc, make(map[uintptr]*visit), 0, config)
}
