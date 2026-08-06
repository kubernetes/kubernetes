/*
Copyright 2025 The Kubernetes Authors.

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

package cel

import (
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"

	"k8s.io/apiserver/pkg/authentication/user"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/lazy"
)

// NewUserInfoMapper creates a traits.Mapper from a user.Info that can be used
// as input when evaluating CEL user validation expressions. The returned mapper
// supports both direct string-key access (e.g. user.extra['example.org/foo'])
// and escaped identifier-based access (e.g. user.extra.example__dot__org__slash__foo).
func NewUserInfoMapper(info user.Info) traits.Mapper {
	lazyMap := lazy.NewMapValue(types.NewObjectType("kubernetes.UserInfo"))
	addField := func(name string, get func() any) {
		lazyMap.Append(name, func(_ *lazy.MapValue) ref.Val {
			return userInfoNativeToValue(get())
		})
	}
	addField("username", func() any { return info.GetName() })
	addField("uid", func() any { return info.GetUID() })
	addField("groups", func() any { return info.GetGroups() })
	addField("extra", func() any { return info.GetExtra() })
	return lazyMap
}

// userInfoNativeToValue converts a Go value to a CEL ref.Val, wrapping maps
// and lists to support __dot__-escaped key lookups.
func userInfoNativeToValue(value any) ref.Val {
	return userInfoUnescapeWrapper(types.DefaultTypeAdapter.NativeToValue(value))
}

// userInfoUnescapeWrapper wraps a CEL value to support __dot__-escaped key lookups
// for maps and lists.
func userInfoUnescapeWrapper(value ref.Val) ref.Val {
	switch v := value.(type) {
	case traits.Mapper:
		return &userInfoUnescapeMapper{Mapper: v}
	case traits.Lister:
		return &userInfoUnescapeLister{Lister: v}
	default:
		return value
	}
}

type userInfoUnescapeMapper struct {
	traits.Mapper
}

func (m *userInfoUnescapeMapper) Find(key ref.Val) (ref.Val, bool) {
	name, ok := userInfoUnescapedName(key)
	if ok {
		key = name
	}
	value, ok := m.Mapper.Find(key)
	return userInfoUnescapeWrapper(value), ok
}

type userInfoUnescapeLister struct {
	traits.Lister
}

func (l *userInfoUnescapeLister) Get(index ref.Val) ref.Val {
	return userInfoUnescapeWrapper(l.Lister.Get(index))
}

func userInfoUnescapedName(key ref.Val) (types.String, bool) {
	n, ok := key.(types.String)
	if !ok {
		return "", false
	}
	ns := string(n)
	name, ok := apiservercel.Unescape(ns)
	if !ok || name == ns {
		return "", false
	}
	return types.String(name), true
}
