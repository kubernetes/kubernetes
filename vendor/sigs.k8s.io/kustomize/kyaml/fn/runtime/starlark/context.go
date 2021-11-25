// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package starlark

import (
	"encoding/json"
	"os"
	"strings"

	"go.starlark.net/starlark"
	"go.starlark.net/starlarkstruct"
	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/internal/forked/github.com/qri-io/starlib/util"
	"sigs.k8s.io/kustomize/kyaml/openapi"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

type Context struct {
	resourceList starlark.Value
}

func (c *Context) predeclared() (starlark.StringDict, error) {
	e, err := env()
	if err != nil {
		return nil, err
	}
	oa, err := oa()
	if err != nil {
		return nil, err
	}
	dict := starlark.StringDict{
		"resource_list": c.resourceList,
		"open_api":      oa,
		"environment":   e,
	}

	return starlark.StringDict{
		"ctx": starlarkstruct.FromStringDict(starlarkstruct.Default, dict),
	}, nil
}

func oa() (starlark.Value, error) {
	return interfaceToValue(openapi.Schema())
}

func env() (starlark.Value, error) {
	env := map[string]interface{}{}
	for _, e := range os.Environ() {
		pair := strings.SplitN(e, "=", 2)
		if len(pair) < 2 {
			continue
		}
		env[pair[0]] = pair[1]
	}
	value, err := util.Marshal(env)
	if err != nil {
		return nil, errors.Wrap(err)
	}
	return value, nil
}

func interfaceToValue(i interface{}) (starlark.Value, error) {
	b, err := json.Marshal(i)
	if err != nil {
		return nil, err
	}

	var in map[string]interface{}
	if err := yaml.Unmarshal(b, &in); err != nil {
		return nil, errors.Wrap(err)
	}

	value, err := util.Marshal(in)
	if err != nil {
		return nil, errors.Wrap(err)
	}
	return value, nil
}
