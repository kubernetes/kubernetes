/*
Copyright 2018 The Kubernetes Authors.

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

package transformer

import (
	"fmt"

	"github.com/evanphx/json-patch"
	"github.com/ghodss/yaml"
	"github.com/pkg/errors"
	"sigs.k8s.io/kustomize/pkg/resid"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/resource"
	"sigs.k8s.io/kustomize/pkg/transformers"
)

// patchJson6902JSONTransformer applies patches.
type patchJson6902JSONTransformer struct {
	target resid.ResId
	patch  jsonpatch.Patch
	rawOp  []byte
}

var _ transformers.Transformer = &patchJson6902JSONTransformer{}

// newPatchJson6902JSONTransformer constructs a PatchJson6902 transformer.
func newPatchJson6902JSONTransformer(
	id resid.ResId, rawOp []byte) (transformers.Transformer, error) {
	op := rawOp
	var err error
	if !isJsonFormat(op) {
		// if it isn't JSON, try to parse it as YAML
		op, err = yaml.YAMLToJSON(rawOp)
		if err != nil {
			return nil, err
		}
	}
	decodedPatch, err := jsonpatch.DecodePatch(op)
	if err != nil {
		return nil, err
	}
	if len(decodedPatch) == 0 {
		return transformers.NewNoOpTransformer(), nil
	}
	return &patchJson6902JSONTransformer{target: id, patch: decodedPatch, rawOp: rawOp}, nil
}

// Transform apply the json patches on top of the base resources.
func (t *patchJson6902JSONTransformer) Transform(m resmap.ResMap) error {
	obj, err := t.findTargetObj(m)
	if err != nil {
		return err
	}
	rawObj, err := obj.MarshalJSON()
	if err != nil {
		return err
	}
	modifiedObj, err := t.patch.Apply(rawObj)
	if err != nil {
		return errors.Wrapf(err, "failed to apply json patch '%s'", string(t.rawOp))
	}
	err = obj.UnmarshalJSON(modifiedObj)
	if err != nil {
		return err
	}
	return nil
}

func (t *patchJson6902JSONTransformer) findTargetObj(
	m resmap.ResMap) (*resource.Resource, error) {
	var matched []resid.ResId
	// TODO(monopole): namespace bug in json patch?
	// Since introduction in PR #300
	// (see pkg/patch/transformer/util.go),
	// this code has treated an empty namespace like a wildcard
	// rather than like an additional restriction to match
	// only the empty namespace.  No test coverage to confirm.
	// Not sure if desired, keeping it for now.
	if t.target.Namespace() != "" {
		matched = m.GetMatchingIds(t.target.NsGvknEquals)
	} else {
		matched = m.GetMatchingIds(t.target.GvknEquals)
	}
	if len(matched) == 0 {
		return nil, fmt.Errorf(
			"couldn't find target %v for json patch", t.target)
	}
	if len(matched) > 1 {
		return nil, fmt.Errorf(
			"found multiple targets %v matching %v for json patch",
			matched, t.target)
	}
	return m[matched[0]], nil
}
