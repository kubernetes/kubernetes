/*
Copyright 2023 The Kubernetes Authors.

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

package apply

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
)

var defaultApplySetParentGVR = schema.GroupVersionResource{Version: "v1", Resource: "secrets"}

// ApplySet tracks the information about an applyset apply/prune
type ApplySet struct {
	// ParentRef is the reference to the parent object that is used to track the applyset.
	ParentRef *ApplySetParentRef

	// resources is the set of all the resources that (might) be part of this applyset.
	resources map[schema.GroupVersionResource]struct{}

	// namespaces is the set of all namespaces that (might) contain objects that are part of this applyset.
	namespaces map[string]struct{}
}

var builtinApplySetParentGVRs = map[schema.GroupVersionResource]bool{
	defaultApplySetParentGVR:                true,
	{Version: "v1", Resource: "configmaps"}: true,
}

// ApplySetParentRef stores object and type meta for the parent object that is used to track the applyset.
type ApplySetParentRef struct {
	Name        string
	Namespace   string
	RESTMapping *meta.RESTMapping
}

// NewApplySet creates a new ApplySet object from a parent reference in the format [RESOURCE][.GROUP]/NAME
func NewApplySet(parentRefStr string, nsFromFlag string, mapper meta.RESTMapper) (*ApplySet, error) {
	parent, err := parentRefFromStr(parentRefStr, mapper)
	if err != nil {
		return nil, fmt.Errorf("invalid parent reference %q: %w", parentRefStr, err)
	}
	if parent.IsNamespaced() {
		parent.Namespace = nsFromFlag
	}
	return &ApplySet{
		resources:  make(map[schema.GroupVersionResource]struct{}),
		namespaces: make(map[string]struct{}),
		ParentRef:  parent,
	}, nil
}

// ID is the label value that we are using to identify this applyset.
func (a ApplySet) ID() string {
	// TODO: base64(sha256(gknn))
	return "placeholder-todo"
}

// Validate imposes restrictions on the parent object that is used to track the applyset.
func (a ApplySet) Validate() error {
	var errors []error
	// TODO: permit CRDs that have the annotation required by the ApplySet specification
	if !builtinApplySetParentGVRs[a.ParentRef.RESTMapping.Resource] {
		errors = append(errors, fmt.Errorf("resource %q is not permitted as an ApplySet parent", a.ParentRef.RESTMapping.Resource))
	}
	if a.ParentRef.IsNamespaced() && a.ParentRef.Namespace == "" {
		errors = append(errors, fmt.Errorf("namespace is required to use namespace-scoped ApplySet"))
	}
	return utilerrors.NewAggregate(errors)
}

func (p *ApplySetParentRef) IsNamespaced() bool {
	return p.RESTMapping.Scope.Name() == meta.RESTScopeNameNamespace
}

// parentRefFromStr creates a new ApplySetParentRef from a parent reference in the format [RESOURCE][.GROUP]/NAME
func parentRefFromStr(parentRefStr string, mapper meta.RESTMapper) (*ApplySetParentRef, error) {
	var gvr schema.GroupVersionResource
	var name string

	if groupRes, nameSuffix, hasTypeInfo := strings.Cut(parentRefStr, "/"); hasTypeInfo {
		name = nameSuffix
		gvr = schema.ParseGroupResource(groupRes).WithVersion("")
	} else {
		name = parentRefStr
		gvr = defaultApplySetParentGVR
	}

	if name == "" {
		return nil, fmt.Errorf("name cannot be blank")
	}

	gvk, err := mapper.KindFor(gvr)
	if err != nil {
		return nil, err
	}
	mapping, err := mapper.RESTMapping(gvk.GroupKind())
	if err != nil {
		return nil, err
	}
	return &ApplySetParentRef{Name: name, RESTMapping: mapping}, nil
}
