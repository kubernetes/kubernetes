/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package protectedattributes

import (
	"fmt"
	"io"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/rbac/validation"
	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/watch"
)

const reflectorResyncPeriod = 5 * time.Minute

func init() {
	pluginFactory := func(client clientset.Interface, config io.Reader) (admission.Interface, error) {
		return NewProtectedAttributes(client), nil
	}

	admission.RegisterPlugin("ProtectedAttributes", pluginFactory)
}

// protectedAttributesController admission controller checks if
// resource being admitted contains any protected attributes (as
// defined by rbac.ProtectedAttribute and
// rbac.ClusterProtectedAttribute). For any protected attributes
// present, it checks that requester has required roles (as defined by
// rbac.Role and rbac.ClusterRole) to create, update, or delete
// protected attributes on that resource.
type protectedAttributesController struct {
	*admission.Handler
	client clientset.Interface

	roleBindings        cache.Indexer
	clusterRoleBindings cache.Store

	protectedAttributes        cache.Indexer
	clusterProtectedAttributes cache.Store
}

// NewProtectedAttributes returns an instance of a protectedAttributes
// admission controller.
func NewProtectedAttributes(c clientset.Interface) admission.Interface {
	roleBindingsIndexer, roleBindingsReflector := cache.NewNamespaceKeyedIndexerAndReflector(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return c.Rbac().RoleBindings(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return c.Rbac().RoleBindings(api.NamespaceAll).Watch(options)
			},
		},
		&rbac.RoleBinding{},
		reflectorResyncPeriod,
	)

	clusterRoleBindingsStore := cache.NewStore(cache.MetaNamespaceKeyFunc)
	clusterRoleBindingsReflector := cache.NewReflector(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return c.Rbac().ClusterRoleBindings().List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return c.Rbac().ClusterRoleBindings().Watch(options)
			},
		},
		&rbac.ClusterRoleBinding{},
		clusterRoleBindingsStore,
		reflectorResyncPeriod,
	)

	protectedAttributesIndexer, protectedAttributesReflector := cache.NewNamespaceKeyedIndexerAndReflector(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return c.Rbac().ProtectedAttributes(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return c.Rbac().ProtectedAttributes(api.NamespaceAll).Watch(options)
			},
		},
		&rbac.ProtectedAttribute{},
		reflectorResyncPeriod,
	)

	clusterProtectedAttributesStore := cache.NewStore(cache.MetaNamespaceKeyFunc)
	clusterProtectedAttributesReflector := cache.NewReflector(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return c.Rbac().ClusterProtectedAttributes().List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return c.Rbac().ClusterProtectedAttributes().Watch(options)
			},
		},
		&rbac.ClusterProtectedAttribute{},
		clusterProtectedAttributesStore,
		reflectorResyncPeriod,
	)

	roleBindingsReflector.Run()
	clusterRoleBindingsReflector.Run()
	protectedAttributesReflector.Run()
	clusterProtectedAttributesReflector.Run()

	return &protectedAttributesController{
		Handler:                    admission.NewHandler(admission.Create, admission.Update, admission.Delete),
		client:                     c,
		roleBindings:               roleBindingsIndexer,
		clusterRoleBindings:        clusterRoleBindingsStore,
		protectedAttributes:        protectedAttributesIndexer,
		clusterProtectedAttributes: clusterProtectedAttributesStore,
	}
}

func (p *protectedAttributesController) Admit(a admission.Attributes) error {
	protectedAttributes, clusterProtectedAttributes, err := p.getProtectedAttributes(a.GetNamespace())
	if err != nil {
		return err
	}

	if len(protectedAttributes) == 0 && len(clusterProtectedAttributes) == 0 {
		// There are no protected attributes, so no checks are required.
		return nil
	}

	roleNames, clusterRoleNames, err := p.getRoleNames(a.GetNamespace(), a.GetUserInfo())
	if err != nil {
		return err
	}

	protectedLabels := map[string][]*rbac.ProtectedAttribute{}
	clusterProtectedLabels := map[string][]*rbac.ClusterProtectedAttribute{}

	protectedAnnotations := map[string][]*rbac.ProtectedAttribute{}
	clusterProtectedAnnotations := map[string][]*rbac.ClusterProtectedAttribute{}

	// Categorize protected attributes for efficient lookups.
	for _, pa := range protectedAttributes {
		switch pa.AttributeKind {
		case rbac.LabelKind:
			protectedLabels[pa.AttributeName] = append(protectedLabels[pa.AttributeName], pa)
		case rbac.AnnotationKind:
			protectedAnnotations[pa.AttributeName] = append(protectedAnnotations[pa.AttributeName], pa)
		}
	}

	for _, cpa := range clusterProtectedAttributes {
		switch cpa.AttributeKind {
		case rbac.LabelKind:
			clusterProtectedLabels[cpa.AttributeName] = append(clusterProtectedLabels[cpa.AttributeName], cpa)
		case rbac.AnnotationKind:
			clusterProtectedAnnotations[cpa.AttributeName] = append(clusterProtectedAnnotations[cpa.AttributeName], cpa)
		}
	}

	// Creates and updates provide the payload for the new resource
	// via a.GetObject(). We need to check if there are any protected
	// attribute violations in that object.
	if a.GetOperation() == admission.Create || a.GetOperation() == admission.Update {
		err := checkAttributes(a.GetObject(),
			roleNames, clusterRoleNames,
			protectedLabels, clusterProtectedLabels,
			protectedAnnotations, clusterProtectedAnnotations,
		)
		if err != nil {
			return admission.NewForbidden(a, err)
		}
	}

	// Updates and deletes need to check whether the object being
	// updated/deleted has any protected attribute violations: we
	// don't want users to be able to update/delete something unless
	// they are authorized.
	if a.GetOperation() == admission.Update || a.GetOperation() == admission.Delete {
		// FIXME(olegshaldybin): actually populate GetOldObject() for deletes.
		if a.GetOldObject() == nil && a.GetOperation() == admission.Delete {
			return nil
		}
		err := checkAttributes(a.GetOldObject(),
			roleNames, clusterRoleNames,
			protectedLabels, clusterProtectedLabels,
			protectedAnnotations, clusterProtectedAnnotations,
		)
		if err != nil {
			return admission.NewForbidden(a, err)
		}
	}

	return nil
}

func (p *protectedAttributesController) getProtectedAttributes(ns string) ([]*rbac.ProtectedAttribute, []*rbac.ClusterProtectedAttribute, error) {
	var protectedAttributes []*rbac.ProtectedAttribute
	var clusterProtectedAttributes []*rbac.ClusterProtectedAttribute

	nsKey := &rbac.ProtectedAttribute{ObjectMeta: api.ObjectMeta{Namespace: ns}}
	paList, err := p.protectedAttributes.Index("namespace", nsKey)
	if err != nil {
		return nil, nil, err
	}

	if len(paList) == 0 {
		// Not in cache, try retrieving directly.
		paListDirect, err := p.client.Rbac().ProtectedAttributes(ns).List(api.ListOptions{})
		if err != nil {
			return nil, nil, err
		}
		for i := range paListDirect.Items {
			protectedAttributes = append(protectedAttributes, &paListDirect.Items[i])
		}
	} else {
		for _, pa := range paList {
			protectedAttributes = append(protectedAttributes, pa.(*rbac.ProtectedAttribute))
		}
	}

	cpaList := p.clusterProtectedAttributes.List()
	if len(cpaList) == 0 {
		// Not in cache, try retrieving directly.
		cpaListDirect, err := p.client.Rbac().ClusterProtectedAttributes().List(api.ListOptions{})
		if err != nil {
			return nil, nil, err
		}
		for i := range cpaListDirect.Items {
			clusterProtectedAttributes = append(clusterProtectedAttributes, &cpaListDirect.Items[i])
		}
	} else {
		for _, cpa := range cpaList {
			clusterProtectedAttributes = append(clusterProtectedAttributes, cpa.(*rbac.ClusterProtectedAttribute))
		}
	}

	return protectedAttributes, clusterProtectedAttributes, nil
}

func (p *protectedAttributesController) getRoleNames(ns string, user user.Info) (sets.String, sets.String, error) {
	roleNames := sets.NewString()
	clusterRoleNames := sets.NewString()

	if user == nil {
		// No user, hence no roles.
		return roleNames, clusterRoleNames, nil
	}

	var clusterRoleBindings []*rbac.ClusterRoleBinding
	clusterRoleBindingsList := p.clusterRoleBindings.List()
	if len(clusterRoleBindingsList) == 0 {
		// Not in cache, try retrieving directly.
		clusterRoleBindingsDirect, err := p.client.Rbac().ClusterRoleBindings().List(api.ListOptions{})
		if err != nil {
			return nil, nil, err
		}
		for i := range clusterRoleBindingsDirect.Items {
			clusterRoleBindings = append(clusterRoleBindings, &clusterRoleBindingsDirect.Items[i])
		}
	} else {
		for _, crb := range clusterRoleBindingsList {
			clusterRoleBindings = append(clusterRoleBindings, crb.(*rbac.ClusterRoleBinding))
		}
	}

	for _, crb := range clusterRoleBindings {
		ok, err := validation.ClusterRoleBindingMatches(crb, user)
		if err != nil {
			return nil, nil, err
		}
		if ok {
			clusterRoleNames.Insert(crb.RoleRef.Name)
		}
	}

	nsKey := &rbac.RoleBinding{ObjectMeta: api.ObjectMeta{Namespace: ns}}
	roleBindingsList, err := p.roleBindings.Index("namespace", nsKey)
	if err != nil {
		return nil, nil, err
	}
	var roleBindings []*rbac.RoleBinding
	if len(roleBindingsList) == 0 {
		// Not in cache, try retrieving directly.
		roleBindingsDirect, err := p.client.Rbac().RoleBindings(ns).List(api.ListOptions{})
		if err != nil {
			return nil, nil, err
		}
		for i := range roleBindingsDirect.Items {
			roleBindings = append(roleBindings, &roleBindingsDirect.Items[i])
		}
	} else {
		for _, rb := range roleBindingsList {
			roleBindings = append(roleBindings, rb.(*rbac.RoleBinding))
		}
	}

	for _, rb := range roleBindings {
		ok, err := validation.RoleBindingMatches(rb, user)
		if err != nil {
			return nil, nil, err
		}
		if ok {
			switch rb.RoleRef.Kind {
			case "ClusterRole":
				clusterRoleNames.Insert(rb.RoleRef.Name)
			case "Role":
				roleNames.Insert(rb.RoleRef.Name)
			}
		}
	}

	return roleNames, clusterRoleNames, nil
}

func checkAttributes(
	obj runtime.Object,
	roles sets.String,
	clusterRoles sets.String,
	protectedLabels map[string][]*rbac.ProtectedAttribute,
	clusterProtectedLabels map[string][]*rbac.ClusterProtectedAttribute,
	protectedAnnotations map[string][]*rbac.ProtectedAttribute,
	clusterProtectedAnnotations map[string][]*rbac.ClusterProtectedAttribute,
) error {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return errors.NewInternalError(err)
	}

	objLabels := accessor.GetLabels()
	objAnnotations := accessor.GetAnnotations()

	passedLabels := sets.NewString()
	passedAnnotations := sets.NewString()
	forbiddenLabels := sets.NewString()
	forbiddenAnnotations := sets.NewString()

	for k, v := range objLabels {
		if labels, ok := clusterProtectedLabels[k]; ok {
			if hasClusterRoleMatch(labels, v, clusterRoles) {
				passedLabels.Insert(k)
			} else {
				forbiddenLabels.Insert(k)
			}
		}
		if labels, ok := protectedLabels[k]; ok {
			if hasRoleMatch(labels, v, roles, clusterRoles) {
				passedLabels.Insert(k)
			} else {
				forbiddenLabels.Insert(k)
			}
		}
	}

	for k, v := range objAnnotations {
		if annotations, ok := clusterProtectedAnnotations[k]; ok {
			if hasClusterRoleMatch(annotations, v, clusterRoles) {
				passedAnnotations.Insert(k)
			} else {
				forbiddenAnnotations.Insert(k)
			}
		}
		if annotations, ok := protectedAnnotations[k]; ok {
			if hasRoleMatch(annotations, v, roles, clusterRoles) {
				passedAnnotations.Insert(k)
			} else {
				forbiddenAnnotations.Insert(k)
			}
		}
	}

	forbiddenLabels = forbiddenLabels.Difference(passedLabels)
	forbiddenAnnotations = forbiddenAnnotations.Difference(passedAnnotations)

	if forbiddenLabels.Len() > 0 || forbiddenAnnotations.Len() > 0 {
		msg := "not enough permissions to use protected attributes: "
		add := []string{}
		if forbiddenLabels.Len() > 0 {
			add = append(add, "labels: "+strings.Join(forbiddenLabels.List(), ","))
		}
		if forbiddenAnnotations.Len() > 0 {
			add = append(add, "annotations: "+strings.Join(forbiddenAnnotations.List(), ","))
		}
		msg += strings.Join(add, "; ")
		return fmt.Errorf(msg)
	}

	return nil
}

func hasRoleMatch(paList []*rbac.ProtectedAttribute, value string, roles sets.String, clusterRoles sets.String) bool {
	for _, pa := range paList {

		hasRole := false

		switch pa.RoleRef.Kind {
		case "ClusterRole":
			hasRole = clusterRoles.Has(pa.RoleRef.Name)
		case "Role":
			hasRole = roles.Has(pa.RoleRef.Name)
		default:
			// This kind should have failed on ProtectedAttribute
			// validation in the first place.
			continue
		}

		if !hasRole {
			continue
		}

		if len(pa.ProtectedValues) == 0 {
			// Role can set any value.
			return true
		}

		for _, protectedValue := range pa.ProtectedValues {
			// Role can set this particular value.
			if protectedValue == value {
				return true
			}
		}
	}

	return false
}

func hasClusterRoleMatch(paList []*rbac.ClusterProtectedAttribute, value string, clusterRoles sets.String) bool {
	for _, pa := range paList {

		hasRole := false

		switch pa.RoleRef.Kind {
		case "ClusterRole":
			hasRole = clusterRoles.Has(pa.RoleRef.Name)
		default:
			// ClusterProtectedAttribute can only reference a ClusterRole,
			// that should have been caught earlier on validation.
			continue
		}

		if !hasRole {
			continue
		}

		if len(pa.ProtectedValues) == 0 {
			// Role can set any value.
			return true
		}

		for _, protectedValue := range pa.ProtectedValues {
			// Role can set this particular value.
			if protectedValue == value {
				return true
			}
		}
	}

	return false
}
