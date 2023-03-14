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
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/klog/v2"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
)

// Label and annotation keys from the ApplySet specification.
// https://git.k8s.io/enhancements/keps/sig-cli/3659-kubectl-apply-prune#design-details-applyset-specification
const (
	// ApplySetToolingAnnotation is the key of the label that indicates which tool is used to manage this ApplySet.
	// Tooling should refuse to mutate ApplySets belonging to other tools.
	// The value must be in the format <toolname>/<semver>.
	// Example value: "kubectl/v1.27" or "helm/v3" or "kpt/v1.0.0"
	ApplySetToolingAnnotation = "applyset.k8s.io/tooling"

	// ApplySetAdditionalNamespacesAnnotation annotation extends the scope of the ApplySet beyond the parent
	// object's own namespace (if any) to include the listed namespaces. The value is a comma-separated
	// list of the names of namespaces other than the parent's namespace in which objects are found
	// Example value: "kube-system,ns1,ns2".
	ApplySetAdditionalNamespacesAnnotation = "applyset.k8s.io/additional-namespaces"

	// ApplySetGRsAnnotation is a list of group-resources used to optimize listing of ApplySet member objects.
	// It is optional in the ApplySet specification, as tools can perform discovery or use a different optimization.
	// However, it is currently required in kubectl.
	// When present, the value of this annotation must be a comma separated list of the group-kinds,
	// in the fully-qualified name format, i.e. <resourcename>.<group>.
	// Example value: "certificates.cert-manager.io,configmaps,deployments.apps,secrets,services"
	ApplySetGRsAnnotation = "applyset.k8s.io/contains-group-resources"

	// ApplySetParentIDLabel is the key of the label that makes object an ApplySet parent object.
	// Its value MUST use the format specified in V1ApplySetIdFormat below
	ApplySetParentIDLabel = "applyset.k8s.io/id"

	// V1ApplySetIdFormat is the format required for the value of ApplySetParentIDLabel (and ApplysetPartOfLabel).
	// The %s segment is the unique ID of the object itself, which MUST be the base64 encoding
	// (using the URL safe encoding of RFC4648) of the hash of the GKNN of the object it is on, in the form:
	// base64(sha256(<name>.<namespace>.<kind>.<group>)).
	V1ApplySetIdFormat = "applyset-%s-v1"

	// ApplysetPartOfLabel is the key of the label which indicates that the object is a member of an ApplySet.
	// The value of the label MUST match the value of ApplySetParentIDLabel on the parent object.
	ApplysetPartOfLabel = "applyset.k8s.io/part-of"
)

var defaultApplySetParentGVR = schema.GroupVersionResource{Version: "v1", Resource: "secrets"}

// ApplySet tracks the information about an applyset apply/prune
type ApplySet struct {
	// parentRef is a reference to the parent object that is used to track the applyset.
	parentRef *ApplySetParentRef

	// toolingID is the value to be used and validated in the applyset.k8s.io/tooling annotation.
	toolingID ApplySetTooling

	// currentResources is the set of resources that are part of the sever-side set as of when the current operation started.
	currentResources map[schema.GroupVersionResource]*meta.RESTMapping

	// currentNamespaces is the set of namespaces that contain objects in this applyset as of when the current operation started.
	currentNamespaces sets.Set[string]

	// updatedResources is the set of resources that will be part of the set as of when the current operation completes.
	updatedResources map[schema.GroupVersionResource]*meta.RESTMapping

	// updatedNamespaces is the set of namespaces that will contain objects in this applyset as of when the current operation completes.
	updatedNamespaces sets.Set[string]

	restMapper meta.RESTMapper

	// client is a client specific to the ApplySet parent object's type
	client resource.RESTClient
}

var builtinApplySetParentGVRs = map[schema.GroupVersionResource]bool{
	defaultApplySetParentGVR:                true,
	{Version: "v1", Resource: "configmaps"}: true,
}

// ApplySetParentRef stores object and type meta for the parent object that is used to track the applyset.
type ApplySetParentRef struct {
	Name      string
	Namespace string
	*meta.RESTMapping
}

func (p ApplySetParentRef) IsNamespaced() bool {
	return p.Scope.Name() == meta.RESTScopeNameNamespace
}

// String returns the string representation of the parent object using the same format
// that we expect to receive in the --applyset flag on the CLI.
func (p ApplySetParentRef) String() string {
	return fmt.Sprintf("%s.%s/%s", p.Resource.Resource, p.Resource.Group, p.Name)
}

type ApplySetTooling struct {
	name    string
	version string
}

func (t ApplySetTooling) String() string {
	return fmt.Sprintf("%s/%s", t.name, t.version)
}

// NewApplySet creates a new ApplySet object tracked by the given parent object.
func NewApplySet(parent *ApplySetParentRef, tooling ApplySetTooling, mapper meta.RESTMapper, client resource.RESTClient) *ApplySet {
	return &ApplySet{
		currentResources:  make(map[schema.GroupVersionResource]*meta.RESTMapping),
		currentNamespaces: make(sets.Set[string]),
		updatedResources:  make(map[schema.GroupVersionResource]*meta.RESTMapping),
		updatedNamespaces: make(sets.Set[string]),
		parentRef:         parent,
		toolingID:         tooling,
		restMapper:        mapper,
		client:            client,
	}
}

const applySetIDPartDelimiter = "."

// ID is the label value that we are using to identify this applyset.
// Format: base64(sha256(<name>.<namespace>.<kind>.<group>)), using the URL safe encoding of RFC4648.

func (a ApplySet) ID() string {
	unencoded := strings.Join([]string{a.parentRef.Name, a.parentRef.Namespace, a.parentRef.GroupVersionKind.Kind, a.parentRef.GroupVersionKind.Group}, applySetIDPartDelimiter)
	hashed := sha256.Sum256([]byte(unencoded))
	b64 := base64.RawURLEncoding.EncodeToString(hashed[:])
	// Label values must start and end with alphanumeric values, so add a known-safe prefix and suffix.
	return fmt.Sprintf(V1ApplySetIdFormat, b64)
}

// Validate imposes restrictions on the parent object that is used to track the applyset.
func (a ApplySet) Validate() error {
	var errors []error
	// TODO: permit CRDs that have the annotation required by the ApplySet specification
	if !builtinApplySetParentGVRs[a.parentRef.Resource] {
		errors = append(errors, fmt.Errorf("resource %q is not permitted as an ApplySet parent", a.parentRef.Resource))
	}
	if a.parentRef.IsNamespaced() && a.parentRef.Namespace == "" {
		errors = append(errors, fmt.Errorf("namespace is required to use namespace-scoped ApplySet"))
	}
	return utilerrors.NewAggregate(errors)
}

func (a *ApplySet) LabelsForMember() map[string]string {
	return map[string]string{
		ApplysetPartOfLabel: a.ID(),
	}
}

// addLabels sets our tracking labels on each object; this should be called as part of loading the objects.
func (a *ApplySet) addLabels(objects []*resource.Info) error {
	applysetLabels := a.LabelsForMember()
	for _, obj := range objects {
		accessor, err := meta.Accessor(obj.Object)
		if err != nil {
			return fmt.Errorf("getting accessor: %w", err)
		}
		labels := accessor.GetLabels()
		if labels == nil {
			labels = make(map[string]string)
		}
		for k, v := range applysetLabels {
			if _, found := labels[k]; found {
				return fmt.Errorf("ApplySet label %q already set in input data", k)
			}
			labels[k] = v
		}
		accessor.SetLabels(labels)
	}

	return nil
}

func (a *ApplySet) FetchParent() error {
	helper := resource.NewHelper(a.client, a.parentRef.RESTMapping)
	obj, err := helper.Get(a.parentRef.Namespace, a.parentRef.Name)
	if errors.IsNotFound(err) {
		return nil
	} else if err != nil {
		return fmt.Errorf("failed to fetch ApplySet parent object %q from server: %w", a.parentRef, err)
	} else if obj == nil {
		return fmt.Errorf("failed to fetch ApplySet parent object %q from server", a.parentRef)
	}

	labels, annotations, err := getLabelsAndAnnotations(obj)
	if err != nil {
		return fmt.Errorf("getting metadata from parent object %q: %w", a.parentRef, err)
	}

	toolAnnotation, hasToolAnno := annotations[ApplySetToolingAnnotation]
	if !hasToolAnno {
		return fmt.Errorf("ApplySet parent object %q already exists and is missing required annotation %q", a.parentRef, ApplySetToolingAnnotation)
	}
	if managedBy := toolingBaseName(toolAnnotation); managedBy != a.toolingID.name {
		return fmt.Errorf("ApplySet parent object %q already exists and is managed by tooling %q instead of %q", a.parentRef, managedBy, a.toolingID.name)
	}

	idLabel, hasIDLabel := labels[ApplySetParentIDLabel]
	if !hasIDLabel {
		return fmt.Errorf("ApplySet parent object %q exists and does not have required label %s", a.parentRef, ApplySetParentIDLabel)
	}
	if idLabel != a.ID() {
		return fmt.Errorf("ApplySet parent object %q exists and has incorrect value for label %q (got: %s, want: %s)", a.parentRef, ApplySetParentIDLabel, idLabel, a.ID())
	}

	if a.currentResources, err = parseResourcesAnnotation(annotations, a.restMapper); err != nil {
		// TODO: handle GVRs for now-deleted CRDs
		return fmt.Errorf("parsing ApplySet annotation on %q: %w", a.parentRef, err)
	}
	a.currentNamespaces = parseNamespacesAnnotation(annotations)
	if a.parentRef.IsNamespaced() {
		a.currentNamespaces.Insert(a.parentRef.Namespace)
	}
	return nil
}
func (a *ApplySet) LabelSelectorForMembers() string {
	return metav1.FormatLabelSelector(&metav1.LabelSelector{
		MatchLabels: a.LabelsForMember(),
	})
}

// AllPrunableResources returns the list of all resources that should be considered for pruning.
// This is potentially a superset of the resources types that actually contain resources.
func (a *ApplySet) AllPrunableResources() []*meta.RESTMapping {
	var ret []*meta.RESTMapping
	for _, m := range a.currentResources {
		ret = append(ret, m)
	}
	return ret
}

// AllPrunableNamespaces returns the list of all namespaces that should be considered for pruning.
// This is potentially a superset of the namespaces that actually contain resources.
func (a *ApplySet) AllPrunableNamespaces() []string {
	var ret []string
	for ns := range a.currentNamespaces {
		ret = append(ret, ns)
	}
	return ret
}

func getLabelsAndAnnotations(obj runtime.Object) (map[string]string, map[string]string, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return nil, nil, err
	}
	return accessor.GetLabels(), accessor.GetAnnotations(), nil
}

func toolingBaseName(toolAnnotation string) string {
	parts := strings.Split(toolAnnotation, "/")
	if len(parts) >= 2 {
		return strings.Join(parts[:len(parts)-1], "/")
	}
	return toolAnnotation
}

func parseResourcesAnnotation(annotations map[string]string, mapper meta.RESTMapper) (map[schema.GroupVersionResource]*meta.RESTMapping, error) {
	annotation, ok := annotations[ApplySetGRsAnnotation]
	if !ok {
		// The spec does not require this annotation. However, 'missing' means 'perform discovery' (as opposed to 'present but empty', which means ' this is an empty set').
		// We return an error because we do not currently support dynamic discovery in kubectl apply.
		return nil, fmt.Errorf("kubectl requires the %q annotation to be set on all ApplySet parent objects", ApplySetGRsAnnotation)
	}
	mappings := make(map[schema.GroupVersionResource]*meta.RESTMapping)
	for _, grString := range strings.Split(annotation, ",") {
		gr := schema.ParseGroupResource(grString)
		gvk, err := mapper.KindFor(gr.WithVersion(""))
		if err != nil {
			return nil, fmt.Errorf("invalid group resource in %q annotation: %w", ApplySetGRsAnnotation, err)
		}
		mapping, err := mapper.RESTMapping(gvk.GroupKind())
		if err != nil {
			return nil, fmt.Errorf("could not find kind for resource in %q annotation: %w", ApplySetGRsAnnotation, err)
		}
		mappings[mapping.Resource] = mapping
	}
	return mappings, nil
}

func parseNamespacesAnnotation(annotations map[string]string) sets.Set[string] {
	annotation, ok := annotations[ApplySetAdditionalNamespacesAnnotation]
	if !ok { // this annotation is completely optional
		return sets.Set[string]{}
	}
	// Don't include an empty namespace
	if annotation == "" {
		return sets.Set[string]{}
	}
	return sets.New(strings.Split(annotation, ",")...)
}

// AddResource registers the given resource and namespace as being part of the updated set of
// resources being applied by the current operation.
func (a *ApplySet) AddResource(resource *meta.RESTMapping, namespace string) {
	a.updatedResources[resource.Resource] = resource
	if resource.Scope == meta.RESTScopeNamespace && namespace != "" {
		a.updatedNamespaces.Insert(namespace)
	}
}

type ApplySetUpdateMode string

var UpdateToLatestSet ApplySetUpdateMode = "latest"
var UpdateToSuperset ApplySetUpdateMode = "superset"

func (a *ApplySet) UpdateParent(mode ApplySetUpdateMode, dryRun cmdutil.DryRunStrategy, validation string) error {
	data, err := json.Marshal(a.buildParentPatch(mode))
	if err != nil {
		return fmt.Errorf("failed to encode patch for ApplySet parent: %w", err)
	}
	err = serverSideApplyRequest(a, data, dryRun, validation, false)
	if err != nil && errors.IsConflict(err) {
		// Try again with conflicts forced
		klog.Warningf("WARNING: failed to update ApplySet: %s\nApplySet field manager %s should own these fields. Retrying with conflicts forced.", err.Error(), a.FieldManager())
		err = serverSideApplyRequest(a, data, dryRun, validation, true)
	}
	if err != nil {
		return fmt.Errorf("failed to update ApplySet: %w", err)
	}
	return nil
}

func serverSideApplyRequest(a *ApplySet, data []byte, dryRun cmdutil.DryRunStrategy, validation string, forceConficts bool) error {
	if dryRun == cmdutil.DryRunClient {
		return nil
	}
	helper := resource.NewHelper(a.client, a.parentRef.RESTMapping).
		DryRun(dryRun == cmdutil.DryRunServer).
		WithFieldManager(a.FieldManager()).
		WithFieldValidation(validation)

	options := metav1.PatchOptions{
		Force: &forceConficts,
	}
	_, err := helper.Patch(
		a.parentRef.Namespace,
		a.parentRef.Name,
		types.ApplyPatchType,
		data,
		&options,
	)
	return err
}

func (a *ApplySet) buildParentPatch(mode ApplySetUpdateMode) *metav1.PartialObjectMetadata {
	var newGRsAnnotation, newNsAnnotation string
	switch mode {
	case UpdateToSuperset:
		// If the apply succeeded but pruning failed, the set of group resources that
		// the ApplySet should track is the superset of the previous and current resources.
		// This ensures that the resources that failed to be pruned are not orphaned from the set.
		grSuperset := sets.KeySet(a.currentResources).Union(sets.KeySet(a.updatedResources))
		newGRsAnnotation = generateResourcesAnnotation(grSuperset)
		newNsAnnotation = generateNamespacesAnnotation(a.currentNamespaces.Union(a.updatedNamespaces), a.parentRef.Namespace)
	case UpdateToLatestSet:
		newGRsAnnotation = generateResourcesAnnotation(sets.KeySet(a.updatedResources))
		newNsAnnotation = generateNamespacesAnnotation(a.updatedNamespaces, a.parentRef.Namespace)
	}

	return &metav1.PartialObjectMetadata{
		TypeMeta: metav1.TypeMeta{
			Kind:       a.parentRef.GroupVersionKind.Kind,
			APIVersion: a.parentRef.GroupVersionKind.GroupVersion().String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      a.parentRef.Name,
			Namespace: a.parentRef.Namespace,
			Annotations: map[string]string{
				ApplySetToolingAnnotation:              a.toolingID.String(),
				ApplySetGRsAnnotation:                  newGRsAnnotation,
				ApplySetAdditionalNamespacesAnnotation: newNsAnnotation,
			},
			Labels: map[string]string{
				ApplySetParentIDLabel: a.ID(),
			},
		},
	}
}

func generateNamespacesAnnotation(namespaces sets.Set[string], skip string) string {
	nsList := namespaces.Clone().Delete(skip).UnsortedList()
	sort.Strings(nsList)
	return strings.Join(nsList, ",")
}

func generateResourcesAnnotation(resources sets.Set[schema.GroupVersionResource]) string {
	var grs []string
	for gvr := range resources {
		grs = append(grs, gvr.GroupResource().String())
	}
	sort.Strings(grs)
	return strings.Join(grs, ",")
}

func (a ApplySet) FieldManager() string {
	return fmt.Sprintf("%s-applyset", a.toolingID.name)
}

// ParseApplySetParentRef creates a new ApplySetParentRef from a parent reference in the format [RESOURCE][.GROUP]/NAME
func ParseApplySetParentRef(parentRefStr string, mapper meta.RESTMapper) (*ApplySetParentRef, error) {
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
