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
	"context"
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
	"k8s.io/client-go/dynamic"
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
	ApplySetToolingAnnotation = "applyset.kubernetes.io/tooling"

	// ApplySetAdditionalNamespacesAnnotation annotation extends the scope of the ApplySet beyond the parent
	// object's own namespace (if any) to include the listed namespaces. The value is a comma-separated
	// list of the names of namespaces other than the parent's namespace in which objects are found
	// Example value: "kube-system,ns1,ns2".
	ApplySetAdditionalNamespacesAnnotation = "applyset.kubernetes.io/additional-namespaces"

	// Deprecated: ApplySetGRsAnnotation is a list of group-resources used to optimize listing of ApplySet member objects.
	// It is optional in the ApplySet specification, as tools can perform discovery or use a different optimization.
	// However, it is currently required in kubectl.
	// When present, the value of this annotation must be a comma separated list of the group-resources,
	// in the fully-qualified name format, i.e. <resourcename>.<group>.
	// Example value: "certificates.cert-manager.io,configmaps,deployments.apps,secrets,services"
	// Deprecated and replaced by ApplySetGKsAnnotation, support for this can be removed in applyset beta or GA.
	DeprecatedApplySetGRsAnnotation = "applyset.kubernetes.io/contains-group-resources"

	// ApplySetGKsAnnotation is a list of group-kinds used to optimize listing of ApplySet member objects.
	// It is optional in the ApplySet specification, as tools can perform discovery or use a different optimization.
	// However, it is currently required in kubectl.
	// When present, the value of this annotation must be a comma separated list of the group-kinds,
	// in the fully-qualified name format, i.e. <kind>.<group>.
	// Example value: "Certificate.cert-manager.io,ConfigMap,deployments.apps,Secret,Service"
	ApplySetGKsAnnotation = "applyset.kubernetes.io/contains-group-kinds"

	// ApplySetParentIDLabel is the key of the label that makes object an ApplySet parent object.
	// Its value MUST use the format specified in V1ApplySetIdFormat below
	ApplySetParentIDLabel = "applyset.kubernetes.io/id"

	// V1ApplySetIdFormat is the format required for the value of ApplySetParentIDLabel (and ApplysetPartOfLabel).
	// The %s segment is the unique ID of the object itself, which MUST be the base64 encoding
	// (using the URL safe encoding of RFC4648) of the hash of the GKNN of the object it is on, in the form:
	// base64(sha256(<name>.<namespace>.<kind>.<group>)).
	V1ApplySetIdFormat = "applyset-%s-v1"

	// ApplysetPartOfLabel is the key of the label which indicates that the object is a member of an ApplySet.
	// The value of the label MUST match the value of ApplySetParentIDLabel on the parent object.
	ApplysetPartOfLabel = "applyset.kubernetes.io/part-of"

	// ApplysetParentCRDLabel is the key of the label that can be set on a CRD to identify
	// the custom resource type it defines (not the CRD itself) as an allowed parent for an ApplySet.
	ApplysetParentCRDLabel = "applyset.kubernetes.io/is-parent-type"
)

var defaultApplySetParentGVR = schema.GroupVersionResource{Version: "v1", Resource: "secrets"}

// ApplySet tracks the information about an applyset apply/prune
type ApplySet struct {
	// parentRef is a reference to the parent object that is used to track the applyset.
	parentRef *ApplySetParentRef

	// toolingID is the value to be used and validated in the applyset.kubernetes.io/tooling annotation.
	toolingID ApplySetTooling

	// currentResources is the set of resources that are part of the sever-side set as of when the current operation started.
	currentResources map[schema.GroupKind]*kindInfo

	// currentNamespaces is the set of namespaces that contain objects in this applyset as of when the current operation started.
	currentNamespaces sets.Set[string]

	// updatedResources is the set of resources that will be part of the set as of when the current operation completes.
	updatedResources map[schema.GroupKind]*kindInfo

	// updatedNamespaces is the set of namespaces that will contain objects in this applyset as of when the current operation completes.
	updatedNamespaces sets.Set[string]

	restMapper meta.RESTMapper

	// client is a client specific to the ApplySet parent object's type
	client resource.RESTClient
}

var builtinApplySetParentGVRs = sets.New[schema.GroupVersionResource](
	defaultApplySetParentGVR,
	schema.GroupVersionResource{Version: "v1", Resource: "configmaps"},
)

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
	Name    string
	Version string
}

func (t ApplySetTooling) String() string {
	return fmt.Sprintf("%s/%s", t.Name, t.Version)
}

// NewApplySet creates a new ApplySet object tracked by the given parent object.
func NewApplySet(parent *ApplySetParentRef, tooling ApplySetTooling, mapper meta.RESTMapper, client resource.RESTClient) *ApplySet {
	return &ApplySet{
		currentResources:  make(map[schema.GroupKind]*kindInfo),
		currentNamespaces: make(sets.Set[string]),
		updatedResources:  make(map[schema.GroupKind]*kindInfo),
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
func (a ApplySet) Validate(ctx context.Context, client dynamic.Interface) error {
	var errors []error
	if a.parentRef.IsNamespaced() && a.parentRef.Namespace == "" {
		errors = append(errors, fmt.Errorf("namespace is required to use namespace-scoped ApplySet"))
	}
	if !builtinApplySetParentGVRs.Has(a.parentRef.Resource) {
		// Determine which custom resource types are allowed as ApplySet parents.
		// Optimization: Since this makes requests, we only do this if they aren't using a default type.
		permittedCRParents, err := a.getAllowedCustomResourceParents(ctx, client)
		if err != nil {
			errors = append(errors, fmt.Errorf("identifying allowed custom resource parent types: %w", err))
		}
		parentRefResourceIgnoreVersion := a.parentRef.Resource.GroupResource().WithVersion("")
		if !permittedCRParents.Has(parentRefResourceIgnoreVersion) {
			errors = append(errors, fmt.Errorf("resource %q is not permitted as an ApplySet parent", a.parentRef.Resource))
		}
	}
	return utilerrors.NewAggregate(errors)
}

func (a *ApplySet) labelForCustomParentCRDs() *metav1.LabelSelector {
	return &metav1.LabelSelector{
		MatchExpressions: []metav1.LabelSelectorRequirement{{
			Key:      ApplysetParentCRDLabel,
			Operator: metav1.LabelSelectorOpExists,
		}},
	}
}

func (a *ApplySet) getAllowedCustomResourceParents(ctx context.Context, client dynamic.Interface) (sets.Set[schema.GroupVersionResource], error) {
	opts := metav1.ListOptions{
		LabelSelector: metav1.FormatLabelSelector(a.labelForCustomParentCRDs()),
	}
	list, err := client.Resource(schema.GroupVersionResource{
		Group:    "apiextensions.k8s.io",
		Version:  "v1",
		Resource: "customresourcedefinitions",
	}).List(ctx, opts)
	if err != nil {
		return nil, err
	}
	set := sets.New[schema.GroupVersionResource]()
	for i := range list.Items {
		// Custom resources must be named `<names.plural>.<group>`
		// and are served under `/apis/<group>/<version>/.../<plural>`
		gr := schema.ParseGroupResource(list.Items[i].GetName())
		set.Insert(gr.WithVersion(""))
	}
	return set, nil
}

func (a *ApplySet) LabelsForMember() map[string]string {
	return map[string]string{
		ApplysetPartOfLabel: a.ID(),
	}
}

// AddLabels sets our tracking labels on each object; this should be called as part of loading the objects.
func (a *ApplySet) AddLabels(objects ...*resource.Info) error {
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

func (a *ApplySet) fetchParent() error {
	helper := resource.NewHelper(a.client, a.parentRef.RESTMapping)
	obj, err := helper.Get(a.parentRef.Namespace, a.parentRef.Name)
	if errors.IsNotFound(err) {
		if !builtinApplySetParentGVRs.Has(a.parentRef.Resource) {
			return fmt.Errorf("custom resource ApplySet parents cannot be created automatically")
		}
		return nil
	} else if err != nil {
		return fmt.Errorf("failed to fetch ApplySet parent object %q: %w", a.parentRef, err)
	} else if obj == nil {
		return fmt.Errorf("failed to fetch ApplySet parent object %q", a.parentRef)
	}

	labels, annotations, err := getLabelsAndAnnotations(obj)
	if err != nil {
		return fmt.Errorf("getting metadata from parent object %q: %w", a.parentRef, err)
	}

	toolAnnotation, hasToolAnno := annotations[ApplySetToolingAnnotation]
	if !hasToolAnno {
		return fmt.Errorf("ApplySet parent object %q already exists and is missing required annotation %q", a.parentRef, ApplySetToolingAnnotation)
	}
	if managedBy := toolingBaseName(toolAnnotation); managedBy != a.toolingID.Name {
		return fmt.Errorf("ApplySet parent object %q already exists and is managed by tooling %q instead of %q", a.parentRef, managedBy, a.toolingID.Name)
	}

	idLabel, hasIDLabel := labels[ApplySetParentIDLabel]
	if !hasIDLabel {
		return fmt.Errorf("ApplySet parent object %q exists and does not have required label %s", a.parentRef, ApplySetParentIDLabel)
	}
	if idLabel != a.ID() {
		return fmt.Errorf("ApplySet parent object %q exists and has incorrect value for label %q (got: %s, want: %s)", a.parentRef, ApplySetParentIDLabel, idLabel, a.ID())
	}

	if a.currentResources, err = parseKindAnnotation(annotations, a.restMapper); err != nil {
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
func (a *ApplySet) AllPrunableResources() []*kindInfo {
	var ret []*kindInfo
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

// kindInfo holds type information about a particular resource type.
type kindInfo struct {
	restMapping *meta.RESTMapping
}

func parseKindAnnotation(annotations map[string]string, mapper meta.RESTMapper) (map[schema.GroupKind]*kindInfo, error) {
	annotation, ok := annotations[ApplySetGKsAnnotation]
	if !ok {
		if annotations[DeprecatedApplySetGRsAnnotation] != "" {
			return parseDeprecatedResourceAnnotation(annotations[DeprecatedApplySetGRsAnnotation], mapper)
		}

		// The spec does not require this annotation. However, 'missing' means 'perform discovery'.
		// We return an error because we do not currently support dynamic discovery in kubectl apply.
		return nil, fmt.Errorf("kubectl requires the %q annotation to be set on all ApplySet parent objects", ApplySetGKsAnnotation)
	}
	mappings := make(map[schema.GroupKind]*kindInfo)
	// Annotation present but empty means that this is currently an empty set.
	if annotation == "" {
		return mappings, nil
	}
	for _, gkString := range strings.Split(annotation, ",") {
		gk := schema.ParseGroupKind(gkString)
		restMapping, err := mapper.RESTMapping(gk)
		if err != nil {
			return nil, fmt.Errorf("could not find mapping for kind in %q annotation: %w", ApplySetGKsAnnotation, err)
		}
		mappings[gk] = &kindInfo{
			restMapping: restMapping,
		}
	}

	return mappings, nil
}

func parseDeprecatedResourceAnnotation(annotation string, mapper meta.RESTMapper) (map[schema.GroupKind]*kindInfo, error) {
	mappings := make(map[schema.GroupKind]*kindInfo)
	// Annotation present but empty means that this is currently an empty set.
	if annotation == "" {
		return mappings, nil
	}
	for _, grString := range strings.Split(annotation, ",") {
		gr := schema.ParseGroupResource(grString)
		gvk, err := mapper.KindFor(gr.WithVersion(""))
		if err != nil {
			return nil, fmt.Errorf("invalid group resource in %q annotation: %w", DeprecatedApplySetGRsAnnotation, err)
		}
		restMapping, err := mapper.RESTMapping(gvk.GroupKind())
		if err != nil {
			return nil, fmt.Errorf("could not find kind for resource in %q annotation: %w", DeprecatedApplySetGRsAnnotation, err)
		}
		mappings[gvk.GroupKind()] = &kindInfo{
			restMapping: restMapping,
		}
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

// addResource registers the given resource and namespace as being part of the updated set of
// resources being applied by the current operation.
func (a *ApplySet) addResource(restMapping *meta.RESTMapping, namespace string) {
	gk := restMapping.GroupVersionKind.GroupKind()
	if _, found := a.updatedResources[gk]; !found {
		a.updatedResources[gk] = &kindInfo{
			restMapping: restMapping,
		}
	}
	if restMapping.Scope == meta.RESTScopeNamespace && namespace != "" {
		a.updatedNamespaces.Insert(namespace)
	}
}

type ApplySetUpdateMode string

var updateToLatestSet ApplySetUpdateMode = "latest"
var updateToSuperset ApplySetUpdateMode = "superset"

func (a *ApplySet) updateParent(mode ApplySetUpdateMode, dryRun cmdutil.DryRunStrategy, validation string) error {
	data, err := json.Marshal(a.buildParentPatch(mode))
	if err != nil {
		return fmt.Errorf("failed to encode patch for ApplySet parent: %w", err)
	}
	// Note that because we are using SSA, we will remove any annotations we don't specify,
	// which is how we remove the deprecated contains-group-resources annotation.
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
	var newGKsAnnotation, newNsAnnotation string
	switch mode {
	case updateToSuperset:
		// If the apply succeeded but pruning failed, the set of group resources that
		// the ApplySet should track is the superset of the previous and current resources.
		// This ensures that the resources that failed to be pruned are not orphaned from the set.
		grSuperset := sets.KeySet(a.currentResources).Union(sets.KeySet(a.updatedResources))
		newGKsAnnotation = generateKindsAnnotation(grSuperset)
		newNsAnnotation = generateNamespacesAnnotation(a.currentNamespaces.Union(a.updatedNamespaces), a.parentRef.Namespace)
	case updateToLatestSet:
		newGKsAnnotation = generateKindsAnnotation(sets.KeySet(a.updatedResources))
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
				ApplySetGKsAnnotation:                  newGKsAnnotation,
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

func generateKindsAnnotation(resources sets.Set[schema.GroupKind]) string {
	var gks []string
	for gk := range resources {
		gks = append(gks, gk.String())
	}
	sort.Strings(gks)
	return strings.Join(gks, ",")
}

func (a ApplySet) FieldManager() string {
	return fmt.Sprintf("%s-applyset", a.toolingID.Name)
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

// Prune deletes any objects from the apiserver that are no longer in the applyset.
func (a *ApplySet) Prune(ctx context.Context, o *ApplyOptions) error {
	printer, err := o.ToPrinter("pruned")
	if err != nil {
		return err
	}
	opt := &ApplySetDeleteOptions{
		CascadingStrategy: o.DeleteOptions.CascadingStrategy,
		DryRunStrategy:    o.DryRunStrategy,
		GracePeriod:       o.DeleteOptions.GracePeriod,

		Printer: printer,

		IOStreams: o.IOStreams,
	}

	if err := a.pruneAll(ctx, o.DynamicClient, o.VisitedUids, opt); err != nil {
		return err
	}

	if err := a.updateParent(updateToLatestSet, o.DryRunStrategy, o.ValidationDirective); err != nil {
		return fmt.Errorf("apply and prune succeeded, but ApplySet update failed: %w", err)
	}

	return nil
}

// BeforeApply should be called before applying the objects.
// It pre-updates the parent object so that it covers the resources that will be applied.
// In this way, even if we are interrupted, we will not leak objects.
func (a *ApplySet) BeforeApply(objects []*resource.Info, dryRunStrategy cmdutil.DryRunStrategy, validationDirective string) error {
	if err := a.fetchParent(); err != nil {
		return err
	}
	// Update the live parent object to the superset of the current and previous resources.
	// Doing this before the actual apply and prune operations improves behavior by ensuring
	// the live object contains the superset on failure. This may cause the next pruning
	// operation to make a larger number of GET requests than strictly necessary, but it prevents
	// object leakage from the set. The superset will automatically be reduced to the correct
	// set by the next successful operation.
	for _, info := range objects {
		a.addResource(info.ResourceMapping(), info.Namespace)
	}
	if err := a.updateParent(updateToSuperset, dryRunStrategy, validationDirective); err != nil {
		return err
	}
	return nil
}
