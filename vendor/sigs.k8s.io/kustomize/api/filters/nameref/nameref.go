package nameref

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/pkg/errors"
	"sigs.k8s.io/kustomize/api/filters/fieldspec"
	"sigs.k8s.io/kustomize/api/resid"
	"sigs.k8s.io/kustomize/api/resmap"
	"sigs.k8s.io/kustomize/api/resource"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/filtersutil"
	"sigs.k8s.io/kustomize/kyaml/kio"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// Filter updates a name references.
type Filter struct {
	// Referrer refers to another resource X by X's name.
	// E.g. A Deployment can refer to a ConfigMap.
	// The Deployment is the Referrer,
	// the ConfigMap is the ReferralTarget.
	// This filter seeks to repair the reference in Deployment, given
	// that the ConfigMap's name may have changed.
	Referrer *resource.Resource

	// NameFieldToUpdate is the field in the Referrer
	// that holds the name requiring an update.
	// This is the field to write.
	NameFieldToUpdate types.FieldSpec

	// ReferralTarget is the source of the new value for
	// the name, always in the 'metadata/name' field.
	// This is the field to read.
	ReferralTarget resid.Gvk

	// Set of resources to scan to find the ReferralTarget.
	ReferralCandidates resmap.ResMap
}

// At time of writing, in practice this is called with a slice with only
// one entry, the node also referred to be the resource in the Referrer field.
func (f Filter) Filter(nodes []*yaml.RNode) ([]*yaml.RNode, error) {
	return kio.FilterAll(yaml.FilterFunc(f.run)).Filter(nodes)
}

// The node passed in here is the same node as held in Referrer;
// that's how the referrer's name field is updated.
// Currently, however, this filter still needs the extra methods on Referrer
// to consult things like the resource Id, its namespace, etc.
// TODO(3455): No filter should use the Resource api; all information
// about names should come from annotations, with helper methods
// on the RNode object.  Resource should get stupider, RNode smarter.
func (f Filter) run(node *yaml.RNode) (*yaml.RNode, error) {
	if err := f.confirmNodeMatchesReferrer(node); err != nil {
		// sanity check.
		return nil, err
	}
	if err := node.PipeE(fieldspec.Filter{
		FieldSpec: f.NameFieldToUpdate,
		SetValue:  f.set,
	}); err != nil {
		return nil, errors.Wrapf(
			err, "updating name reference in '%s' field of '%s'",
			f.NameFieldToUpdate.Path, f.Referrer.CurId().String())
	}
	return node, nil
}

// This function is called on the node found at FieldSpec.Path.
// It's some node in the Referrer.
func (f Filter) set(node *yaml.RNode) error {
	if yaml.IsMissingOrNull(node) {
		return nil
	}
	switch node.YNode().Kind {
	case yaml.ScalarNode:
		return f.setScalar(node)
	case yaml.MappingNode:
		return f.setMapping(node)
	case yaml.SequenceNode:
		return applyFilterToSeq(seqFilter{
			setScalarFn:  f.setScalar,
			setMappingFn: f.setMapping,
		}, node)
	default:
		return fmt.Errorf("node must be a scalar, sequence or map")
	}
}

// This method used when NameFieldToUpdate doesn't lead to
// one scalar field (typically called 'name'), but rather
// leads to a map field (called anything). In this case we
// must complete the field path, looking for both  a 'name'
// and a 'namespace' field to help select the proper
// ReferralTarget to read the name and namespace from.
func (f Filter) setMapping(node *yaml.RNode) error {
	if node.YNode().Kind != yaml.MappingNode {
		return fmt.Errorf("expect a mapping node")
	}
	nameNode, err := node.Pipe(yaml.FieldMatcher{Name: "name"})
	if err != nil {
		return errors.Wrap(err, "trying to match 'name' field")
	}
	if nameNode == nil {
		// This is a _configuration_ error; the field path
		// specified in NameFieldToUpdate.Path doesn't resolve
		// to a map with a 'name' field, so we have no idea what
		// field to update with a new name.
		return fmt.Errorf("path config error; no 'name' field in node")
	}
	candidates, err := f.filterMapCandidatesByNamespace(node)
	if err != nil {
		return err
	}
	oldName := nameNode.YNode().Value
	referral, err := f.selectReferral(oldName, candidates)
	if err != nil || referral == nil {
		// Nil referral means nothing to do.
		return err
	}
	f.recordTheReferral(referral)
	if referral.GetName() == oldName && referral.GetNamespace() == "" {
		// The name has not changed, nothing to do.
		return nil
	}
	if err = node.PipeE(yaml.FieldSetter{
		Name:        "name",
		StringValue: referral.GetName(),
	}); err != nil {
		return err
	}
	if referral.GetNamespace() == "" {
		// Don't write an empty string into the namespace field, as
		// it should not replace the value "default".  The empty
		// string is handled as a wild card here, not as an implicit
		// specification of the "default" k8s namespace.
		return nil
	}
	return node.PipeE(yaml.FieldSetter{
		Name:        "namespace",
		StringValue: referral.GetNamespace(),
	})
}

func (f Filter) filterMapCandidatesByNamespace(
	node *yaml.RNode) ([]*resource.Resource, error) {
	namespaceNode, err := node.Pipe(yaml.FieldMatcher{Name: "namespace"})
	if err != nil {
		return nil, errors.Wrap(err, "trying to match 'namespace' field")
	}
	if namespaceNode == nil {
		return f.ReferralCandidates.Resources(), nil
	}
	namespace := namespaceNode.YNode().Value
	nsMap := f.ReferralCandidates.GroupedByOriginalNamespace()
	if candidates, ok := nsMap[namespace]; ok {
		return candidates, nil
	}
	nsMap = f.ReferralCandidates.GroupedByCurrentNamespace()
	// This could be nil, or an empty list.
	return nsMap[namespace], nil
}

func (f Filter) setScalar(node *yaml.RNode) error {
	referral, err := f.selectReferral(
		node.YNode().Value, f.ReferralCandidates.Resources())
	if err != nil || referral == nil {
		// Nil referral means nothing to do.
		return err
	}
	f.recordTheReferral(referral)
	if referral.GetName() == node.YNode().Value {
		// The name has not changed, nothing to do.
		return nil
	}
	return node.PipeE(yaml.FieldSetter{StringValue: referral.GetName()})
}

// In the resource, make a note that it is referred to by the Referrer.
func (f Filter) recordTheReferral(referral *resource.Resource) {
	referral.AppendRefBy(f.Referrer.CurId())
}

// getRoleRefGvk returns a Gvk in the roleRef field. Return error
// if the roleRef, roleRef/apiGroup or roleRef/kind is missing.
func getRoleRefGvk(res json.Marshaler) (*resid.Gvk, error) {
	n, err := filtersutil.GetRNode(res)
	if err != nil {
		return nil, err
	}
	roleRef, err := n.Pipe(yaml.Lookup("roleRef"))
	if err != nil {
		return nil, err
	}
	if roleRef.IsNil() {
		return nil, fmt.Errorf("roleRef cannot be found in %s", n.MustString())
	}
	apiGroup, err := roleRef.Pipe(yaml.Lookup("apiGroup"))
	if err != nil {
		return nil, err
	}
	if apiGroup.IsNil() {
		return nil, fmt.Errorf(
			"apiGroup cannot be found in roleRef %s", roleRef.MustString())
	}
	kind, err := roleRef.Pipe(yaml.Lookup("kind"))
	if err != nil {
		return nil, err
	}
	if kind.IsNil() {
		return nil, fmt.Errorf(
			"kind cannot be found in roleRef %s", roleRef.MustString())
	}
	return &resid.Gvk{
		Group: apiGroup.YNode().Value,
		Kind:  kind.YNode().Value,
	}, nil
}

// sieveFunc returns true if the resource argument satisfies some criteria.
type sieveFunc func(*resource.Resource) bool

// doSieve uses a function to accept or ignore resources from a list.
// If list is nil, returns immediately.
// It's a filter obviously, but that term is overloaded here.
func doSieve(list []*resource.Resource, fn sieveFunc) (s []*resource.Resource) {
	for _, r := range list {
		if fn(r) {
			s = append(s, r)
		}
	}
	return
}

func acceptAll(r *resource.Resource) bool {
	return true
}

func previousNameMatches(name string) sieveFunc {
	return func(r *resource.Resource) bool {
		for _, id := range r.PrevIds() {
			if id.Name == name {
				return true
			}
		}
		return false
	}
}

func previousIdSelectedByGvk(gvk *resid.Gvk) sieveFunc {
	return func(r *resource.Resource) bool {
		for _, id := range r.PrevIds() {
			if id.IsSelected(gvk) {
				return true
			}
		}
		return false
	}
}

// If the we are updating a 'roleRef/name' field, the 'apiGroup' and 'kind'
// fields in the same 'roleRef' map must be considered.
// If either object is cluster-scoped (!IsNamespaceableKind), there
// can be a referral.
// E.g. a RoleBinding (which exists in a namespace) can refer
// to a ClusterRole (cluster-scoped) object.
// https://kubernetes.io/docs/reference/access-authn-authz/rbac/#role-and-clusterrole
// Likewise, a ClusterRole can refer to a Secret (in a namespace).
// Objects in different namespaces generally cannot refer to other
// with some exceptions (e.g. RoleBinding and ServiceAccount are both
// namespaceable, but the former can refer to accounts in other namespaces).
func (f Filter) roleRefFilter() sieveFunc {
	if !strings.HasSuffix(f.NameFieldToUpdate.Path, "roleRef/name") {
		return acceptAll
	}
	roleRefGvk, err := getRoleRefGvk(f.Referrer)
	if err != nil {
		return acceptAll
	}
	return previousIdSelectedByGvk(roleRefGvk)
}

func prefixSuffixEquals(other resource.ResCtx) sieveFunc {
	return func(r *resource.Resource) bool {
		return r.PrefixesSuffixesEquals(other)
	}
}

func (f Filter) sameCurrentNamespaceAsReferrer() sieveFunc {
	referrerCurId := f.Referrer.CurId()
	if !referrerCurId.IsNamespaceableKind() {
		// If the referrer is cluster-scoped, let anything through.
		return acceptAll
	}
	return func(r *resource.Resource) bool {
		if !r.CurId().IsNamespaceableKind() {
			// Allow cluster-scoped through.
			return true
		}
		if r.GetKind() == "ServiceAccount" {
			// Allow service accounts through, even though they
			// are in a namespace.  A RoleBinding in another namespace
			// can reference them.
			return true
		}
		return referrerCurId.IsNsEquals(r.CurId())
	}
}

// selectReferral picks the best referral from a list of candidates.
func (f Filter) selectReferral(
	// The name referral that may need to be updated.
	oldName string,
	candidates []*resource.Resource) (*resource.Resource, error) {
	candidates = doSieve(candidates, previousNameMatches(oldName))
	candidates = doSieve(candidates, previousIdSelectedByGvk(&f.ReferralTarget))
	candidates = doSieve(candidates, f.roleRefFilter())
	candidates = doSieve(candidates, f.sameCurrentNamespaceAsReferrer())
	if len(candidates) == 1 {
		return candidates[0], nil
	}
	candidates = doSieve(candidates, prefixSuffixEquals(f.Referrer))
	if len(candidates) == 1 {
		return candidates[0], nil
	}
	if len(candidates) == 0 {
		return nil, nil
	}
	if allNamesAreTheSame(candidates) {
		// Just take the first one.
		return candidates[0], nil
	}
	ids := getIds(candidates)
	f.failureDetails(candidates)
	return nil, fmt.Errorf(" found multiple possible referrals: %s", ids)
}

func (f Filter) failureDetails(resources []*resource.Resource) {
	fmt.Printf(
		"\n**** Too many possible referral targets to referrer:\n%s\n",
		f.Referrer.MustYaml())
	for i, r := range resources {
		fmt.Printf(
			"--- possible referral %d:\n%s", i, r.MustYaml())
		fmt.Println("------")
	}
}

func allNamesAreTheSame(resources []*resource.Resource) bool {
	name := resources[0].GetName()
	for i := 1; i < len(resources); i++ {
		if name != resources[i].GetName() {
			return false
		}
	}
	return true
}

func getIds(rs []*resource.Resource) string {
	var result []string
	for _, r := range rs {
		result = append(result, r.CurId().String())
	}
	return strings.Join(result, ", ")
}

func checkEqual(k, a, b string) error {
	if a != b {
		return fmt.Errorf(
			"node-referrerOriginal '%s' mismatch '%s' != '%s'",
			k, a, b)
	}
	return nil
}

func (f Filter) confirmNodeMatchesReferrer(node *yaml.RNode) error {
	meta, err := node.GetMeta()
	if err != nil {
		return err
	}
	gvk := f.Referrer.GetGvk()
	if err = checkEqual(
		"APIVersion", meta.APIVersion, gvk.ApiVersion()); err != nil {
		return err
	}
	if err = checkEqual(
		"Kind", meta.Kind, gvk.Kind); err != nil {
		return err
	}
	if err = checkEqual(
		"Name", meta.Name, f.Referrer.GetName()); err != nil {
		return err
	}
	if err = checkEqual(
		"Namespace", meta.Namespace, f.Referrer.GetNamespace()); err != nil {
		return err
	}
	return nil
}
