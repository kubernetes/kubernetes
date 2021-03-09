// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package provider

import (
	"log"

	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/api/internal/conflict"
	"sigs.k8s.io/kustomize/api/internal/validate"
	"sigs.k8s.io/kustomize/api/internal/wrappy"
	"sigs.k8s.io/kustomize/api/konfig"
	"sigs.k8s.io/kustomize/api/resource"
)

// DepProvider is a dependency provider.
//
// The instances it returns are either
//   - old implementations backed by k8sdeps code,
//   - new implementations backed by kyaml code.
//
// History:
//
// kubectl depends on k8s.io code, and at the time of writing, so
// does kustomize.  Code that imports k8s.io/api* cannot be imported
// back into k8s.io/*, yet kustomize appears inside k8s.io/kubectl.
//
// To allow kustomize to appear inside kubectl, yet still be developed
// outside kubectl, the kustomize code was divided into the following
// packages
//
//    api/
//      k8sdeps/  (and internal/ks8deps/)
//      ifc/
//      krusty/
//      everythingElse/
//
// with the following rules:
//
//   - Only k8sdeps/ may import k8s.io/api*.
//
//   - Only krusty/ (and its internals) may import k8sdeps/.
//     I.e., ifc/ and everythingElse/ must not
//     import k8sdeps/ or k8s.io/api*.
//
//   - Code in krusty/ may use code in k8sdeps/ to create
//     objects then inject said objects into
//     everythingElse/ behind dependency neutral interfaces.
//
// The idea was to periodically copy, not import, the large k8sdeps/
// tree (plus a snippet from krusty/kustomizer.go) into the kubectl
// codebase via a large PR, and have kubectl depend on the rest via
// normal importing.
//
// Over 2019, however, kubectl underwent large changes including
// a switch to Go modules, and a concerted attempt to extract kubectl
// from the k8s repo. This made large kustomize integration PRs too
// intrusive to review.
//
// In 2020, kubectl is based on Go modules, and almost entirely
// extracted from the k8s.io repositories, and further the kyaml
// library has a appeared as a viable replacement to k8s.io/api*
// KRM manipulation code.
//
// The new plan is to eliminate k8sdeps/ entirely, along with its
// k8s.io/api* dependence, allowing kustomize code to be imported
// into kubectl via normal Go module imports.  Then the kustomize API
// code can then move into the github.com/kubernetes-sigs/cli-utils
// repo.  The kustomize CLI in github.com/kubernetes-sigs/kustomize
// and the kubectl CLI can then both depend on the kustomize API.
//
// So, all code that depends on k8sdeps must go behind interfaces,
// and kustomize must be factored to choose the implementation.
//
// That problem has been reduced to three interfaces, each having
// two implementations. (1) is k8sdeps-based, (2) is kyaml-based.
//
//   - ifc.Kunstructured
//
//       1) api/k8sdeps/kunstruct.UnstructAdapter
//
//            This adapts structs in
//            k8s.io/apimachinery/pkg/apis/meta/v1/unstructured
//            to ifc.Kunstructured.
//
//       2) api/wrappy.WNode
//
//            This adapts sigs.k8s.io/kustomize/kyaml/yaml.RNode
//            to ifc.Unstructured.
//
//            At time of writing, implementation started.
//            Further reducing the size of ifc.Kunstructed
//            would really reduce the work
//            (e.g. drop Vars, drop ReplacementTranformer).
//
//   - resource.ConflictDetector
//
//       1) api/internal/k8sdeps/conflict.conflictDetectorJson
//          api/internal/k8sdeps/conflict.conflictDetectorSm
//
//            Uses k8s.io/apimachinery/pkg/util/strategicpatch,
//            apimachinery/pkg/util/mergepatch, etc. to merge
//            resource.Resource instances.
//
//       2) api/internal/conflict.smPatchMergeOnlyDetector
//
//           At time of writing, this doesn't report conflicts,
//           but it does know how to merge patches. Conflict
//           reporting isn't vital to kustomize function.  It's
//           rare that a person would configure one transformer
//           with many patches, much less so many that it became
//           hard to spot conflicts.  In the case of an undetected
//           conflict, the last patch applied wins, likely what
//           the user wants anyway.  Regardless, the effect of this
//           is plainly visible and usable in the output, even if
//           a conflict happened but wasn't reported as an error.
//
//   - ifc.Validator
//
//       1) api/k8sdeps/validator.KustValidator
//
//            Uses k8s.io/apimachinery/pkg/api/validation and
//            friends to validate strings.
//
//       2) api/internal/validate.FieldValidator
//
//            See TODO inside the validator for status.
//            At time of writing, this is a do-nothing
//            validator as it's not critical to kustomize function.
//
// Proposed plan:
//  [x] Ship kustomize with the ability to switch from 1 to 2 via
//      an --enable_kyaml flag.
//  [x] Make --enable_kyaml true by default.
//  [x] When 2 is not noticeably more buggy than 1, delete 1.
//      I.e. delete k8sdeps/, transitively deleting all k8s.io/api* deps.
//      This DepProvider should be left in place to retain these
//      comments, but it will have only one choice.
//  [x] The way is now clear to reintegrate into kubectl.
//      This should be done ASAP; the last step is cleanup.
//  [ ] Cleanup.  With only one impl of Kunstructure remaining,
//      that interface and WNode can be deleted, along with this
//      DepProvider.  The other two interfaces could be dropped too.
//
// When the above is done, kustomize will use yaml.RNode and/or
// KRM Config Functions directly and exclusively.
// If you're reading this, plan not done.
//
type DepProvider struct {
	kFactory                 ifc.KunstructuredFactory
	resourceFactory          *resource.Factory
	conflictDectectorFactory resource.ConflictDetectorFactory
	fieldValidator           ifc.Validator
}

// The dependencies this method needs have been deleted -
// see comments above.  This method will be deleted
// along with DepProvider in the final step.
func makeK8sdepBasedInstances() *DepProvider {
	log.Fatal("This binary cannot use k8s.io code; it must use kyaml.")
	return nil
}

func makeKyamlBasedInstances() *DepProvider {
	kf := &wrappy.WNodeFactory{}
	rf := resource.NewFactory(kf)
	return &DepProvider{
		kFactory:                 kf,
		resourceFactory:          rf,
		conflictDectectorFactory: conflict.NewFactory(),
		fieldValidator:           validate.NewFieldValidator(),
	}
}

func NewDepProvider(useKyaml bool) *DepProvider {
	if useKyaml {
		return makeKyamlBasedInstances()
	}
	return makeK8sdepBasedInstances()
}

func NewDefaultDepProvider() *DepProvider {
	return NewDepProvider(konfig.FlagEnableKyamlDefaultValue)
}

func (dp *DepProvider) GetKunstructuredFactory() ifc.KunstructuredFactory {
	return dp.kFactory
}

func (dp *DepProvider) GetResourceFactory() *resource.Factory {
	return dp.resourceFactory
}

func (dp *DepProvider) GetConflictDetectorFactory() resource.ConflictDetectorFactory {
	return dp.conflictDectectorFactory
}

func (dp *DepProvider) GetFieldValidator() ifc.Validator {
	return dp.fieldValidator
}
