/*
Copyright 2016 The Kubernetes Authors.

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

package generators

import (
	"bytes"
	"fmt"
	"io"
	"path"
	"reflect"
	"sort"
	"strings"

	"k8s.io/code-generator/cmd/conversion-gen/args"
	"k8s.io/code-generator/pkg/apidefinitions"
	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
	"k8s.io/klog/v2"
)

// These are the comment tags that carry parameters for conversion generation.
const (
	// e.g., "+k8s:conversion-gen=<peer-pkg>" in doc.go, where <peer-pkg> is the
	// import path of the package the peer types are defined in.
	// e.g., "+k8s:conversion-gen=false" in a type's comment will let
	// conversion-gen skip that type.
	tagName = "k8s:conversion-gen"
	// e.g. "+k8s:conversion-gen:explicit-from=net/url.Values" in the type comment
	// will result in generating conversion from net/url.Values.
	explicitFromTagName = "k8s:conversion-gen:explicit-from"
)

func extractTagValues(tagName string, comments []string) ([]string, error) {
	tags, err := gengo.ExtractFunctionStyleCommentTags("+", []string{tagName}, comments)
	if err != nil {
		return nil, err
	}
	tagList, exists := tags[tagName]
	if !exists {
		return nil, nil
	}
	values := make([]string, len(tagList))
	for i, v := range tagList {
		values[i] = v.Value
	}
	return values, nil
}

func extractTag(comments []string) ([]string, error) {
	return extractTagValues(tagName, comments)
}

func extractExplicitFromTag(comments []string) ([]string, error) {
	return extractTagValues(explicitFromTagName, comments)
}

func isCopyOnly(comments []string) (bool, error) {
	values, err := extractTagValues("k8s:conversion-fn", comments)
	if err != nil {
		return false, err
	}
	return len(values) == 1 && values[0] == "copy-only", nil
}

func isDrop(comments []string) (bool, error) {
	values, err := extractTagValues("k8s:conversion-fn", comments)
	if err != nil {
		return false, err
	}
	return len(values) == 1 && values[0] == "drop", nil
}

type hubTagState int

const (
	hubTagUnset hubTagState = iota
	hubTagSet
	hubTagOptedOut
)

func extractHubType(t *types.Type) (hubTagState, error) {
	values, err := extractTagValues("k8s:hubType", t.CommentLines)
	if err != nil {
		return hubTagUnset, err
	}
	switch {
	case len(values) == 0:
		return hubTagUnset, nil
	case values[0] == "false":
		return hubTagOptedOut, nil
	default:
		return hubTagSet, nil
	}
}

// getGroupName returns the API group pkg belongs to. Peer package are consulted
// as needed to determine the group name.
func getGroupName(context *generator.Context, pkg *types.Package, peerPkgs []string) (string, bool, error) {
	candidates := []*types.Package{pkg}
	for _, peerPath := range peerPkgs {
		if peer := context.Universe[peerPath]; peer != nil {
			candidates = append(candidates, peer)
		}
	}
	for _, c := range candidates {
		group, ok, err := apidefinitions.GroupNameForPackage(c.Comments)
		if err != nil {
			return "", false, err
		}
		if ok {
			return group, true, nil
		}
	}
	return "", false, nil
}

// TODO: This is created only to reduce number of changes in a single PR.
// Remove it and use PublicNamer instead.
func conversionNamer() *namer.NameStrategy {
	return &namer.NameStrategy{
		Join: func(pre string, in []string, post string) string {
			return strings.Join(in, "_")
		},
		PrependPackageNames: 1,
	}
}

func defaultFnNamer() *namer.NameStrategy {
	return &namer.NameStrategy{
		Prefix: "SetDefaults_",
		Join: func(pre string, in []string, post string) string {
			return pre + strings.Join(in, "_") + post
		},
	}
}

// NameSystems returns the name system used by the generators in this package.
func NameSystems() namer.NameSystems {
	return namer.NameSystems{
		"public":    conversionNamer(),
		"raw":       namer.NewRawNamer("", nil),
		"defaultfn": defaultFnNamer(),
	}
}

// DefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func DefaultNameSystem() string {
	return "public"
}

func getPeerTypeFor(context *generator.Context, t *types.Type, potenialPeerPkgs []string) *types.Type {
	for _, ppp := range potenialPeerPkgs {
		p := context.Universe.Package(ppp)
		if p == nil {
			continue
		}
		if p.Has(t.Name.Name) {
			return p.Type(t.Name.Name)
		}
	}
	return nil
}

type conversionPair struct {
	inType  *types.Type
	outType *types.Type
}

// All of the types in conversions map are of type "DeclarationOf" with
// the underlying type being "Func".
type conversionFuncMap map[conversionPair]*types.Type

// Returns all manually-defined conversion functions in the package.
func getManualConversionFunctions(context *generator.Context, pkg *types.Package, manualMap conversionFuncMap) {
	if pkg == nil {
		klog.Warning("Skipping nil package passed to getManualConversionFunctions")
		return
	}
	klog.V(3).Infof("Scanning for conversion functions in %v", pkg.Path)

	scopeName := types.Ref(conversionPackagePath, "Scope").Name
	errorName := types.Ref("", "error").Name
	buffer := &bytes.Buffer{}
	sw := generator.NewSnippetWriter(buffer, context, "$", "$")

	for _, f := range pkg.Functions {
		if f.Underlying == nil || f.Underlying.Kind != types.Func {
			klog.Errorf("Malformed function: %#v", f)
			continue
		}
		if f.Underlying.Signature == nil {
			klog.Errorf("Function without signature: %#v", f)
			continue
		}
		klog.V(6).Infof("Considering function %s", f.Name)
		signature := f.Underlying.Signature
		// Check whether the function is conversion function.
		// Note that all of them have signature:
		// func Convert_inType_To_outType(inType, outType, conversion.Scope) error
		if signature.Receiver != nil {
			klog.V(6).Infof("%s has a receiver", f.Name)
			continue
		}
		if len(signature.Parameters) != 3 || signature.Parameters[2].Type.Name != scopeName {
			klog.V(6).Infof("%s has wrong parameters", f.Name)
			continue
		}
		if len(signature.Results) != 1 || signature.Results[0].Type.Name != errorName {
			klog.V(6).Infof("%s has wrong results", f.Name)
			continue
		}
		inType := signature.Parameters[0].Type
		outType := signature.Parameters[1].Type
		if inType.Kind != types.Pointer || outType.Kind != types.Pointer {
			klog.V(6).Infof("%s has wrong parameter types", f.Name)
			continue
		}
		// Now check if the name satisfies the convention.
		// TODO: This should call the Namer directly.
		args := argsFromType(inType.Elem, outType.Elem)
		sw.Do("Convert_$.inType|public$_To_$.outType|public$", args)
		if f.Name.Name == buffer.String() {
			klog.V(2).Infof("Found conversion function %s", f.Name)
			key := conversionPair{inType.Elem, outType.Elem}
			// We might scan the same package twice, and that's OK.
			if v, ok := manualMap[key]; ok && v != nil && v.Name.Package != pkg.Path {
				panic(fmt.Sprintf("duplicate static conversion defined: %s -> %s from:\n%s.%s\n%s.%s", key.inType, key.outType, v.Name.Package, v.Name.Name, f.Name.Package, f.Name.Name))
			}
			manualMap[key] = f
		} else {
			// prevent user error when they don't get the correct conversion signature
			if strings.HasPrefix(f.Name.Name, "Convert_") {
				klog.Errorf("Rename function %s %s -> %s to match expected conversion signature", f.Name.Package, f.Name.Name, buffer.String())
			}
			klog.V(3).Infof("%s has wrong name", f.Name)
		}
		buffer.Reset()
	}
}

func GetTargets(context *generator.Context, args *args.Args) []generator.Target {
	boilerplate, err := gengo.GoBoilerplate(args.GoHeaderFile, args.GeneratedBuildTag, gengo.StdGeneratedBy)
	if err != nil {
		klog.Fatalf("Failed loading boilerplate: %v", err)
	}

	var idOpts []apidefinitions.Option
	if len(args.LintRules) > 0 {
		idOpts = append(idOpts, apidefinitions.WithLintRules(args.LintRules...))
	}

	targetList := []generator.Target{}

	// Accumulate pre-existing conversion functions.
	// TODO: This is too ad-hoc.  We need a better way.
	manualConversions := conversionFuncMap{}

	// Record types that are memory equivalent. A type is memory equivalent
	// if it has the same memory layout and no nested manual conversion is
	// defined.
	// TODO: in the future, relax the nested manual conversion requirement
	//   if we can show that a large enough types are memory identical but
	//   have non-trivial conversion
	memoryEquivalentTypes := equalMemoryTypes{}

	// First load other "input" packages.  We do this as a single call because
	// it is MUCH faster.
	filteredInputs := make([]string, 0, len(context.Inputs))
	otherPkgs := make([]string, 0, len(context.Inputs))
	pkgToPeers := map[string][]string{}
	pkgToExternal := map[string]string{}

	for _, i := range context.Inputs {
		klog.V(3).Infof("considering pkg %q", i)
		pkg := context.Universe[i]

		info, err := apidefinitions.Identify(pkg, apidefinitions.Conversion, idOpts...)
		if err != nil {
			klog.Fatal(err)
		}
		if !info.ShouldGenerate() {
			klog.V(3).Infof("  no tag")
			continue
		}
		filteredInputs = append(filteredInputs, i)

		// Sole +k8s:conversion-gen=false: emit only the package's
		// hand-written conversions, no peer-driven standard conversions.
		if !info.IsExplicitOnly() {
			peerPkgs := info.PeerPackages()
			klog.V(3).Infof("  peers: %q", peerPkgs)
			pkgToPeers[i] = peerPkgs
			otherPkgs = append(otherPkgs, peerPkgs...)
		}

		externalTypes := info.ExternalTypes()
		if externalTypes != i {
			klog.V(3).Infof("  external types: %q", externalTypes)
			otherPkgs = append(otherPkgs, externalTypes)
		}
		pkgToExternal[i] = externalTypes
	}

	// Make sure explicit peer-packages are added.
	peers := args.BasePeerDirs
	peers = append(peers, args.ExtraPeerDirs...)
	if expanded, err := context.FindPackages(peers...); err != nil {
		klog.Fatalf("cannot find peer packages: %v", err)
	} else {
		otherPkgs = append(otherPkgs, expanded...)
		// for each pkg, add these extras, too
		for k := range pkgToPeers {
			pkgToPeers[k] = append(pkgToPeers[k], expanded...)
		}
	}

	if len(otherPkgs) > 0 {
		if _, err := context.LoadPackages(otherPkgs...); err != nil {
			klog.Fatalf("cannot load packages: %v", err)
		}
	}
	// update context.Order to the latest context.Universe
	orderer := namer.Orderer{Namer: namer.NewPublicNamer(1)}
	context.Order = orderer.OrderUniverse(context.Universe)

	// Look for conversion functions in the peer-packages.
	for _, pp := range otherPkgs {
		p := context.Universe[pp]
		if p == nil {
			klog.Fatalf("failed to find pkg: %s", pp)
		}
		getManualConversionFunctions(context, p, manualConversions)
	}

	// We are generating conversions only for packages that are explicitly
	// passed as InputDir.
	for _, i := range filteredInputs {
		klog.V(3).Infof("considering pkg %q", i)
		pkg := context.Universe[i]

		// Add conversion and defaulting functions.
		getManualConversionFunctions(context, pkg, manualConversions)

		// Find the right input pkg, which might not be this one.
		externalTypes := pkgToExternal[i]

		// typesPkg is where the versioned types are defined. Sometimes it is
		// different from pkg. For example, kubernetes core/v1 types are defined
		// in k8s.io/api/core/v1, while pkg is at pkg/api/v1.
		typesPkg := context.Universe[externalTypes]

		unsafeEquality := TypesEqual(memoryEquivalentTypes)
		if args.SkipUnsafe {
			unsafeEquality = noEquality{}
		}

		targetList = append(targetList,
			&generator.SimpleTarget{
				PkgName:       path.Base(pkg.Path),
				PkgPath:       pkg.Path,
				PkgDir:        pkg.Dir, // output pkg is the same as the input
				HeaderComment: boilerplate,
				FilterFunc: func(c *generator.Context, t *types.Type) bool {
					return t.Name.Package == typesPkg.Path
				},
				GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
					return []generator.Generator{
						NewGenConversion(args.OutputFile, typesPkg.Path, pkg.Path, manualConversions, pkgToPeers[pkg.Path], unsafeEquality),
					}
				},
			})
	}

	// If there is a manual conversion defined between two types, exclude it
	// from being a candidate for unsafe conversion
	for k, v := range manualConversions {
		copyOnly, err := isCopyOnly(v.CommentLines)
		if err != nil {
			klog.Errorf("error extracting tags: %v", err)
		} else if copyOnly {
			klog.V(4).Infof("Conversion function %s will not block memory copy because it is copy-only", v.Name)
			continue
		}
		// this type should be excluded from all equivalence, because the converter must be called.
		memoryEquivalentTypes.Skip(k.inType, k.outType)
	}

	if err := validateAndCheckRules(context, args, filteredInputs, pkgToPeers, pkgToExternal, memoryEquivalentTypes); err != nil {
		klog.Fatalf("%v", err)
	}

	return targetList
}

func validateAndCheckRules(context *generator.Context, args *args.Args, filteredInputs []string, pkgToPeers map[string][]string, pkgToExternal map[string]string, memoryEquivalentTypes equalMemoryTypes) error {
	h, err := newHub(context, pkgToPeers, pkgToExternal, filteredInputs)
	if err != nil {
		return err
	}
	if err := h.checkConversionLintRules(); err != nil {
		return err
	}
	return h.reportViolations(context, args, memoryEquivalentTypes)
}

// hubs holds the +k8s:hubType types discovered across a set of conversion inputs.
type hubs struct {
	pkgToPeers    map[string][]string
	pkgToExternal map[string]string

	groupTypeToHubPkg       map[groupAndTypeName][]pkgAndType
	groupToInternalPkgPaths map[string]map[string]bool
}

type groupAndTypeName struct {
	group string
	name  string
}

type pkgAndType struct {
	pkgPath string
	t       *types.Type
}

func newHub(context *generator.Context, pkgToPeers map[string][]string, pkgToExternal map[string]string, filteredInputs []string) (*hubs, error) {
	h := &hubs{pkgToPeers: pkgToPeers, pkgToExternal: pkgToExternal}
	h.groupTypeToHubPkg = map[groupAndTypeName][]pkgAndType{}
	h.groupToInternalPkgPaths = map[string]map[string]bool{}
	groupToVersioned := map[string][]string{}

	for _, inputPath := range filteredInputs {
		pkg := context.Universe[inputPath]
		if pkg == nil {
			continue
		}
		groupName, ok, err := getGroupName(context, pkg, h.pkgToPeers[inputPath])
		if err != nil {
			return nil, fmt.Errorf("error getting group name for pkg %s: %w", inputPath, err)
		}
		if !ok {
			continue
		}
		groupToVersioned[groupName] = append(groupToVersioned[groupName], inputPath)

		if h.groupToInternalPkgPaths[groupName] == nil {
			h.groupToInternalPkgPaths[groupName] = map[string]bool{}
		}
		info, err := apidefinitions.Identify(pkg, apidefinitions.Conversion)
		if err != nil {
			return nil, fmt.Errorf("failed to identify package %s: %w", inputPath, err)
		}
		for _, peer := range info.PeerPackages() {
			h.groupToInternalPkgPaths[groupName][peer] = true
		}
	}

	for groupName, versionedPkgs := range groupToVersioned {
		for _, pkgPath := range versionedPkgs {
			pkg := context.Universe[h.pkgToExternal[pkgPath]]
			if pkg == nil {
				continue
			}
			for _, t := range pkg.Types {
				if t.Kind != types.Struct {
					continue
				}
				state, err := extractHubType(t)
				if err != nil {
					return nil, fmt.Errorf("error checking hubType tag for type %s in %s: %w", t.Name, pkgPath, err)
				}
				if state == hubTagSet {
					gt := groupAndTypeName{group: groupName, name: t.Name.Name}
					h.groupTypeToHubPkg[gt] = append(h.groupTypeToHubPkg[gt], pkgAndType{pkgPath: pkgPath, t: t})
				}
			}
		}
	}
	return h, nil
}

// checkConversionLintRules enforces conversion specific lint rules.
func (h *hubs) checkConversionLintRules() error {
	for gt, locs := range h.groupTypeToHubPkg {
		if len(locs) > 1 {
			var paths []string
			for _, l := range locs {
				paths = append(paths, l.pkgPath)
			}
			return fmt.Errorf("Type %q in group %q has multiple hub types: %v", gt.name, gt.group, paths)
		}
	}
	return nil
}

// requireHubTypesLintRule aliases the args-package rule name so it can be referenced
// where the args package identifier is shadowed by an *args.Args parameter.
const requireHubTypesLintRule = args.RequireHubTypesLintRule

// reportViolations collects conversion-gen's API rule violations.
func (h *hubs) reportViolations(context *generator.Context, args *args.Args, memoryEquivalentTypes equalMemoryTypes) error {
	var violations []violation

	// hub_memory_identity check
	for gt, locs := range h.groupTypeToHubPkg {
		if len(locs) != 1 {
			continue
		}
		loc := locs[0]
		var peerType *types.Type
		for ipkgPath := range h.groupToInternalPkgPaths[gt.group] {
			ipkg := context.Universe[ipkgPath]
			if ipkg == nil {
				continue
			}
			if t, ok := ipkg.Types[gt.name]; ok {
				peerType = t
				break
			}
		}
		if peerType == nil {
			klog.V(3).Infof("Hub type %s in %s has no peer internal type in group %s, skipping check", gt.name, loc.pkgPath, gt.group)
			continue
		}
		for _, d := range memoryEquivalentTypes.Divergences(loc.t, peerType) {
			violations = append(violations, violation{hubMemoryIdentityRule, loc.t.Name.Package, gt.name, d.Path})
		}
	}

	// hub_type_missing check.
	if args.HasRule(requireHubTypesLintRule) {
		for groupName, ipkgs := range h.groupToInternalPkgPaths {
			for ipkgPath := range ipkgs {
				ipkg := context.Universe[ipkgPath]
				if ipkg == nil {
					continue
				}
				for _, t := range ipkg.Types {
					if t.Kind != types.Struct {
						continue
					}
					state, err := extractHubType(t)
					if err != nil {
						return fmt.Errorf("error checking opt-out for internal type %s: %w", t.Name, err)
					}
					if state == hubTagOptedOut {
						continue
					}
					if len(h.groupTypeToHubPkg[groupAndTypeName{group: groupName, name: t.Name.Name}]) == 0 {
						violations = append(violations, violation{hubTypeMissingRule, ipkgPath, t.Name.Name, ""})
					}
				}
			}
		}
	}

	if args.ReportFilename == "" {
		return nil
	}
	return writeViolationReport(args.ReportFilename, violations)
}

type equalMemoryTypes map[conversionPair]string

func (e equalMemoryTypes) Skip(a, b *types.Type) {
	e[conversionPair{a, b}] = "manual conversion defined"
	e[conversionPair{b, a}] = "manual conversion defined"
}

func (e equalMemoryTypes) Equal(a, b *types.Type) bool {
	reason, _ := e.cachingEqual(a, b, nil)
	return reason == ""
}

// Divergence is one memory-identity difference between two types.
type Divergence struct {
	// Path is the type path to the field with the difference.
	// ".FieldName" for field types "[*]" for slice types, "[key]" for map key types, "[value]" for
	// map value types.
	Path string
	// Detail is a human-readable description of the divergence at Path.
	Detail string
}

// Divergences returns every structural difference between two types.
func (e equalMemoryTypes) Divergences(a, b *types.Type) []Divergence {
	var out []Divergence
	e.divergences("", a, b, nil, &out)
	sort.Slice(out, func(i, j int) bool {
		if out[i].Path != out[j].Path {
			return out[i].Path < out[j].Path
		}
		return out[i].Detail < out[j].Detail
	})
	return out
}

func (e equalMemoryTypes) divergences(path string, a, b *types.Type, stack []conversionPair, out *[]Divergence) {
	if a == b {
		return
	}
	// check both directions of conversion
	if r, ok := e[conversionPair{a, b}]; ok && r == "manual conversion defined" {
		*out = append(*out, Divergence{Path: path, Detail: r})
		return
	}
	if r, ok := e[conversionPair{b, a}]; ok && r == "manual conversion defined" {
		*out = append(*out, Divergence{Path: path, Detail: r})
		return
	}

	in, ou := unwrapAlias(a), unwrapAlias(b)
	if in == ou {
		return
	}
	if in.Kind != ou.Kind {
		*out = append(*out, Divergence{Path: path, Detail: fmt.Sprintf("different kinds: %s vs %s", in.Kind, ou.Kind)})
		return
	}
	for _, v := range stack {
		if v.inType == in && v.outType == ou {
			return
		}
	}
	stack = append(stack, conversionPair{in, ou})

	switch in.Kind {
	case types.Struct:
		inByName := map[string]types.Member{}
		for _, m := range in.Members {
			inByName[m.Name] = m
		}
		ouByName := map[string]types.Member{}
		for _, m := range ou.Members {
			ouByName[m.Name] = m
		}
		// Find all extra/missing fields
		var extra, missing []string
		for _, m := range in.Members {
			if _, ok := ouByName[m.Name]; !ok {
				extra = append(extra, m.Name)
			}
		}
		for _, m := range ou.Members {
			if _, ok := inByName[m.Name]; !ok {
				missing = append(missing, m.Name)
			}
		}
		sort.Strings(extra)
		sort.Strings(missing)
		for _, n := range extra {
			*out = append(*out, Divergence{Path: joinField(path, n), Detail: "extra field in external type"})
		}
		for _, n := range missing {
			*out = append(*out, Divergence{Path: joinField(path, n), Detail: "missing field in external type (present in internal)"})
		}
		// Field-order mismatch
		if len(extra) == 0 && len(missing) == 0 {
			for i := 0; i < len(in.Members) && i < len(ou.Members); i++ {
				if in.Members[i].Name != ou.Members[i].Name {
					*out = append(*out, Divergence{Path: path, Detail: fmt.Sprintf("field order mismatch at index %d", i)})
					break
				}
			}
		}
		// Check fields
		for _, m := range in.Members {
			om, ok := ouByName[m.Name]
			if !ok {
				continue
			}
			e.divergences(joinField(path, m.Name), m.Type, om.Type, stack, out)
		}
	case types.Pointer:
		e.divergences(path, in.Elem, ou.Elem, stack, out)
	case types.Map:
		e.divergences(path+"[key]", in.Key, ou.Key, stack, out)
		e.divergences(path+"[value]", in.Elem, ou.Elem, stack, out)
	case types.Slice:
		e.divergences(path+"[*]", in.Elem, ou.Elem, stack, out)
	case types.Interface:
		*out = append(*out, Divergence{Path: path, Detail: "interfaces are not supported for memory equality"})
	case types.Builtin:
		if in.Name.Name != ou.Name.Name {
			*out = append(*out, Divergence{Path: path, Detail: fmt.Sprintf("different builtin types: %s vs %s", in.Name.Name, ou.Name.Name)})
		}
	default:
		*out = append(*out, Divergence{Path: path, Detail: fmt.Sprintf("kind %s is not supported for zero-copy conversion", in.Kind)})
	}
}

func joinField(path, field string) string {
	if path == "" {
		return field
	}
	return path + "." + field
}

func (e equalMemoryTypes) cachingEqual(a, b *types.Type, alreadyVisitedStack []conversionPair) (reason string, cacheable bool) {
	if a == b {
		return "", true
	}
	if reason, ok := e[conversionPair{a, b}]; ok {
		return reason, true
	}
	if reason, ok := e[conversionPair{b, a}]; ok {
		return reason, true
	}
	resReason, cacheable := e.equal(a, b, alreadyVisitedStack)
	if cacheable {
		e[conversionPair{a, b}] = resReason
		e[conversionPair{b, a}] = resReason
	}
	return resReason, cacheable
}

// equal checks if a and b are memory-identical.
func (e equalMemoryTypes) equal(a, b *types.Type, alreadyVisitedStack []conversionPair) (reason string, cacheable bool) {
	in, out := unwrapAlias(a), unwrapAlias(b)
	switch {
	case in == out:
		return "", true
	case in.Kind == out.Kind:
		for _, v := range alreadyVisitedStack {
			if v.inType == in && v.outType == out {
				return "", false
			}
		}
		alreadyVisitedStack = append(alreadyVisitedStack, conversionPair{in, out})

		switch in.Kind {
		case types.Struct:
			// Check for missing/extra fields by name
			inMembers := map[string]types.Member{}
			for _, m := range in.Members {
				inMembers[m.Name] = m
			}
			outMembers := map[string]types.Member{}
			for _, m := range out.Members {
				outMembers[m.Name] = m
			}

			var extraIn []string
			for _, m := range in.Members {
				if _, ok := outMembers[m.Name]; !ok {
					extraIn = append(extraIn, m.Name)
				}
			}
			var extraOut []string
			for _, m := range out.Members {
				if _, ok := inMembers[m.Name]; !ok {
					extraOut = append(extraOut, m.Name)
				}
			}

			if len(extraIn) > 0 || len(extraOut) > 0 {
				var diffs []string
				for _, name := range extraIn {
					diffs = append(diffs, fmt.Sprintf("extra field in external type: %s", name))
				}
				for _, name := range extraOut {
					diffs = append(diffs, fmt.Sprintf("missing field in external type (present in internal): %s", name))
				}
				return "\n" + strings.Join(diffs, "\n"), true
			}

			// Check fields
			var diffs []string
			cacheable = true
			for i := 0; i < len(in.Members); i++ {
				inMember := in.Members[i]
				outMember := out.Members[i]

				if inMember.Name != outMember.Name {
					diffs = append(diffs, fmt.Sprintf("field order mismatch at index %d: external has %q, internal has %q", i, inMember.Name, outMember.Name))
					break
				}

				memberReason, memberCacheable := e.cachingEqual(inMember.Type, outMember.Type, alreadyVisitedStack)
				if memberReason != "" {
					diffs = append(diffs, formatNestedReason(fmt.Sprintf("member %q", inMember.Name), memberReason))
				}
				if !memberCacheable {
					cacheable = false
				}
			}
			if len(diffs) > 0 {
				return "\n" + strings.Join(diffs, "\n"), cacheable
			}
			return "", cacheable
		case types.Pointer:
			reason, cacheable := e.cachingEqual(in.Elem, out.Elem, alreadyVisitedStack)
			if reason != "" {
				return formatNestedReason("pointer element", reason), cacheable
			}
			return "", cacheable
		case types.Map:
			keyReason, keyCacheable := e.cachingEqual(in.Key, out.Key, alreadyVisitedStack)
			if keyReason != "" {
				return formatNestedReason("map key", keyReason), keyCacheable
			}
			valueReason, valueCacheable := e.cachingEqual(in.Elem, out.Elem, alreadyVisitedStack)
			if valueReason != "" {
				return formatNestedReason("map value", valueReason), valueCacheable
			}
			return "", keyCacheable && valueCacheable
		case types.Slice:
			reason, cacheable := e.cachingEqual(in.Elem, out.Elem, alreadyVisitedStack)
			if reason != "" {
				return formatNestedReason("slice element", reason), cacheable
			}
			return "", cacheable
		case types.Builtin:
			if in.Name.Name != out.Name.Name {
				return fmt.Sprintf("different builtin types: %s vs %s", in.Name.Name, out.Name.Name), true
			}
			return "", true
		default:
			return fmt.Sprintf("%s not supported for memory equality", in.Kind), true
		}
	}
	return fmt.Sprintf("different kinds: %s vs %s", in.Kind, out.Kind), true
}

func findMember(t *types.Type, name string) (types.Member, bool) {
	if t.Kind != types.Struct {
		return types.Member{}, false
	}
	for _, member := range t.Members {
		if member.Name == name {
			return member, true
		}
	}
	return types.Member{}, false
}

// unwrapAlias recurses down aliased types to find the bedrock type.
func unwrapAlias(in *types.Type) *types.Type {
	for in.Kind == types.Alias {
		in = in.Underlying
	}
	return in
}

func indent(s, prefix string) string {
	lines := strings.Split(s, "\n")
	for i, line := range lines {
		if line != "" {
			lines[i] = prefix + line
		}
	}
	return strings.Join(lines, "\n")
}

func formatNestedReason(label string, reason string) string {
	if strings.HasPrefix(reason, "\n") {
		return fmt.Sprintf("%s:%s", label, indent(reason, "  "))
	}
	return fmt.Sprintf("%s: %s", label, reason)
}

const (
	runtimePackagePath    = "k8s.io/apimachinery/pkg/runtime"
	conversionPackagePath = "k8s.io/apimachinery/pkg/conversion"
)

type noEquality struct{}

func (noEquality) Equal(_, _ *types.Type) bool { return false }

type TypesEqual interface {
	Equal(a, b *types.Type) bool
}

// genConversion produces a file with a autogenerated conversions.
type genConversion struct {
	generator.GoGenerator
	// the package that contains the types that conversion func are going to be
	// generated for
	typesPackage string
	// the package that the conversion funcs are going to be output to
	outputPackage string
	// packages that contain the peer of types in typesPacakge
	peerPackages        []string
	manualConversions   conversionFuncMap
	imports             namer.ImportTracker
	types               []*types.Type
	explicitConversions []conversionPair
	skippedFields       map[*types.Type][]string
	useUnsafe           TypesEqual
}

func NewGenConversion(outputFilename, typesPackage, outputPackage string, manualConversions conversionFuncMap, peerPkgs []string, useUnsafe TypesEqual) generator.Generator {
	return &genConversion{
		GoGenerator: generator.GoGenerator{
			OutputFilename: outputFilename,
		},
		typesPackage:        typesPackage,
		outputPackage:       outputPackage,
		peerPackages:        peerPkgs,
		manualConversions:   manualConversions,
		imports:             generator.NewImportTrackerForPackage(outputPackage),
		types:               []*types.Type{},
		explicitConversions: []conversionPair{},
		skippedFields:       map[*types.Type][]string{},
		useUnsafe:           useUnsafe,
	}
}

func (g *genConversion) Namers(c *generator.Context) namer.NameSystems {
	// Have the raw namer for this file track what it imports.
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.outputPackage, g.imports),
		"publicIT": &namerPlusImportTracking{
			delegate: conversionNamer(),
			tracker:  g.imports,
		},
	}
}

type namerPlusImportTracking struct {
	delegate namer.Namer
	tracker  namer.ImportTracker
}

func (n *namerPlusImportTracking) Name(t *types.Type) string {
	n.tracker.AddType(t)
	return n.delegate.Name(t)
}

func (g *genConversion) convertibleOnlyWithinPackage(inType, outType *types.Type) bool {
	var t *types.Type
	var other *types.Type
	if inType.Name.Package == g.typesPackage {
		t, other = inType, outType
	} else {
		t, other = outType, inType
	}

	if t.Name.Package != g.typesPackage {
		return false
	}
	// If the type has opted out, skip it.
	tagvals, err := extractTag(t.CommentLines)
	if err != nil {
		klog.Errorf("Type %v: error extracting tags: %v", t, err)
		return false
	}

	if tagvals != nil {
		if tagvals[0] != "false" {
			klog.Fatalf("Type %v: unsupported %s value: %q", t, tagName, tagvals[0])
		}
		klog.V(2).Infof("type %v requests no conversion generation, skipping", t)
		return false
	}
	// TODO: Consider generating functions for other kinds too.
	if t.Kind != types.Struct {
		return false
	}
	// Also, filter out private types.
	if namer.IsPrivateGoName(other.Name.Name) {
		return false
	}
	return true
}

func getExplicitFromTypes(t *types.Type) []types.Name {
	comments := t.SecondClosestCommentLines
	comments = append(comments, t.CommentLines...)
	result := []types.Name{}
	paths, err := extractExplicitFromTag(comments)
	if err != nil {
		klog.Errorf("Error extracting explicit-from tag for %v: %v", t.Name, err)
		return result
	}
	for _, path := range paths {
		items := strings.Split(path, ".")
		if len(items) != 2 {
			klog.Errorf("Unexpected k8s:conversion-gen:explicit-from tag: %s", path)
			continue
		}
		switch {
		case items[0] == "net/url" && items[1] == "Values":
		default:
			klog.Fatalf("Not supported k8s:conversion-gen:explicit-from tag: %s", path)
		}
		result = append(result, types.Name{Package: items[0], Name: items[1]})
	}
	return result
}

func (g *genConversion) Filter(c *generator.Context, t *types.Type) bool {
	convertibleWithPeer := func() bool {
		peerType := getPeerTypeFor(c, t, g.peerPackages)
		if peerType == nil {
			return false
		}
		if !g.convertibleOnlyWithinPackage(t, peerType) {
			return false
		}
		g.types = append(g.types, t)
		return true
	}()

	explicitlyConvertible := func() bool {
		inTypes := getExplicitFromTypes(t)
		if len(inTypes) == 0 {
			return false
		}
		for i := range inTypes {
			pair := conversionPair{
				inType:  &types.Type{Name: inTypes[i]},
				outType: t,
			}
			g.explicitConversions = append(g.explicitConversions, pair)
		}
		return true
	}()

	return convertibleWithPeer || explicitlyConvertible
}

func (g *genConversion) isOtherPackage(pkg string) bool {
	if pkg == g.outputPackage {
		return false
	}
	if strings.HasSuffix(pkg, `"`+g.outputPackage+`"`) {
		return false
	}
	return true
}

func (g *genConversion) Imports(c *generator.Context) (imports []string) {
	var importLines []string
	for _, singleImport := range g.imports.ImportLines() {
		if g.isOtherPackage(singleImport) {
			importLines = append(importLines, singleImport)
		}
	}
	return importLines
}

func argsFromType(inType, outType *types.Type) generator.Args {
	return generator.Args{
		"inType":  inType,
		"outType": outType,
	}
}

const nameTmpl = "Convert_$.inType|publicIT$_To_$.outType|publicIT$"

func (g *genConversion) preexists(inType, outType *types.Type) (*types.Type, bool) {
	function, ok := g.manualConversions[conversionPair{inType, outType}]
	return function, ok
}

func (g *genConversion) preexistsPointers(inType, outType *types.Type) (*types.Type, bool) {
	if inType.Kind != types.Pointer {
		return nil, false
	}
	if outType.Kind != types.Pointer {
		return nil, false
	}
	return g.preexists(inType.Elem, outType.Elem)
}

func (g *genConversion) Init(c *generator.Context, w io.Writer) error {
	klogV := klog.V(6)
	if klogV.Enabled() {
		if m, ok := g.useUnsafe.(equalMemoryTypes); ok {
			var result []string
			klogV.Info("All objects without identical memory layout:")
			for k, v := range m {
				if v == "" {
					continue
				}
				result = append(result, fmt.Sprintf("  %s -> %s: %s", k.inType, k.outType, v))
			}
			sort.Strings(result)
			for _, s := range result {
				klogV.Info(s)
			}
		}
	}
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	sw.Do("func init() {\n", nil)
	sw.Do("localSchemeBuilder.Register(RegisterConversions)\n", nil)
	sw.Do("}\n", nil)

	scheme := c.Universe.Type(types.Name{Package: runtimePackagePath, Name: "Scheme"})
	schemePtr := &types.Type{
		Kind: types.Pointer,
		Elem: scheme,
	}
	sw.Do("// RegisterConversions adds conversion functions to the given scheme.\n", nil)
	sw.Do("// Public to allow building arbitrary schemes.\n", nil)
	sw.Do("func RegisterConversions(s $.|raw$) error {\n", schemePtr)
	for _, t := range g.types {
		peerType := getPeerTypeFor(c, t, g.peerPackages)
		if _, found := g.preexists(t, peerType); !found {
			args := argsFromType(t, peerType).With("Scope", types.Ref(conversionPackagePath, "Scope"))
			sw.Do("if err := s.AddGeneratedConversionFunc((*$.inType|raw$)(nil), (*$.outType|raw$)(nil), func(a, b interface{}, scope $.Scope|raw$) error { return "+nameTmpl+"(a.(*$.inType|raw$), b.(*$.outType|raw$), scope) }); err != nil { return err }\n", args)
		}
		if _, found := g.preexists(peerType, t); !found {
			args := argsFromType(peerType, t).With("Scope", types.Ref(conversionPackagePath, "Scope"))
			sw.Do("if err := s.AddGeneratedConversionFunc((*$.inType|raw$)(nil), (*$.outType|raw$)(nil), func(a, b interface{}, scope $.Scope|raw$) error { return "+nameTmpl+"(a.(*$.inType|raw$), b.(*$.outType|raw$), scope) }); err != nil { return err }\n", args)
		}
	}

	for i := range g.explicitConversions {
		args := argsFromType(g.explicitConversions[i].inType, g.explicitConversions[i].outType).With("Scope", types.Ref(conversionPackagePath, "Scope"))
		sw.Do("if err := s.AddGeneratedConversionFunc((*$.inType|raw$)(nil), (*$.outType|raw$)(nil), func(a, b interface{}, scope $.Scope|raw$) error { return "+nameTmpl+"(a.(*$.inType|raw$), b.(*$.outType|raw$), scope) }); err != nil { return err }\n", args)
	}

	var pairs []conversionPair
	for pair, t := range g.manualConversions {
		if t.Name.Package != g.outputPackage {
			continue
		}
		pairs = append(pairs, pair)
	}
	// sort by name of the conversion function
	sort.Slice(pairs, func(i, j int) bool {
		return g.manualConversions[pairs[i]].Name.Name < g.manualConversions[pairs[j]].Name.Name
	})
	for _, pair := range pairs {
		args := argsFromType(pair.inType, pair.outType).With("Scope", types.Ref(conversionPackagePath, "Scope")).With("fn", g.manualConversions[pair])
		sw.Do("if err := s.AddConversionFunc((*$.inType|raw$)(nil), (*$.outType|raw$)(nil), func(a, b interface{}, scope $.Scope|raw$) error { return $.fn|raw$(a.(*$.inType|raw$), b.(*$.outType|raw$), scope) }); err != nil { return err }\n", args)
	}

	sw.Do("return nil\n", nil)
	sw.Do("}\n\n", nil)
	return sw.Error()
}

func (g *genConversion) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	klog.V(5).Infof("generating for type %v", t)
	sw := generator.NewSnippetWriter(w, c, "$", "$")

	if peerType := getPeerTypeFor(c, t, g.peerPackages); peerType != nil {
		g.generateConversion(t, peerType, sw)
		g.generateConversion(peerType, t, sw)
	}

	for _, inTypeName := range getExplicitFromTypes(t) {
		inPkg, ok := c.Universe[inTypeName.Package]
		if !ok {
			klog.Errorf("Unrecognized package: %s", inTypeName.Package)
			continue
		}
		inType, ok := inPkg.Types[inTypeName.Name]
		if !ok {
			klog.Errorf("Unrecognized type in package %s: %s", inTypeName.Package, inTypeName.Name)
			continue
		}
		switch {
		case inType.Name.Package == "net/url" && inType.Name.Name == "Values":
			g.generateFromURLValues(inType, t, sw)
		default:
			klog.Errorf("Not supported input type: %#v", inType.Name)
		}
	}

	return sw.Error()
}

func (g *genConversion) generateConversion(inType, outType *types.Type, sw *generator.SnippetWriter) {
	args := argsFromType(inType, outType).
		With("Scope", types.Ref(conversionPackagePath, "Scope"))

	sw.Do("func auto"+nameTmpl+"(in *$.inType|raw$, out *$.outType|raw$, s $.Scope|raw$) error {\n", args)
	g.generateFor(inType, outType, sw)
	sw.Do("return nil\n", nil)
	sw.Do("}\n\n", nil)

	if _, found := g.preexists(inType, outType); found {
		// There is a public manual Conversion method: use it.
	} else if skipped := g.skippedFields[inType]; len(skipped) != 0 {
		// The inType had some fields we could not generate.
		klog.Errorf("Warning: could not find nor generate a final Conversion function for %v -> %v", inType, outType)
		klog.Errorf("  the following fields need manual conversion:")
		for _, f := range skipped {
			klog.Errorf("      - %v", f)
		}
	} else {
		// Emit a public conversion function.
		sw.Do("// "+nameTmpl+" is an autogenerated conversion function.\n", args)
		sw.Do("func "+nameTmpl+"(in *$.inType|raw$, out *$.outType|raw$, s $.Scope|raw$) error {\n", args)
		sw.Do("return auto"+nameTmpl+"(in, out, s)\n", args)
		sw.Do("}\n\n", nil)
	}
}

// we use the system of shadowing 'in' and 'out' so that the same code is valid
// at any nesting level. This makes the autogenerator easy to understand, and
// the compiler shouldn't care.
func (g *genConversion) generateFor(inType, outType *types.Type, sw *generator.SnippetWriter) {
	klog.V(4).Infof("generating %v -> %v", inType, outType)
	var f func(*types.Type, *types.Type, *generator.SnippetWriter)

	switch inType.Kind {
	case types.Builtin:
		f = g.doBuiltin
	case types.Map:
		f = g.doMap
	case types.Slice:
		f = g.doSlice
	case types.Struct:
		f = g.doStruct
	case types.Pointer:
		f = g.doPointer
	case types.Alias:
		f = g.doAlias
	default:
		f = g.doUnknown
	}

	f(inType, outType, sw)
}

func (g *genConversion) doBuiltin(inType, outType *types.Type, sw *generator.SnippetWriter) {
	if inType == outType {
		sw.Do("*out = *in\n", nil)
	} else {
		sw.Do("*out = $.|raw$(*in)\n", outType)
	}
}

func (g *genConversion) doMap(inType, outType *types.Type, sw *generator.SnippetWriter) {
	sw.Do("*out = make($.|raw$, len(*in))\n", outType)
	if isDirectlyAssignable(inType.Key, outType.Key) {
		sw.Do("for key, val := range *in {\n", nil)
		if isDirectlyAssignable(inType.Elem, outType.Elem) {
			if inType.Key == outType.Key {
				sw.Do("(*out)[key] = ", nil)
			} else {
				sw.Do("(*out)[$.|raw$(key)] = ", outType.Key)
			}
			if inType.Elem == outType.Elem {
				sw.Do("val\n", nil)
			} else {
				sw.Do("$.|raw$(val)\n", outType.Elem)
			}
		} else {
			conversionExists := true
			conditionalConversionExists := false
			if function, ok := g.preexists(inType.Elem, outType.Elem); ok {
				sw.Do("newVal := new($.|raw$)\n", outType.Elem)
				sw.Do("if err := $.|raw$(&val, newVal, s); err != nil {\n", function)
			} else if function, ok := g.preexistsPointers(inType.Elem, outType.Elem); ok {
				sw.Do("newVal := new($.|raw$)\n", outType.Elem)
				sw.Do("if val != nil {\n", nil)
				sw.Do("*newVal = new($.|raw$)\n", outType.Elem.Elem)
				sw.Do("if err := $.|raw$(val, *newVal, s); err != nil {\n", function)
				conditionalConversionExists = true
			} else if g.convertibleOnlyWithinPackage(inType.Elem, outType.Elem) {
				sw.Do("newVal := new($.|raw$)\n", outType.Elem)
				sw.Do("if err := "+nameTmpl+"(&val, newVal, s); err != nil {\n", argsFromType(inType.Elem, outType.Elem))
			} else {
				args := argsFromType(inType.Elem, outType.Elem)
				sw.Do("// FIXME: Provide conversion function to convert $.inType|raw$ to $.outType|raw$\n", args)
				sw.Do("compileErrorOnMissingConversion()\n", nil)
				conversionExists = false
			}
			if conversionExists {
				sw.Do("return err\n", nil)
				sw.Do("}\n", nil)
				if conditionalConversionExists {
					sw.Do("}\n", nil)
				}
				if inType.Key == outType.Key {
					sw.Do("(*out)[key] = *newVal\n", nil)
				} else {
					sw.Do("(*out)[$.|raw$(key)] = *newVal\n", outType.Key)
				}
			}
		}
	} else {
		// TODO: Implement it when necessary.
		sw.Do("for range *in {\n", nil)
		sw.Do("// FIXME: Converting unassignable keys unsupported $.|raw$\n", inType.Key)
	}
	sw.Do("}\n", nil)
}

func (g *genConversion) doSlice(inType, outType *types.Type, sw *generator.SnippetWriter) {
	sw.Do("*out = make($.|raw$, len(*in))\n", outType)
	if inType.Elem == outType.Elem && inType.Elem.Kind == types.Builtin {
		sw.Do("copy(*out, *in)\n", nil)
	} else {
		sw.Do("for i := range *in {\n", nil)
		if isDirectlyAssignable(inType.Elem, outType.Elem) {
			if inType.Elem == outType.Elem {
				sw.Do("(*out)[i] = (*in)[i]\n", nil)
			} else {
				sw.Do("(*out)[i] = $.|raw$((*in)[i])\n", outType.Elem)
			}
		} else {
			conversionExists := true
			conditionalConversionExists := false
			if function, ok := g.preexists(inType.Elem, outType.Elem); ok {
				sw.Do("if err := $.|raw$(&(*in)[i], &(*out)[i], s); err != nil {\n", function)
			} else if function, ok := g.preexistsPointers(inType.Elem, outType.Elem); ok {
				sw.Do("if (*in)[i] != nil {\n", nil)
				sw.Do("(*out)[i] = new($.|raw$)\n", outType.Elem.Elem)
				sw.Do("if err := $.|raw$((*in)[i], (*out)[i], s); err != nil {\n", function)
				conditionalConversionExists = true
			} else if g.convertibleOnlyWithinPackage(inType.Elem, outType.Elem) {
				sw.Do("if err := "+nameTmpl+"(&(*in)[i], &(*out)[i], s); err != nil {\n", argsFromType(inType.Elem, outType.Elem))
			} else {
				args := argsFromType(inType.Elem, outType.Elem)
				sw.Do("// FIXME: Provide conversion function to convert $.inType|raw$ to $.outType|raw$\n", args)
				sw.Do("compileErrorOnMissingConversion()\n", nil)
				conversionExists = false
			}
			if conversionExists {
				sw.Do("return err\n", nil)
				sw.Do("}\n", nil)
				if conditionalConversionExists {
					sw.Do("}\n", nil)
				}
			}
		}
		sw.Do("}\n", nil)
	}
}

func (g *genConversion) doStruct(inType, outType *types.Type, sw *generator.SnippetWriter) {
	for _, inMember := range inType.Members {
		tagvals, err := extractTag(inMember.CommentLines)
		if err != nil {
			klog.Errorf("Member %v.%v: error extracting tags: %v", inType, inMember.Name, err)
		}
		if tagvals != nil && tagvals[0] == "false" {
			// This field is excluded from conversion.
			sw.Do("// INFO: in."+inMember.Name+" opted out of conversion generation\n", nil)
			continue
		}
		outMember, found := findMember(outType, inMember.Name)
		if !found {
			// This field doesn't exist in the peer.
			sw.Do("// WARNING: in."+inMember.Name+" requires manual conversion: does not exist in peer-type\n", nil)
			g.skippedFields[inType] = append(g.skippedFields[inType], inMember.Name)
			continue
		}

		if namer.IsPrivateGoName(inMember.Name) && g.outputPackage != inType.Name.Package {
			sw.Do("// WARNING: in."+inMember.Name+" is not exported and cannot be read\n", nil)
			g.skippedFields[inType] = append(g.skippedFields[inType], inMember.Name)
			continue
		}
		if namer.IsPrivateGoName(outMember.Name) && g.outputPackage != outType.Name.Package {
			sw.Do("// WARNING: out."+inMember.Name+" is not exported and cannot be set\n", nil)
			g.skippedFields[inType] = append(g.skippedFields[inType], inMember.Name)
			continue
		}

		inMemberType, outMemberType := inMember.Type, outMember.Type
		// create a copy of both underlying types but give them the top level alias name (since aliases
		// are assignable)
		if underlying := unwrapAlias(inMemberType); underlying != inMemberType {
			copied := *underlying
			copied.Name = inMemberType.Name
			inMemberType = &copied
		}
		if underlying := unwrapAlias(outMemberType); underlying != outMemberType {
			copied := *underlying
			copied.Name = outMemberType.Name
			outMemberType = &copied
		}

		args := argsFromType(inMemberType, outMemberType).With("name", inMember.Name)

		// try a direct memory copy for any type that has exactly equivalent values
		if g.useUnsafe.Equal(inMemberType, outMemberType) {
			args = args.
				With("Pointer", types.Ref("unsafe", "Pointer")).
				With("SliceHeader", types.Ref("reflect", "SliceHeader"))
			switch inMemberType.Kind {
			case types.Pointer:
				sw.Do("out.$.name$ = ($.outType|raw$)($.Pointer|raw$(in.$.name$))\n", args)
				continue
			case types.Map:
				sw.Do("out.$.name$ = *(*$.outType|raw$)($.Pointer|raw$(&in.$.name$))\n", args)
				continue
			case types.Slice:
				sw.Do("out.$.name$ = *(*$.outType|raw$)($.Pointer|raw$(&in.$.name$))\n", args)
				continue
			}
		}

		// check based on the top level name, not the underlying names
		if function, ok := g.preexists(inMember.Type, outMember.Type); ok {
			dropFn, err := isDrop(function.CommentLines)
			if err != nil {
				klog.Errorf("Error extracting drop tag for function %s: %v", function.Name, err)
			} else if dropFn {
				continue
			}
			// copy-only functions that are directly assignable can be inlined instead of invoked.
			// As an example, conversion functions exist that allow types with private fields to be
			// correctly copied between types. These functions are equivalent to a memory assignment,
			// and are necessary for the reflection path, but should not block memory conversion.
			// Convert_unversioned_Time_to_unversioned_Time is an example of this logic.
			copyOnly, copyErr := isCopyOnly(function.CommentLines)
			if copyErr != nil {
				klog.Errorf("Error extracting copy-only tag for function %s: %v", function.Name, copyErr)
				copyOnly = false
			}
			if !copyOnly || !g.isFastConversion(inMemberType, outMemberType) {
				args["function"] = function
				sw.Do("if err := $.function|raw$(&in.$.name$, &out.$.name$, s); err != nil {\n", args)
				sw.Do("return err\n", nil)
				sw.Do("}\n", nil)
				continue
			}
			klog.V(2).Infof("Skipped function %s because it is copy-only and we can use direct assignment", function.Name)
		}

		// If we can't auto-convert, punt before we emit any code.
		if inMemberType.Kind != outMemberType.Kind {
			sw.Do("// WARNING: in."+inMember.Name+" requires manual conversion: inconvertible types ("+
				inMemberType.String()+" vs "+outMemberType.String()+")\n", nil)
			g.skippedFields[inType] = append(g.skippedFields[inType], inMember.Name)
			continue
		}

		switch inMemberType.Kind {
		case types.Builtin:
			if inMemberType == outMemberType {
				sw.Do("out.$.name$ = in.$.name$\n", args)
			} else {
				sw.Do("out.$.name$ = $.outType|raw$(in.$.name$)\n", args)
			}
		case types.Map, types.Slice, types.Pointer:
			if g.isDirectlyAssignable(inMemberType, outMemberType) {
				sw.Do("out.$.name$ = in.$.name$\n", args)
				continue
			}

			sw.Do("if in.$.name$ != nil {\n", args)
			sw.Do("in, out := &in.$.name$, &out.$.name$\n", args)
			g.generateFor(inMemberType, outMemberType, sw)
			sw.Do("} else {\n", nil)
			sw.Do("out.$.name$ = nil\n", args)
			sw.Do("}\n", nil)
		case types.Struct:
			if g.isDirectlyAssignable(inMemberType, outMemberType) {
				sw.Do("out.$.name$ = in.$.name$\n", args)
				continue
			}
			conversionExists := true
			if g.convertibleOnlyWithinPackage(inMemberType, outMemberType) {
				sw.Do("if err := "+nameTmpl+"(&in.$.name$, &out.$.name$, s); err != nil {\n", args)
			} else {
				args := argsFromType(inMemberType, outMemberType)
				sw.Do("// FIXME: Provide conversion function to convert $.inType|raw$ to $.outType|raw$\n", args)
				sw.Do("compileErrorOnMissingConversion()\n", nil)
				conversionExists = false
			}
			if conversionExists {
				sw.Do("return err\n", nil)
				sw.Do("}\n", nil)
			}
		case types.Alias:
			if isDirectlyAssignable(inMemberType, outMemberType) {
				if inMemberType == outMemberType {
					sw.Do("out.$.name$ = in.$.name$\n", args)
				} else {
					sw.Do("out.$.name$ = $.outType|raw$(in.$.name$)\n", args)
				}
			} else {
				conversionExists := true
				if g.convertibleOnlyWithinPackage(inMemberType, outMemberType) {
					sw.Do("if err := "+nameTmpl+"(&in.$.name$, &out.$.name$, s); err != nil {\n", args)
				} else {
					args := argsFromType(inMemberType, outMemberType)
					sw.Do("// FIXME: Provide conversion function to convert $.inType|raw$ to $.outType|raw$\n", args)
					sw.Do("compileErrorOnMissingConversion()\n", nil)
					conversionExists = false
				}
				if conversionExists {
					sw.Do("return err\n", nil)
					sw.Do("}\n", nil)
				}
			}
		default:
			conversionExists := true
			if g.convertibleOnlyWithinPackage(inMemberType, outMemberType) {
				sw.Do("if err := "+nameTmpl+"(&in.$.name$, &out.$.name$, s); err != nil {\n", args)
			} else {
				args := argsFromType(inMemberType, outMemberType)
				sw.Do("// FIXME: Provide conversion function to convert $.inType|raw$ to $.outType|raw$\n", args)
				sw.Do("compileErrorOnMissingConversion()\n", nil)
				conversionExists = false
			}
			if conversionExists {
				sw.Do("return err\n", nil)
				sw.Do("}\n", nil)
			}
		}
	}
}

func (g *genConversion) isFastConversion(inType, outType *types.Type) bool {
	switch inType.Kind {
	case types.Builtin:
		return true
	case types.Map, types.Slice, types.Pointer, types.Struct, types.Alias:
		return g.isDirectlyAssignable(inType, outType)
	default:
		return false
	}
}

func (g *genConversion) isDirectlyAssignable(inType, outType *types.Type) bool {
	return unwrapAlias(inType) == unwrapAlias(outType)
}

func (g *genConversion) doPointer(inType, outType *types.Type, sw *generator.SnippetWriter) {
	sw.Do("*out = new($.Elem|raw$)\n", outType)
	if isDirectlyAssignable(inType.Elem, outType.Elem) {
		if inType.Elem == outType.Elem {
			sw.Do("**out = **in\n", nil)
		} else {
			sw.Do("**out = $.|raw$(**in)\n", outType.Elem)
		}
	} else {
		conversionExists := true
		if function, ok := g.preexists(inType.Elem, outType.Elem); ok {
			sw.Do("if err := $.|raw$(*in, *out, s); err != nil {\n", function)
		} else if g.convertibleOnlyWithinPackage(inType.Elem, outType.Elem) {
			sw.Do("if err := "+nameTmpl+"(*in, *out, s); err != nil {\n", argsFromType(inType.Elem, outType.Elem))
		} else {
			args := argsFromType(inType.Elem, outType.Elem)
			sw.Do("// FIXME: Provide conversion function to convert $.inType|raw$ to $.outType|raw$\n", args)
			sw.Do("compileErrorOnMissingConversion()\n", nil)
			conversionExists = false
		}
		if conversionExists {
			sw.Do("return err\n", nil)
			sw.Do("}\n", nil)
		}
	}
}

func (g *genConversion) doAlias(inType, outType *types.Type, sw *generator.SnippetWriter) {
	// TODO: Add support for aliases.
	g.doUnknown(inType, outType, sw)
}

func (g *genConversion) doUnknown(inType, outType *types.Type, sw *generator.SnippetWriter) {
	sw.Do("// FIXME: Type $.|raw$ is unsupported.\n", inType)
}

func (g *genConversion) generateFromURLValues(inType, outType *types.Type, sw *generator.SnippetWriter) {
	args := generator.Args{
		"inType":  inType,
		"outType": outType,
		"Scope":   types.Ref(conversionPackagePath, "Scope"),
	}
	sw.Do("func auto"+nameTmpl+"(in *$.inType|raw$, out *$.outType|raw$, s $.Scope|raw$) error {\n", args)
	for _, outMember := range outType.Members {
		tagvals, err := extractTag(outMember.CommentLines)
		if err != nil {
			klog.Errorf("Member %v.%v: error extracting tags: %v", outType, outMember.Name, err)
		}
		if tagvals != nil && tagvals[0] == "false" {
			// This field is excluded from conversion.
			sw.Do("// INFO: in."+outMember.Name+" opted out of conversion generation\n", nil)
			continue
		}
		jsonTag := reflect.StructTag(outMember.Tags).Get("json")
		index := strings.Index(jsonTag, ",")
		if index == -1 {
			index = len(jsonTag)
		}
		if index == 0 {
			memberArgs := generator.Args{
				"name": outMember.Name,
			}
			sw.Do("// WARNING: Field $.name$ does not have json tag, skipping.\n\n", memberArgs)
			continue
		}
		memberArgs := generator.Args{
			"name": outMember.Name,
			"tag":  jsonTag[:index],
		}
		sw.Do("if values, ok := map[string][]string(*in)[\"$.tag$\"]; ok && len(values) > 0 {\n", memberArgs)
		g.fromValuesEntry(inType.Underlying.Elem, outMember, sw)
		sw.Do("} else {\n", nil)
		g.setZeroValue(outMember, sw)
		sw.Do("}\n", nil)
	}
	sw.Do("return nil\n", nil)
	sw.Do("}\n\n", nil)

	if _, found := g.preexists(inType, outType); found {
		// There is a public manual Conversion method: use it.
	} else {
		// Emit a public conversion function.
		sw.Do("// "+nameTmpl+" is an autogenerated conversion function.\n", args)
		sw.Do("func "+nameTmpl+"(in *$.inType|raw$, out *$.outType|raw$, s $.Scope|raw$) error {\n", args)
		sw.Do("return auto"+nameTmpl+"(in, out, s)\n", args)
		sw.Do("}\n\n", nil)
	}
}

func (g *genConversion) fromValuesEntry(inType *types.Type, outMember types.Member, sw *generator.SnippetWriter) {
	memberArgs := generator.Args{
		"name": outMember.Name,
		"type": outMember.Type,
	}
	if function, ok := g.preexists(inType, outMember.Type); ok {
		args := memberArgs.With("function", function)
		sw.Do("if err := $.function|raw$(&values, &out.$.name$, s); err != nil {\n", args)
		sw.Do("return err\n", nil)
		sw.Do("}\n", nil)
		return
	}
	switch {
	case outMember.Type == types.String:
		sw.Do("out.$.name$ = values[0]\n", memberArgs)
	case g.useUnsafe.Equal(inType, outMember.Type):
		args := memberArgs.With("Pointer", types.Ref("unsafe", "Pointer"))
		switch inType.Kind {
		case types.Pointer:
			sw.Do("out.$.name$ = ($.type|raw$)($.Pointer|raw$(&values))\n", args)
		case types.Map, types.Slice:
			sw.Do("out.$.name$ = *(*$.type|raw$)($.Pointer|raw$(&values))\n", args)
		default:
			// TODO: Support other types to allow more auto-conversions.
			sw.Do("// FIXME: out.$.name$ is of not yet supported type and requires manual conversion\n", memberArgs)
		}
	default:
		// TODO: Support other types to allow more auto-conversions.
		sw.Do("// FIXME: out.$.name$ is of not yet supported type and requires manual conversion\n", memberArgs)
	}
}

func (g *genConversion) setZeroValue(outMember types.Member, sw *generator.SnippetWriter) {
	outMemberType := unwrapAlias(outMember.Type)
	memberArgs := generator.Args{
		"name":  outMember.Name,
		"alias": outMember.Type,
		"type":  outMemberType,
	}

	switch outMemberType.Kind {
	case types.Builtin:
		switch outMemberType {
		case types.String:
			sw.Do("out.$.name$ = \"\"\n", memberArgs)
		case types.Int64, types.Int32, types.Int16, types.Int, types.Uint64, types.Uint32, types.Uint16, types.Uint:
			sw.Do("out.$.name$ = 0\n", memberArgs)
		case types.Uintptr, types.Byte:
			sw.Do("out.$.name$ = 0\n", memberArgs)
		case types.Float64, types.Float32, types.Float:
			sw.Do("out.$.name$ = 0\n", memberArgs)
		case types.Bool:
			sw.Do("out.$.name$ = false\n", memberArgs)
		default:
			sw.Do("// FIXME: out.$.name$ is of unsupported type and requires manual conversion\n", memberArgs)
		}
	case types.Struct:
		if outMemberType == outMember.Type {
			sw.Do("out.$.name$ = $.type|raw${}\n", memberArgs)
		} else {
			sw.Do("out.$.name$ = $.alias|raw$($.type|raw${})\n", memberArgs)
		}
	case types.Map, types.Slice, types.Pointer:
		sw.Do("out.$.name$ = nil\n", memberArgs)
	case types.Alias:
		// outMemberType was already unwrapped from aliases - so that should never happen.
		sw.Do("// FIXME: unexpected error for out.$.name$\n", memberArgs)
	case types.Interface, types.Array:
		sw.Do("out.$.name$ = nil\n", memberArgs)
	default:
		sw.Do("// FIXME: out.$.name$ is of unsupported type and requires manual conversion\n", memberArgs)
	}
}

func isDirectlyAssignable(inType, outType *types.Type) bool {
	// TODO: This should maybe check for actual assignability between the two
	// types, rather than superficial traits that happen to indicate it is
	// assignable in the ways we currently use this code.
	return inType.IsAssignable() && (inType.IsPrimitive() || isSamePackage(inType, outType))
}

func isSamePackage(inType, outType *types.Type) bool {
	return inType.Name.Package == outType.Name.Package
}
