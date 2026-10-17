/*
Copyright The Kubernetes Authors.

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

package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/constant"
	"go/format"
	"go/token"
	"go/types"
	"maps"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"unicode"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/ssa"
)

// Reporter abstracts error reporting for both CLI and testing.TB callers.
type Reporter interface {
	Errorf(format string, args ...any)
	Fatalf(format string, args ...any)
}

type cliReporter struct {
	failed bool
}

func (r *cliReporter) Errorf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "ERROR: "+format+"\n", args...)
	r.failed = true
}

func (r *cliReporter) Fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "FATAL: "+format+"\n", args...)
	os.Exit(1)
}

func main() {
	kubeRoot := flag.String("kube-root", filepath.Join("..", ".."), "path to kubernetes repository root")
	verifyOnly := flag.Bool("verify-only", false, "verify generated files without writing")
	flag.Parse()

	r := &cliReporter{}
	for _, cfg := range DefaultConfigs(*kubeRoot) {
		Verify(r, cfg, !*verifyOnly)
	}
	if r.failed {
		os.Exit(1)
	}
}

// DefaultConfigs returns the informer trimming codegen configurations for kube-apiserver and kube-controller-manager.
func DefaultConfigs(kubeRoot string) []Config {
	return []Config{
		{
			KubeRoot:         kubeRoot,
			BinaryPkg:        "k8s.io/kubernetes/cmd/kube-apiserver",
			TargetPkgName:    "apiserver",
			GeneratorCmd:     "hack/update-codegen.sh informertrim",
			InformerIface:    "k8s.io/client-go/informers/core/v1.Interface",
			InformerMethod:   "Pods",
			InformerTypes:    []string{"k8s.io/client-go/informers/core/v1.PodInformer", "k8s.io/client-go/listers/core/v1.PodLister", "k8s.io/client-go/listers/core/v1.PodNamespaceLister"},
			WriteClientIface: "k8s.io/client-go/kubernetes/typed/core/v1.PodInterface",
			RootType:         "k8s.io/api/core/v1.Pod",
			GeneratedFile:    filepath.Join(kubeRoot, "pkg", "controlplane", "apiserver", "zz_generated.pod_trim.go"),
		},
		{
			KubeRoot:         kubeRoot,
			BinaryPkg:        "k8s.io/kubernetes/cmd/kube-controller-manager",
			TargetPkgName:    "app",
			GeneratorCmd:     "hack/update-codegen.sh informertrim",
			InformerIface:    "k8s.io/client-go/informers/core/v1.Interface",
			InformerMethod:   "Pods",
			InformerTypes:    []string{"k8s.io/client-go/informers/core/v1.PodInformer", "k8s.io/client-go/listers/core/v1.PodLister", "k8s.io/client-go/listers/core/v1.PodNamespaceLister"},
			WriteClientIface: "k8s.io/client-go/kubernetes/typed/core/v1.PodInterface",
			RootType:         "k8s.io/api/core/v1.Pod",
			GeneratedFile:    filepath.Join(kubeRoot, "cmd", "kube-controller-manager", "app", "zz_generated.pod_trim.go"),
		},
	}
}

// Config specifies the target binary, informer wiring, and root API type for
// interprocedural Go SSA informer field-trimming code generation.
type Config struct {
	KubeRoot         string   // optional path to kubernetes repo root
	BinaryPkg        string   // e.g. "k8s.io/kubernetes/cmd/kube-apiserver"
	TargetPkgName    string   // e.g. "apiserver"
	GeneratorCmd     string   // command shown in generated file header
	InformerIface    string   // e.g. "k8s.io/client-go/informers/core/v1.Interface"
	InformerMethod   string   // informer method name on InformerIface
	InformerTypes    []string // informer and lister interface type keys
	WriteClientIface string   // typed write client interface key
	RootType         string   // fully-qualified root struct type key
	GeneratedFile    string   // output file path
}

type ssaProvenance struct {
	Root, Path             string
	OrigStruct, TargetType string
	Fn                     *ssa.Function
	Free                   map[int]ssaProvenance
}

type fieldAddrKey struct {
	base  ssa.Value
	field int
}

const objectMetaTypeKey = "k8s.io/apimachinery/pkg/apis/meta/v1.ObjectMeta"

var metav1ObjectGetters = map[string]string{
	"GetName": "Name", "GetGenerateName": "GenerateName", "GetNamespace": "Namespace",
	"GetUID": "UID", "GetResourceVersion": "ResourceVersion", "GetGeneration": "Generation",
	"GetSelfLink": "SelfLink", "GetCreationTimestamp": "CreationTimestamp",
	"GetDeletionTimestamp": "DeletionTimestamp", "GetDeletionGracePeriodSeconds": "DeletionGracePeriodSeconds",
	"GetLabels": "Labels", "GetAnnotations": "Annotations", "GetFinalizers": "Finalizers",
	"GetOwnerReferences": "OwnerReferences", "GetManagedFields": "ManagedFields",
}

// Verify runs interprocedural Go SSA analysis for cfg, optionally updates cfg.GeneratedFile,
// and asserts that cfg.GeneratedFile matches the SSA-generated source code.
func Verify(t Reporter, cfg Config, update bool) {
	if h, ok := t.(interface{ Helper() }); ok {
		h.Helper()
	}
	generated := Generate(t, cfg)
	if update {
		if err := os.WriteFile(cfg.GeneratedFile, generated, 0644); err != nil {
			t.Fatalf("failed to write %s: %v", cfg.GeneratedFile, err)
		}
	}
	existing, err := os.ReadFile(cfg.GeneratedFile)
	if err != nil {
		t.Fatalf("failed to read %s: %v", cfg.GeneratedFile, err)
	}
	if !bytes.Equal(generated, existing) {
		t.Fatalf("%s is out of date with Go SSA analysis; run '%s'", cfg.GeneratedFile, cfg.GeneratorCmd)
	}
}

// Generate builds the generic-instantiated Go SSA program for cfg.BinaryPkg on demand, discovers
// all wired informer/lister consumers for cfg.RootType, traces every struct field read across the
// call graph, and emits formatted Go source code for cfg.GeneratedFile.
func Generate(t Reporter, cfg Config) []byte {
	if h, ok := t.(interface{ Helper() }); ok {
		h.Helper()
	}
	pkgCfg := &packages.Config{
		Dir: cfg.KubeRoot,
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles |
			packages.NeedImports | packages.NeedDeps | packages.NeedTypes |
			packages.NeedSyntax | packages.NeedTypesInfo | packages.NeedTypesSizes,
	}
	rootPkgs, err := packages.Load(pkgCfg, cfg.BinaryPkg)
	if err != nil || packages.PrintErrors(rootPkgs) > 0 {
		t.Fatalf("packages.Load(%s) failed: %v", cfg.BinaryPkg, err)
	}

	prog := ssa.NewProgram(rootPkgs[0].Fset, ssa.InstantiateGenerics|ssa.BareInits)
	rawPkgFuncs := make(map[*ssa.Package][]*ssa.Function)
	builtPkgFuncs := make(map[*ssa.Package][]*ssa.Function)
	ifaceImpls := make(map[string][]*ssa.Function)
	methodCallersByName := make(map[string][]*ssa.Package)
	var informerCandidatePkgs []*ssa.Package
	seenCandidatePkg := make(map[*ssa.Package]bool)
	pkgCallDeps := make(map[*types.Package][]*types.Package)

	packages.Visit(rootPkgs, nil, func(p *packages.Package) {
		if p.Types == nil || p.IllTyped {
			return
		}
		if !isCandidateK8sPkg(p.PkgPath) {
			prog.CreatePackage(p.Types, nil, nil, true)
			p.Syntax = nil
			p.TypesInfo = nil
			return
		}
		ssaPkg := prog.CreatePackage(p.Types, p.Syntax, p.TypesInfo, true)
		for _, obj := range p.TypesInfo.Defs {
			if tf, ok := obj.(*types.Func); ok {
				if fn := prog.FuncValue(tf); fn != nil && !shouldSkipSSAFunction(fn) {
					rawPkgFuncs[ssaPkg] = append(rawPkgFuncs[ssaPkg], fn)
					if fn.Signature != nil && fn.Signature.Recv() != nil && !strings.Contains(strings.ToLower(fn.String()), "fake") {
						ifaceImpls[fn.Name()] = append(ifaceImpls[fn.Name()], fn)
					}
				}
			}
		}
		seenMethod := make(map[string]bool)
		seenDepPkg := make(map[*types.Package]bool)
		for _, obj := range p.TypesInfo.Uses {
			if obj == nil {
				continue
			}
			if obj.Name() == "InformerFor" || obj.Name() == "ForResource" || obj.Name() == cfg.InformerMethod {
				if !seenCandidatePkg[ssaPkg] {
					seenCandidatePkg[ssaPkg] = true
					informerCandidatePkgs = append(informerCandidatePkgs, ssaPkg)
				}
			}
			if obj.Pkg() != nil {
				op := obj.Pkg().Path()
				if strings.HasPrefix(op, "k8s.io/client-go/informers") || strings.HasPrefix(op, "k8s.io/client-go/listers") || op == "k8s.io/controller-manager/pkg/informerfactory" {
					if !seenCandidatePkg[ssaPkg] {
						seenCandidatePkg[ssaPkg] = true
						informerCandidatePkgs = append(informerCandidatePkgs, ssaPkg)
					}
				}
				switch obj.(type) {
				case *types.Func, *types.Var:
					if obj.Pkg() != p.Types && !seenDepPkg[obj.Pkg()] {
						seenDepPkg[obj.Pkg()] = true
						pkgCallDeps[p.Types] = append(pkgCallDeps[p.Types], obj.Pkg())
					}
				}
			}
			if tf, ok := obj.(*types.Func); ok {
				if sig, ok := tf.Type().(*types.Signature); ok && sig.Recv() != nil && !seenMethod[tf.Name()] {
					seenMethod[tf.Name()] = true
					methodCallersByName[tf.Name()] = append(methodCallersByName[tf.Name()], ssaPkg)
				}
			}
		}
		p.Syntax = nil
		p.TypesInfo = nil
	})

	reachableTypesPkg := map[*types.Package]bool{rootPkgs[0].Types: true}
	for queue := []*types.Package{rootPkgs[0].Types}; len(queue) > 0; queue = queue[1:] {
		for _, dep := range pkgCallDeps[queue[0]] {
			if !reachableTypesPkg[dep] {
				reachableTypesPkg[dep] = true
				queue = append(queue, dep)
			}
		}
	}
	filteredCandidates := informerCandidatePkgs[:0]
	for _, ssaPkg := range informerCandidatePkgs {
		if reachableTypesPkg[ssaPkg.Pkg] {
			filteredCandidates = append(filteredCandidates, ssaPkg)
		}
	}
	informerCandidatePkgs = filteredCandidates

	ensurePkgFuncs := func(ssaPkg *ssa.Package) []*ssa.Function {
		if ssaPkg == nil {
			return nil
		}
		if fns, ok := builtPkgFuncs[ssaPkg]; ok {
			return fns
		}
		ssaPkg.Build()
		var all []*ssa.Function
		seenFn := make(map[*ssa.Function]bool)
		var visitFn func(fn *ssa.Function)
		visitFn = func(fn *ssa.Function) {
			if fn == nil || seenFn[fn] || shouldSkipSSAFunction(fn) {
				return
			}
			seenFn[fn] = true
			if len(fn.Blocks) > 0 && fnPkg(fn) == ssaPkg && fn.Name() != "init" {
				all = append(all, fn)
			}
			for _, anon := range fn.AnonFuncs {
				visitFn(anon)
			}
			var ops [10]*ssa.Value
			for _, b := range fn.Blocks {
				for _, instr := range b.Instrs {
					if mi, ok := instr.(*ssa.MakeInterface); ok && !types.IsInterface(mi.X.Type()) {
						mset := prog.MethodSets.MethodSet(mi.X.Type())
						for sel := range mset.Methods() {
							if sel.Obj().(*types.Func).Signature().TypeParams() == nil {
								if mFn := prog.MethodValue(sel); mFn != nil && fnPkg(mFn) == ssaPkg {
									visitFn(mFn)
								}
							}
						}
					}
					for _, op := range instr.Operands(ops[:0]) {
						if opFn, ok := (*op).(*ssa.Function); ok && fnPkg(opFn) == ssaPkg {
							visitFn(opFn)
						}
					}
				}
			}
		}
		if initFn := ssaPkg.Func("init"); initFn != nil {
			visitFn(initFn)
		}
		for _, fn := range rawPkgFuncs[ssaPkg] {
			visitFn(fn)
		}
		builtPkgFuncs[ssaPkg] = all
		return all
	}

	lastDot := strings.LastIndexByte(cfg.RootType, '.')
	rootPkgPath, rootTypeName := cfg.RootType[:lastDot], cfg.RootType[lastDot+1:]
	rootPkg := prog.ImportedPackage(rootPkgPath).Pkg
	metav1Pkg := prog.ImportedPackage("k8s.io/apimachinery/pkg/apis/meta/v1").Pkg
	rootNamed := rootPkg.Scope().Lookup(rootTypeName).Type().(*types.Named)

	// Recursively discover all reachable structs in rootPkgPath plus metav1.ObjectMeta.
	trackedTypes := make(map[string]*types.Named)
	var discoverStructs func(t types.Type)
	discoverStructs = func(t types.Type) {
		named, ok := unwrapType(t).(*types.Named)
		if !ok || named.Obj() == nil || named.Obj().Pkg() == nil {
			return
		}
		st, ok := named.Underlying().(*types.Struct)
		if !ok {
			return
		}
		key := named.Obj().Pkg().Path() + "." + named.Obj().Name()
		if trackedTypes[key] != nil {
			return
		}
		if named.Obj().Pkg().Path() != rootPkgPath && key != objectMetaTypeKey {
			return
		}
		trackedTypes[key] = named
		for i := 0; i < st.NumFields(); i++ {
			if st.Field(i).Exported() {
				discoverStructs(st.Field(i).Type())
			}
		}
	}
	discoverStructs(rootNamed)

	informerTypeSet := make(map[string]bool, len(cfg.InformerTypes))
	for _, it := range cfg.InformerTypes {
		informerTypeSet[it] = true
	}
	resourcePlural := strings.ToLower(cfg.InformerMethod)
	hasInformerParamOrField := func(sig *types.Signature) bool {
		if sig == nil {
			return false
		}
		for i := 0; i < sig.Params().Len(); i++ {
			pt := sig.Params().At(i).Type()
			if informerTypeSet[namedTypeKey(pt)] {
				return true
			}
			if st, ok := unwrapType(pt).Underlying().(*types.Struct); ok {
				for j := 0; j < st.NumFields(); j++ {
					if st.Field(j).Exported() && informerTypeSet[namedTypeKey(st.Field(j).Type())] {
						return true
					}
				}
			}
		}
		return false
	}

	// Phase 1: Discover packages wired with cfg.InformerIface.<InformerMethod>() informers/listers.
	wiredPkgs := make(map[*ssa.Package]bool)
	var wiredPkgList []*ssa.Package
	addWiredPkg := func(p *ssa.Package) {
		if p != nil && !wiredPkgs[p] {
			wiredPkgs[p] = true
			wiredPkgList = append(wiredPkgList, p)
		}
	}
	for _, ssaPkg := range informerCandidatePkgs {
		for _, fn := range ensurePkgFuncs(ssaPkg) {
			forEachInstr(fn, func(instr ssa.Instruction) {
				if call, ok := instr.(*ssa.Call); ok && call.Common().IsInvoke() {
					cc := call.Common()
					recvKey, mName := namedTypeKey(cc.Value.Type()), cc.Method.Name()
					if recvKey == "k8s.io/client-go/informers.SharedInformerFactory" {
						if mName == "InformerFor" && len(cc.Args) > 0 && !strings.HasPrefix(fn.Name(), "InformerFor") {
							if mi, ok := cc.Args[0].(*ssa.MakeInterface); ok {
								if namedTypeKey(mi.X.Type()) == cfg.RootType {
									t.Errorf("unexpected SharedInformerFactory.InformerFor(%s) in %s", cfg.RootType, fn)
								}
							} else {
								t.Errorf("non-static SharedInformerFactory.InformerFor(%s) in %s", cc.Args[0], fn)
							}
						} else if mName == "ForResource" && len(cc.Args) > 0 && !strings.HasPrefix(fn.Name(), "ForResource") {
							if !hasNonTargetResourceGuard(fn, cc.Args[0], resourcePlural) {
								t.Errorf("unapproved SharedInformerFactory.ForResource call in %s (must guard against %q)", fn, resourcePlural)
							}
						}
					} else if (recvKey == cfg.InformerIface && mName == cfg.InformerMethod) ||
						(recvKey == "k8s.io/controller-manager/pkg/informerfactory.InformerFactory" && mName == "ForResource" && !strings.HasPrefix(fn.Name(), "ForResource")) {
						addWiredPkg(ssaPkg)
					}
				}
			})
		}
	}
	for range 4 {
		for _, ssaPkg := range wiredPkgList {
			pkgPath := ssaPkg.Pkg.Path()
			isCtrlPkg := strings.HasPrefix(pkgPath, "k8s.io/kubernetes/pkg/controller/")
			for _, fn := range ensurePkgFuncs(ssaPkg) {
				forEachInstr(fn, func(instr ssa.Instruction) {
					if call, ok := instr.(*ssa.Call); ok {
						if callee := call.Common().StaticCallee(); callee != nil {
							calleePkg := fnPkg(callee)
							calleePkgPath := fnPkgPath(callee)
							if hasInformerParamOrField(callee.Signature) ||
								(isCtrlPkg && (strings.HasPrefix(calleePkgPath, pkgPath+"/") || (pkgPath == "k8s.io/kubernetes/pkg/controller/replication" && calleePkgPath == "k8s.io/kubernetes/pkg/controller/replicaset"))) {
								addWiredPkg(calleePkg)
							}
						}
					}
				})
			}
		}
	}

	// Phase 2: Collect informer event handlers, indexers, tainted struct fields, and lister callers in wiredPkgs.
	type consumerEntry struct {
		fn        *ssa.Function
		paramProv map[int]ssaProvenance
	}
	var entries []consumerEntry
	seenEntry := make(map[string]bool)
	addSeed := func(fn *ssa.Function, paramIdxs ...int) {
		if fn != nil && !shouldSkipSSAFunction(fn) {
			if p := fnPkg(fn); p != nil {
				ensurePkgFuncs(p)
			}
			if len(fn.Blocks) > 0 {
				pm := make(map[int]ssaProvenance, len(paramIdxs))
				for _, idx := range paramIdxs {
					if idx < len(fn.Params) {
						pm[idx] = ssaProvenance{Root: cfg.RootType}
					}
				}
				key := fn.String() + "#" + encodeSSAParamProv(pm)
				if !seenEntry[key] {
					seenEntry[key] = true
					entries = append(entries, consumerEntry{fn: fn, paramProv: pm})
				}
			}
		}
	}
	taintedIfaceMethods := make(map[string]bool)
	taintedStructFields := make(map[string]bool)
	for _, ssaPkg := range wiredPkgList {
		for _, fn := range ensurePkgFuncs(ssaPkg) {
			informerVals := make(map[ssa.Value]bool)
			for _, p := range fn.Params {
				if informerTypeSet[namedTypeKey(p.Type())] {
					informerVals[p] = true
				}
			}
			for range 2 {
				forEachInstr(fn, func(instr ssa.Instruction) {
					switch ins := instr.(type) {
					case *ssa.FieldAddr:
						if informerTypeSet[namedTypeKey(ins.Type())] {
							informerVals[ins] = true
						}
					case *ssa.UnOp:
						if ins.Op == token.MUL && informerVals[ins.X] {
							informerVals[ins] = true
						}
					case *ssa.Store:
						if fa, ok := ins.Addr.(*ssa.FieldAddr); ok {
							sfKey := structFieldKey(fa)
							if sfKey != "" {
								if informerVals[ins.Val] {
									taintedStructFields[sfKey] = true
								} else if sig, ok := ins.Val.Type().Underlying().(*types.Signature); ok && calleeCanReturnRootType(sig, cfg.RootType) {
									taintedStructFields[sfKey] = true
									if cb := resolveSSAClosure(ins.Val, nil); cb.Fn != nil {
										addSeed(cb.Fn)
									}
								}
							}
						}
					case *ssa.Call:
						cc := ins.Common()
						if cc.IsInvoke() {
							recvKey, m := namedTypeKey(cc.Value.Type()), cc.Method.Name()
							if (m == "Informer" || m == "Lister") && (informerTypeSet[recvKey] || informerVals[cc.Value]) {
								informerVals[ins] = true
							} else if m == "ForResource" && recvKey == "k8s.io/controller-manager/pkg/informerfactory.InformerFactory" && !strings.HasPrefix(fn.Name(), "ForResource") {
								informerVals[ins] = true
							} else if (m == "GetIndexer" || m == "GetStore") && informerVals[cc.Value] {
								informerVals[ins] = true
							} else if (m == "AddEventHandler" || m == "AddEventHandlerWithResyncPeriod" || m == "AddEventHandlerWithOptions") && informerVals[cc.Value] && len(cc.Args) > 0 {
								if extractEventHandlerFuncs(prog, cc.Args[0], addSeed) == 0 {
									t.Errorf("unrecognized event handler registration in %s: could not statically extract handler funcs", fn)
								}
							} else if m == "AddIndexers" && informerVals[cc.Value] && len(cc.Args) > 0 {
								extractIndexerFuncsFromFn(fn, addSeed)
							}
						} else if callee := cc.StaticCallee(); callee != nil {
							for _, arg := range cc.Args {
								if informerVals[arg] {
									if cPkg := fnPkg(callee); cPkg != nil {
										for _, pFn := range ensurePkgFuncs(cPkg) {
											extractIndexerFuncsFromFn(pFn, addSeed)
										}
									}
								}
							}
						}
					}
				})
			}
		}
	}
	for _, ssaPkg := range wiredPkgList {
		for _, fn := range ensurePkgFuncs(ssaPkg) {
			shouldSeed := false
			callsLister := false
			forEachInstr(fn, func(instr ssa.Instruction) {
				switch ins := instr.(type) {
				case *ssa.FieldAddr:
					if taintedStructFields[structFieldKey(ins)] {
						shouldSeed = true
					}
				case *ssa.Call:
					cc := ins.Common()
					if cc.IsInvoke() {
						recvKey, m := namedTypeKey(cc.Value.Type()), cc.Method.Name()
						if informerTypeSet[recvKey] && (m == "List" || m == "Get") {
							callsLister = true
							shouldSeed = true
						} else if calleeCanReturnRootType(cc.Method.Type().(*types.Signature), cfg.RootType) {
							shouldSeed = true
						}
					} else if callee := cc.StaticCallee(); callee != nil && calleeCanReturnRootType(callee.Signature, cfg.RootType) {
						shouldSeed = true
					}
				}
			})
			if shouldSeed {
				addSeed(fn)
			}
			if callsLister && fn.Signature != nil && fn.Signature.Recv() != nil && calleeCanReturnRootType(fn.Signature, cfg.RootType) {
				recvType := fn.Signature.Recv().Type()
				for _, callerPkg := range methodCallersByName[fn.Name()] {
					for _, caller := range ensurePkgFuncs(callerPkg) {
						forEachInstr(caller, func(instr ssa.Instruction) {
							if call, ok := instr.(*ssa.Call); ok && call.Common().IsInvoke() && call.Common().Method.Name() == fn.Name() {
								if iface, ok := unwrapType(call.Common().Value.Type()).Underlying().(*types.Interface); ok && types.Implements(recvType, iface) {
									taintedIfaceMethods[namedTypeKey(call.Common().Value.Type())+"."+fn.Name()] = true
									addSeed(caller)
								}
							}
						})
					}
				}
			}
		}
	}
	if len(entries) == 0 {
		t.Fatalf("failed to discover any %s consumer entrypoints", cfg.RootType)
	}

	// Phase 3: Interprocedural SSA value-flow analysis per consumer package.
	observed := make(map[string]map[string]map[string]bool, len(trackedTypes))
	for tp := range trackedTypes {
		observed[tp] = make(map[string]map[string]bool)
	}
	if trackedTypes[objectMetaTypeKey] != nil {
		observed[cfg.RootType]["ObjectMeta"] = map[string]bool{"informer": true}
		observed[objectMetaTypeKey]["ResourceVersion"] = map[string]bool{"informer": true}
	}

	var activeConsumer string
	recordField := func(structType, fieldName string) {
		if fields, ok := observed[structType]; ok && activeConsumer != "" {
			if fields[fieldName] == nil {
				fields[fieldName] = make(map[string]bool)
			}
			fields[fieldName][activeConsumer] = true
		}
	}
	var recordAllStructFields func(structType string)
	recordAllStructFields = func(structType string) {
		named := trackedTypes[structType]
		if named == nil {
			return
		}
		st := named.Underlying().(*types.Struct)
		for i := 0; i < st.NumFields(); i++ {
			if st.Field(i).Exported() {
				recordField(structType, st.Field(i).Name())
			}
		}
	}
	seenRecursiveStruct := make(map[string]bool)
	var recordAllStructFieldsRecursive func(t types.Type)
	recordAllStructFieldsRecursive = func(t types.Type) {
		key := namedTypeKey(t)
		named := trackedTypes[key]
		if named == nil || seenRecursiveStruct[activeConsumer+"#"+key] {
			return
		}
		seenRecursiveStruct[activeConsumer+"#"+key] = true
		st := named.Underlying().(*types.Struct)
		for i := 0; i < st.NumFields(); i++ {
			if st.Field(i).Exported() {
				recordField(key, st.Field(i).Name())
				recordAllStructFieldsRecursive(st.Field(i).Type())
			}
		}
	}

	type visitKey struct {
		fn  *ssa.Function
		sig string
	}
	type visitResult struct {
		ret   []ssaProvenance
		depth int
	}
	visitedByConsumer := make(map[string]map[visitKey]visitResult)
	for _, entry := range entries {
		entryPkg := fnPkg(entry.fn)
		if entryPkg == nil || entryPkg.Pkg == nil {
			continue
		}
		activeConsumer = consumerPkgShortName(entryPkg.Pkg.Path())
		visited := visitedByConsumer[activeConsumer]
		if visited == nil {
			visited = make(map[visitKey]visitResult)
			visitedByConsumer[activeConsumer] = visited
		}

		var analyzeFn func(fn *ssa.Function, paramProv map[int]ssaProvenance, depth int) []ssaProvenance
		analyzeFn = func(fn *ssa.Function, paramProv map[int]ssaProvenance, depth int) []ssaProvenance {
			if fn == nil || depth > 20 || shouldSkipSSAFunction(fn) {
				return nil
			}
			if p := fnPkg(fn); p != nil {
				ensurePkgFuncs(p)
			}
			if len(fn.Blocks) == 0 {
				return nil
			}
			key := visitKey{fn: fn, sig: encodeSSAParamProv(paramProv)}
			if prev, ok := visited[key]; ok && depth >= prev.depth {
				return prev.ret
			}
			visited[key] = visitResult{depth: depth}

			valProv := make(map[ssa.Value]ssaProvenance)
			fieldProv := make(map[fieldAddrKey]ssaProvenance)
			complitAllocs := make(map[ssa.Value]bool)
			for i, p := range fn.Params {
				if prov, ok := paramProv[i]; ok {
					valProv[p] = prov
				}
			}
			for i, fv := range fn.FreeVars {
				if prov, ok := paramProv[-(i + 1)]; ok {
					valProv[fv] = prov
				}
			}

			var returnProvs []ssaProvenance
			isFinalPass := false
			for pass := range 8 {
				prevCount := len(valProv) + len(fieldProv)
				taintCallResult := func(callVal *ssa.Call, p ssaProvenance) {
					if callVal == nil {
						return
					}
					if _, isTuple := callVal.Type().(*types.Tuple); !isTuple {
						valProv[callVal] = p
					}
					if callVal.Referrers() != nil {
						for _, ref := range *callVal.Referrers() {
							if ext, ok := ref.(*ssa.Extract); ok && ext.Index == 0 {
								valProv[ext] = p
							}
						}
					}
				}
				callCallee := func(cc *ssa.CallCommon, callVal *ssa.Call, callee *ssa.Function, isInvoke bool, extra map[int]ssaProvenance) {
					var childProv map[int]ssaProvenance
					setParam := func(k int, v ssaProvenance) {
						if v.Root != "" || v.Fn != nil {
							if childProv == nil {
								childProv = make(map[int]ssaProvenance)
							}
							childProv[k] = v
						}
					}
					for k, v := range extra {
						setParam(k, v)
					}
					offset := 0
					if isInvoke && callee.Signature.Recv() != nil {
						offset = 1
						setParam(0, valProv[cc.Value])
					}
					for i, arg := range cc.Args {
						if i+offset < callee.Signature.Params().Len() || i+offset < len(callee.Params) {
							p := valProv[arg]
							if p.Fn == nil {
								c := resolveSSAClosure(arg, valProv)
								p.Fn, p.Free = c.Fn, c.Free
							}
							setParam(i+offset, p)
						}
					}
					if len(childProv) == 0 {
						cPkg := fnPkg(callee)
						if !calleeCanReturnRootType(callee.Signature, cfg.RootType) || (!wiredPkgs[cPkg] && fnPkgPath(callee) != "k8s.io/kubernetes/pkg/controller") {
							return
						}
					}
					retProvs := analyzeFn(callee, childProv, depth+1)
					if callVal != nil && len(retProvs) == 1 && (retProvs[0].Root != "" || retProvs[0].Fn != nil) {
						valProv[callVal] = retProvs[0]
					} else if callVal != nil && callVal.Referrers() != nil {
						for _, ref := range *callVal.Referrers() {
							if ext, ok := ref.(*ssa.Extract); ok && ext.Index < len(retProvs) && (retProvs[ext.Index].Root != "" || retProvs[ext.Index].Fn != nil) {
								valProv[ext] = retProvs[ext.Index]
							}
						}
					}
				}
				checkGetter := func(name string, recv ssa.Value, callVal *ssa.Call) bool {
					fName, ok := metav1ObjectGetters[name]
					if !ok || valProv[recv].Root != cfg.RootType {
						return false
					}
					if isFinalPass && (callVal == nil || !isSSALoggingOnlyValue(callVal, 0)) {
						recordField(cfg.RootType, "ObjectMeta")
						recordField(objectMetaTypeKey, fName)
					}
					if callVal != nil {
						valProv[callVal] = ssaProvenance{Root: cfg.RootType, Path: "ObjectMeta." + fName}
					}
					return true
				}
				checkInterfaceBoxing := func(ins ssa.Value, x ssa.Value, opName string) {
					if p := valProv[x]; p.Root != "" {
						valProv[ins] = p
						if isFinalPass && trackedTypes[namedTypeKey(x.Type())] != nil {
							if iface, ok := ins.Type().Underlying().(*types.Interface); ok && iface.NumMethods() == 0 && fn.Name() != "trimInformerObject" && !isApprovedAnyBoxing(ins, 0) {
								t.Errorf("unapproved %s to any on tainted %s value (%s) in %s", opName, rootTypeName, x.Type(), fn)
							}
						}
					}
				}

				forEachInstr(fn, func(instr ssa.Instruction) {
					switch ins := instr.(type) {
					case *ssa.Alloc:
						if ins.Comment == "complit" && trackedTypes[namedTypeKey(ins.Type())] != nil {
							complitAllocs[ins] = true
						}
					case *ssa.Store:
						if p := valProv[ins.Val]; !complitAllocs[ins.Addr] && (p.Root != "" || p.Fn != nil) {
							if alloc, ok := ins.Addr.(*ssa.Alloc); ok && !complitAllocs[alloc] {
								valProv[alloc] = p
							} else if idxAddr, ok := ins.Addr.(*ssa.IndexAddr); ok {
								valProv[idxAddr.X] = p
							} else if fa, ok := ins.Addr.(*ssa.FieldAddr); ok && !complitAllocs[fa.X] {
								fieldProv[fieldAddrKey{base: fa.X, field: fa.Field}] = p
								valProv[fa] = p
								if p.Root != "" && trackedTypes[namedTypeKey(fa.X.Type())] == nil && trackedTypes[namedTypeKey(ins.Val.Type())] != nil {
									valProv[fa.X] = p
								} else if isFinalPass && p.Root == cfg.RootType && p.Path != "" && valProv[fa.X].Root == "" {
									recordAllStructFieldsRecursive(ins.Val.Type())
								}
							}
						}
					case *ssa.MapUpdate:
						if p := valProv[ins.Value]; p.Root != "" {
							valProv[ins.Map] = p
						}
					case *ssa.UnOp:
						if ins.Op == token.MUL && !complitAllocs[ins.X] && (valProv[ins.X].Root != "" || valProv[ins.X].Fn != nil) {
							valProv[ins] = valProv[ins.X]
						}
					case *ssa.FieldAddr:
						if complitAllocs[ins.X] {
							complitAllocs[ins] = true
						} else if taintedStructFields[structFieldKey(ins)] {
							valProv[ins] = ssaProvenance{Root: cfg.RootType}
						} else if fp, ok := fieldProv[fieldAddrKey{base: ins.X, field: ins.Field}]; ok {
							valProv[ins] = fp
						} else if p := valProv[ins.X]; p.Root != "" && trackedTypes[namedTypeKey(ins.X.Type())] == nil && trackedTypes[namedTypeKey(ins.Type())] != nil {
							valProv[ins] = p
						} else if !isStoreOnlySSAFieldAddr(ins) && !isSSALoggingOnlyValue(ins, 0) {
							propagateSSAFieldAccess(ins, ins.X, ins.Field, valProv, recordField, isFinalPass)
						}
					case *ssa.Field:
						if !isSSALoggingOnlyValue(ins, 0) {
							propagateSSAFieldAccess(ins, ins.X, ins.Field, valProv, recordField, isFinalPass)
						}
					case *ssa.ChangeType:
						if p := valProv[ins.X]; p.Root != "" || p.Fn != nil {
							if fromKey, toKey := namedTypeKey(ins.X.Type()), namedTypeKey(ins.Type()); trackedTypes[fromKey] != nil && trackedTypes[toKey] != nil && fromKey != toKey {
								p.OrigStruct, p.TargetType = fromKey, toKey
							}
							valProv[ins] = p
						}
					case *ssa.Range:
						if p := valProv[ins.X]; p.Root != "" {
							valProv[ins] = p
						}
					case *ssa.Next:
						if p := valProv[ins.Iter]; p.Root != "" {
							valProv[ins] = p
						}
					case *ssa.IndexAddr, *ssa.Index, *ssa.Lookup, *ssa.Slice, *ssa.Convert, *ssa.Phi:
						for _, op := range ins.Operands(nil) {
							if p := valProv[*op]; p.Root != "" || p.Fn != nil {
								valProv[ins.(ssa.Value)] = p
								break
							}
						}
					case *ssa.ChangeInterface:
						checkInterfaceBoxing(ins, ins.X, "ChangeInterface")
					case *ssa.TypeAssert:
						if p := valProv[ins.X]; p.Root != "" {
							ak := namedTypeKey(ins.AssertedType)
							_, isIface := ins.AssertedType.Underlying().(*types.Interface)
							if ak == cfg.RootType || ak == "k8s.io/client-go/tools/cache.DeletedFinalStateUnknown" || isIface {
								valProv[ins] = p
							}
						}
					case *ssa.Extract:
						if p := valProv[ins.Tuple]; p.Root != "" {
							if ins.Index == 0 {
								valProv[ins] = p
							} else if _, isNext := ins.Tuple.(*ssa.Next); isNext && (ins.Index == 1 || ins.Index == 2) && typeCanCarryRoot(ins.Type(), cfg.RootType) {
								valProv[ins] = ssaProvenance{Root: cfg.RootType}
							}
						}
					case *ssa.MakeInterface:
						checkInterfaceBoxing(ins, ins.X, "MakeInterface")
					case *ssa.MakeClosure:
						if cb := resolveSSAClosure(ins, valProv); cb.Fn != nil {
							valProv[ins] = cb
							if isFinalPass && len(cb.Free) > 0 {
								analyzeFn(cb.Fn, cb.Free, depth+1)
							}
						}
					case ssa.CallInstruction:
						cc, callVal := ins.Common(), ins.Value()
						if b, ok := cc.Value.(*ssa.Builtin); ok {
							if b.Name() == "append" || b.Name() == "copy" {
								for _, arg := range cc.Args {
									if p := valProv[arg]; p.Root != "" && callVal != nil {
										valProv[callVal] = p
										break
									}
								}
							}
							return
						}
						if cc.IsInvoke() {
							mName, recvKey := cc.Method.Name(), namedTypeKey(cc.Value.Type())
							if cfg.WriteClientIface != "" && recvKey == cfg.WriteClientIface && (mName == "Update" || mName == "UpdateStatus" || mName == "Create") {
								for _, arg := range cc.Args {
									if valProv[arg].Root == cfg.RootType && namedTypeKey(arg.Type()) == cfg.RootType && isFinalPass {
										t.Errorf("tainted %s from informer cache passed to %s.%s in %s", rootTypeName, cfg.WriteClientIface, mName, fn)
									}
								}
							}
							if mName == "DeepCopy" || mName == "DeepCopyObject" {
								if p := valProv[cc.Value]; p.Root != "" && callVal != nil {
									valProv[callVal] = p
								}
								return
							}
							if (informerTypeSet[recvKey] && (mName == "List" || mName == "Get")) ||
								taintedIfaceMethods[recvKey+"."+mName] ||
								(valProv[cc.Value].Root == cfg.RootType && (mName == "ByIndex" || mName == "Index" || mName == "List" || mName == "Get" || mName == "GetByKey")) {
								taintCallResult(callVal, ssaProvenance{Root: cfg.RootType})
								return
							}
							if checkGetter(mName, cc.Value, callVal) {
								return
							}
							if iface, ok := unwrapType(cc.Value.Type()).Underlying().(*types.Interface); ok {
								for _, target := range ifaceImpls[mName] {
									if target.Signature.Recv() != nil && (types.Implements(target.Signature.Recv().Type(), iface) || types.Implements(types.NewPointer(target.Signature.Recv().Type()), iface)) {
										callCallee(cc, callVal, target, true, nil)
									}
								}
							}
							return
						}
						if callee := cc.StaticCallee(); callee != nil {
							cName := callee.Name()
							if (cName == "DeepCopy" || cName == "DeepCopyObject") && len(cc.Args) > 0 {
								if p := valProv[cc.Args[0]]; p.Root != "" && callVal != nil {
									valProv[callVal] = p
								}
								return
							}
							if len(cc.Args) > 0 && checkGetter(cName, cc.Args[0], callVal) {
								return
							}
							if len(cc.Args) > 0 && valProv[cc.Args[0]].Root == cfg.RootType {
								switch cName {
								case "GetControllerOf", "GetControllerOfNoCopy":
									if isFinalPass {
										recordField(cfg.RootType, "ObjectMeta")
										recordField(objectMetaTypeKey, "OwnerReferences")
									}
									if callVal != nil {
										valProv[callVal] = ssaProvenance{Root: cfg.RootType, Path: "ObjectMeta.OwnerReferences"}
									}
									return
								case "MetaNamespaceKeyFunc", "DeletionHandlingMetaNamespaceKeyFunc", "KeyFunc", "NamespacedKey", "PodKey":
									if isFinalPass {
										recordField(cfg.RootType, "ObjectMeta")
										recordField(objectMetaTypeKey, "Namespace")
										recordField(objectMetaTypeKey, "Name")
									}
									return
								case "HasAnnotation":
									if isFinalPass {
										recordField(cfg.RootType, "ObjectMeta")
										recordField(objectMetaTypeKey, "Annotations")
									}
									return
								case "HasLabel":
									if isFinalPass {
										recordField(cfg.RootType, "ObjectMeta")
										recordField(objectMetaTypeKey, "Labels")
									}
									return
								}
							}
							calleePkgPath := fnPkgPath(callee)
							if calleePkgPath == "slices" {
								for _, arg := range cc.Args {
									if valProv[arg].Root != "" {
										elemKey := namedTypeKey(arg.Type())
										if strings.HasPrefix(cName, "EqualFunc") && len(cc.Args) == 3 {
											if cb := resolveSSAClosure(cc.Args[2], valProv); cb.Fn != nil {
												cp := map[int]ssaProvenance{0: valProv[arg], 1: valProv[arg]}
												for k, v := range cb.Free {
													cp[k] = v
												}
												analyzeFn(cb.Fn, cp, depth+1)
											} else if isFinalPass && trackedTypes[elemKey] != nil {
												recordAllStructFields(elemKey)
											}
										} else if isFinalPass && strings.HasPrefix(cName, "Equal") && trackedTypes[elemKey] != nil {
											recordAllStructFields(elemKey)
										}
									}
								}
								return
							}
							if strings.HasPrefix(calleePkgPath, "k8s.io/") {
								callCallee(cc, callVal, callee, false, nil)
							}
							return
						}
						if unop, ok := cc.Value.(*ssa.UnOp); ok && unop.Op == token.MUL {
							if glob, ok := unop.X.(*ssa.Global); ok && glob.Name() == "KeyFunc" && len(cc.Args) > 0 && valProv[cc.Args[0]].Root == cfg.RootType {
								if isFinalPass {
									recordField(cfg.RootType, "ObjectMeta")
									recordField(objectMetaTypeKey, "Namespace")
									recordField(objectMetaTypeKey, "Name")
								}
								return
							}
						}
						if valProv[cc.Value].Root == cfg.RootType {
							taintCallResult(callVal, ssaProvenance{Root: cfg.RootType})
						}
						if cb := resolveSSAClosure(cc.Value, valProv); cb.Fn != nil {
							callCallee(cc, callVal, cb.Fn, false, cb.Free)
							return
						}
						if sig, ok := cc.Value.Type().Underlying().(*types.Signature); ok && sig.Recv() == nil {
							for _, candidate := range ensurePkgFuncs(fnPkg(fn)) {
								if candidate.Signature.Recv() == nil && types.AssignableTo(candidate.Type(), cc.Value.Type()) {
									callCallee(cc, callVal, candidate, false, nil)
								}
							}
						}
					case *ssa.Return:
						if len(ins.Results) > len(returnProvs) {
							returnProvs = make([]ssaProvenance, len(ins.Results))
						}
						for i, res := range ins.Results {
							if p := valProv[res]; p.Root != "" || p.Fn != nil {
								returnProvs[i] = p
							} else if cb := resolveSSAClosure(res, valProv); cb.Fn != nil {
								returnProvs[i] = cb
							}
						}
					}
				})
				if isFinalPass {
					break
				}
				if pass >= 1 && len(valProv)+len(fieldProv) == prevCount {
					isFinalPass = true
				} else if pass == 6 {
					isFinalPass = true
				}
			}
			visited[key] = visitResult{ret: returnProvs, depth: depth}
			return returnProvs
		}
		analyzeFn(entry.fn, entry.paramProv, 0)
	}

	src, err := emitTrimGoSource(cfg, rootPkg, metav1Pkg, rootNamed, trackedTypes, observed)
	if err != nil {
		t.Fatalf("emitTrimGoSource failed: %v", err)
	}
	return src
}

func emitTrimGoSource(cfg Config, rootPkg, metav1Pkg *types.Package, rootNamed *types.Named, trackedTypes map[string]*types.Named, observed map[string]map[string]map[string]bool) ([]byte, error) {
	rootPkgAlias := path.Base(path.Dir(rootPkg.Path())) + path.Base(rootPkg.Path())
	rootTypeName := rootNamed.Obj().Name()
	rootVar := strings.ToLower(rootTypeName)
	rootTrimFn := "trim" + rootTypeName

	pkgAlias := func(pkg *types.Package) string {
		if pkg == metav1Pkg {
			return "metav1"
		}
		return rootPkgAlias
	}
	var unreadFieldCount func(named *types.Named) int
	unreadFieldCount = func(named *types.Named) int {
		st := named.Underlying().(*types.Struct)
		key := namedTypeKey(named)
		obs := observed[key]
		if len(obs) == 0 {
			return 0
		}
		unread := st.NumFields() - len(obs)
		for i := 0; i < st.NumFields(); i++ {
			if f := st.Field(i); f.Embedded() && len(obs[f.Name()]) > 0 {
				if embNamed, ok := unwrapType(f.Type()).(*types.Named); ok && trackedTypes[namedTypeKey(embNamed)] != nil {
					unread += unreadFieldCount(embNamed)
				}
			}
		}
		return unread
	}
	compactingPtrField := func(elemNamed *types.Named) (*types.Var, *types.Named) {
		st := elemNamed.Underlying().(*types.Struct)
		obs := observed[namedTypeKey(elemNamed)]
		if len(obs) != 1 {
			return nil, nil
		}
		for i := 0; i < st.NumFields(); i++ {
			f := st.Field(i)
			if len(obs[f.Name()]) > 0 {
				if ptr, ok := f.Type().(*types.Pointer); ok {
					if targetNamed, ok := ptr.Elem().(*types.Named); ok && trackedTypes[namedTypeKey(targetNamed)] != nil {
						tSt := targetNamed.Underlying().(*types.Struct)
						tObs := observed[namedTypeKey(targetNamed)]
						if len(tObs) > 0 && len(tObs) < tSt.NumFields() {
							allNilable := true
							for j := 0; j < tSt.NumFields(); j++ {
								if tf := tSt.Field(j); len(tObs[tf.Name()]) > 0 && !isNilableType(tf.Type()) {
									allNilable = false
									break
								}
							}
							if allNilable {
								return f, targetNamed
							}
						}
					}
				}
			}
		}
		return nil, nil
	}

	var buf bytes.Buffer
	var emitFields func(named *types.Named, srcExpr string, customExpr map[string]string)
	emitFields = func(named *types.Named, srcExpr string, customExpr map[string]string) {
		st := named.Underlying().(*types.Struct)
		fullType := namedTypeKey(named)
		for i := 0; i < st.NumFields(); i++ {
			f := st.Field(i)
			fName := f.Name()
			if fName == "TypeMeta" {
				continue
			}
			consumers := observed[fullType][fName]
			if len(consumers) == 0 {
				continue
			}
			if _, isValStruct := f.Type().Underlying().(*types.Struct); isValStruct && customExpr[fName] == "" {
				if subNamed, ok := f.Type().(*types.Named); ok && trackedTypes[namedTypeKey(subNamed)] != nil && unreadFieldCount(subNamed) > 0 && (named == rootNamed || f.Embedded()) {
					subSrc := srcExpr + "." + fName
					if f.Embedded() && named != rootNamed {
						subSrc = srcExpr
					}
					fmt.Fprintf(&buf, "%s: %s.%s{\n", fName, pkgAlias(subNamed.Obj().Pkg()), subNamed.Obj().Name())
					emitFields(subNamed, subSrc, customExpr)
					buf.WriteString("},\n")
					continue
				}
			}
			valExpr := srcExpr + "." + fName
			if custom, ok := customExpr[fName]; ok {
				valExpr = custom
			}
			fmt.Fprintf(&buf, "%s: %s, // %s\n", fName, valExpr, strings.Join(slices.Sorted(maps.Keys(consumers)), ", "))
		}
	}

	// Discover top-level slice fields that require in-place slice trimming helpers.
	rootSt := rootNamed.Underlying().(*types.Struct)
	var preTrimCalls []string
	var sliceHelpers []*types.Named
	seenSliceHelpers := make(map[string]bool)
	for i := 0; i < rootSt.NumFields(); i++ {
		topField := rootSt.Field(i)
		if topField.Name() == "TypeMeta" || len(observed[cfg.RootType][topField.Name()]) == 0 {
			continue
		}
		topNamed, ok := unwrapType(topField.Type()).(*types.Named)
		if !ok || trackedTypes[namedTypeKey(topNamed)] == nil {
			continue
		}
		topSt := topNamed.Underlying().(*types.Struct)
		for j := 0; j < topSt.NumFields(); j++ {
			sf := topSt.Field(j)
			if len(observed[namedTypeKey(topNamed)][sf.Name()]) == 0 {
				continue
			}
			if sl, ok := sf.Type().(*types.Slice); ok {
				if elemNamed, ok := sl.Elem().(*types.Named); ok && elemNamed.Obj().Pkg() == rootPkg && unreadFieldCount(elemNamed) >= 2 {
					fnName := "trim" + pluralize(elemNamed.Obj().Name()) + "InPlace"
					preTrimCalls = append(preTrimCalls, fmt.Sprintf("%s(%s.%s.%s)", fnName, rootVar, topField.Name(), sf.Name()))
					if !seenSliceHelpers[namedTypeKey(elemNamed)] {
						seenSliceHelpers[namedTypeKey(elemNamed)] = true
						sliceHelpers = append(sliceHelpers, elemNamed)
					}
				}
			}
		}
	}

	fmt.Fprintf(&buf, "/*\nCopyright The Kubernetes Authors.\n\nLicensed under the Apache License, Version 2.0 (the \"License\");\nyou may not use this file except in compliance with the License.\nYou may obtain a copy of the License at\n\n    http://www.apache.org/licenses/LICENSE-2.0\n\nUnless required by applicable law or agreed to in writing, software\ndistributed under the License is distributed on an \"AS IS\" BASIS,\nWITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\nSee the License for the specific language governing permissions and\nlimitations under the License.\n*/\n\n// Code generated by %s; DO NOT EDIT.\n\npackage %s\n\nimport (\n\t%s \"%s\"\n\t\"k8s.io/apimachinery/pkg/api/meta\"\n\tmetav1 \"k8s.io/apimachinery/pkg/apis/meta/v1\"\n)\n\n",
		cfg.GeneratorCmd, cfg.TargetPkgName, rootPkgAlias, rootPkg.Path())
	fmt.Fprintf(&buf, "// trimInformerObject strips unused fields from objects stored in the controlplane\n// versionedInformers cache. For %s, it zeroes whole struct fields that have zero\n// reads across kube-apiserver's %sInformer consumers, generated via interprocedural\n// Go SSA analysis in test/utils/informertrim.\nfunc trimInformerObject(obj interface{}) (interface{}, error) {\n\tif %s, ok := obj.(*%s.%s); ok {\n\t\treturn %s(%s), nil\n\t}\n\tif accessor, err := meta.Accessor(obj); err == nil && accessor.GetManagedFields() != nil {\n\t\taccessor.SetManagedFields(nil)\n\t}\n\treturn obj, nil\n}\n\n",
		pluralize(rootTypeName), rootTypeName, rootVar, rootPkgAlias, rootTypeName, rootTrimFn, rootVar)

	fmt.Fprintf(&buf, "func %s(%s *%s.%s) *%s.%s {\n\tif %s == nil {\n\t\treturn nil\n\t}\n", rootTrimFn, rootVar, rootPkgAlias, rootTypeName, rootPkgAlias, rootTypeName, rootVar)
	for _, call := range preTrimCalls {
		fmt.Fprintf(&buf, "\t%s\n", call)
	}
	fmt.Fprintf(&buf, "\t*%s = %s.%s{\n", rootVar, rootPkgAlias, rootTypeName)
	emitFields(rootNamed, rootVar, nil)
	fmt.Fprintf(&buf, "}\nreturn %s\n}\n\n", rootVar)

	var ptrSliceHelpers []*types.Named
	seenPtrSlice := make(map[string]bool)
	for _, elemNamed := range sliceHelpers {
		elemName := elemNamed.Obj().Name()
		fnName := "trim" + pluralize(elemName) + "InPlace"
		words := splitCamelWords(elemName)
		sliceVar := strings.ToLower(pluralize(words[len(words)-1]))
		itemVar := sliceVar[:1]
		if strings.HasSuffix(elemName, "Status") {
			itemVar = camelInitials(elemName)
		}

		custom := make(map[string]string)
		var collectCustomSlices func(named *types.Named)
		collectCustomSlices = func(named *types.Named) {
			st := named.Underlying().(*types.Struct)
			for i := 0; i < st.NumFields(); i++ {
				sf := st.Field(i)
				if sf.Embedded() {
					if emb, ok := unwrapType(sf.Type()).(*types.Named); ok && trackedTypes[namedTypeKey(emb)] != nil {
						collectCustomSlices(emb)
					}
				}
				if sl, ok := sf.Type().(*types.Slice); ok {
					if subElem, ok := sl.Elem().(*types.Named); ok && trackedTypes[namedTypeKey(subElem)] != nil {
						if ptrF, _ := compactingPtrField(subElem); ptrF != nil {
							subFn := "trim" + pluralize(subElem.Obj().Name()) + "InPlace"
							custom[sf.Name()] = fmt.Sprintf("%s(%s.%s)", subFn, itemVar, sf.Name())
							if !seenPtrSlice[namedTypeKey(subElem)] {
								seenPtrSlice[namedTypeKey(subElem)] = true
								ptrSliceHelpers = append(ptrSliceHelpers, subElem)
							}
						}
					}
				}
			}
		}
		collectCustomSlices(elemNamed)

		fmt.Fprintf(&buf, "func %s(%s []%s.%s) {\nfor i := range %s {\n%s := &%s[i]\n*%s = %s.%s{\n",
			fnName, sliceVar, rootPkgAlias, elemName, sliceVar, itemVar, sliceVar, itemVar, rootPkgAlias, elemName)
		emitFields(elemNamed, itemVar, custom)
		buf.WriteString("}\n}\n}\n\n")
	}

	for _, subElem := range ptrSliceHelpers {
		subName := subElem.Obj().Name()
		fnName := "trim" + pluralize(subName) + "InPlace"
		sliceVar := strings.ToLower(splitCamelWords(subName)[0])
		ptrField, ptrTarget := compactingPtrField(subElem)
		ptrVar := camelInitials(ptrField.Name())
		ptrTargetSt := ptrTarget.Underlying().(*types.Struct)
		var nonNilChecks []string
		for i := 0; i < ptrTargetSt.NumFields(); i++ {
			f := ptrTargetSt.Field(i)
			if len(observed[namedTypeKey(ptrTarget)][f.Name()]) > 0 && isNilableType(f.Type()) {
				nonNilChecks = append(nonNilChecks, ptrVar+"."+f.Name()+" != nil")
			}
		}
		fmt.Fprintf(&buf, "func %s(%s []%s.%s) []%s.%s {\nkeep := false\nfor i := range %s {\n%s := %s[i].%s\nif %s != nil && (%s) {\n*%s = %s.%s{\n",
			fnName, sliceVar, rootPkgAlias, subName, rootPkgAlias, subName, sliceVar, ptrVar, sliceVar, ptrField.Name(), ptrVar, strings.Join(nonNilChecks, " || "), ptrVar, rootPkgAlias, ptrTarget.Obj().Name())
		emitFields(ptrTarget, ptrVar, nil)
		fmt.Fprintf(&buf, "}\n%s[i] = %s.%s{\n", sliceVar, rootPkgAlias, subName)
		emitFields(subElem, sliceVar+"[i]", map[string]string{ptrField.Name(): ptrVar})
		fmt.Fprintf(&buf, "}\nkeep = true\n} else {\n%s[i] = %s.%s{}\n}\n}\nif !keep {\nreturn nil\n}\nreturn %s\n}\n", sliceVar, rootPkgAlias, subName, sliceVar)
	}
	return format.Source(buf.Bytes())
}

func isCandidateK8sPkg(pkgPath string) bool {
	return strings.HasPrefix(pkgPath, "k8s.io/") &&
		!strings.HasPrefix(pkgPath, "k8s.io/client-go/") &&
		!strings.HasPrefix(pkgPath, "k8s.io/klog") &&
		!strings.HasPrefix(pkgPath, "k8s.io/component-base/metrics")
}

func fnPkg(fn *ssa.Function) *ssa.Package {
	if fn == nil {
		return nil
	}
	if fn.Pkg != nil {
		return fn.Pkg
	}
	if orig := fn.Origin(); orig != nil && orig.Pkg != nil {
		return orig.Pkg
	}
	if fn.Parent() != nil {
		return fnPkg(fn.Parent())
	}
	return nil
}

func fnPkgPath(fn *ssa.Function) string {
	if p := fnPkg(fn); p != nil && p.Pkg != nil {
		return p.Pkg.Path()
	}
	return ""
}

func hasNonTargetResourceGuard(fn *ssa.Function, gvrArg ssa.Value, resourcePlural string) bool {
	if call, ok := gvrArg.(*ssa.Call); ok && len(call.Common().Args) == 1 {
		if c, ok := call.Common().Args[0].(*ssa.Const); ok && c.Value != nil && c.Value.Kind() == constant.String {
			return !strings.EqualFold(constant.StringVal(c.Value), resourcePlural)
		}
	}
	guarded := false
	forEachInstr(fn, func(instr ssa.Instruction) {
		if bo, ok := instr.(*ssa.BinOp); ok && (bo.Op == token.EQL || bo.Op == token.NEQ) {
			for _, op := range [2]ssa.Value{bo.X, bo.Y} {
				if c, ok := op.(*ssa.Const); ok && c.Value != nil && c.Value.Kind() == constant.String && strings.EqualFold(constant.StringVal(c.Value), resourcePlural) {
					guarded = true
				}
			}
		}
	})
	return guarded
}

func forEachInstr(fn *ssa.Function, visit func(instr ssa.Instruction)) {
	for _, b := range fn.Blocks {
		for _, instr := range b.Instrs {
			visit(instr)
		}
	}
}

func propagateSSAFieldAccess(ins, x ssa.Value, fieldIdx int, valProv map[ssa.Value]ssaProvenance, recordField func(string, string), isFinalPass bool) {
	if p := valProv[x]; p.Root != "" {
		sName, fName := namedTypeKey(x.Type()), fieldNameOf(x.Type(), fieldIdx)
		if p.OrigStruct != "" && sName == p.TargetType {
			sName = p.OrigStruct
		}
		if sName == "k8s.io/client-go/tools/cache.DeletedFinalStateUnknown" && fName == "Obj" {
			valProv[ins] = p
		} else if fName != "" {
			path := p.Path
			if fName != "TypeMeta" {
				if path != "" {
					path += "."
				}
				path += fName
			}
			valProv[ins] = ssaProvenance{Root: p.Root, Path: path}
			if isFinalPass {
				recordField(sName, fName)
			}
		}
	}
}

func extractEventHandlerFuncs(prog *ssa.Program, v ssa.Value, addSeed func(fn *ssa.Function, paramIdxs ...int)) int {
	for {
		if _, ok := v.(*ssa.Alloc); ok || v == nil {
			break
		}
		if ins, ok := v.(ssa.Instruction); ok && len(ins.Operands(nil)) == 1 {
			v = *ins.Operands(nil)[0]
		} else {
			break
		}
	}
	count := 0
	if alloc, ok := v.(*ssa.Alloc); ok && alloc.Referrers() != nil {
		switch namedTypeKey(alloc.Type()) {
		case "k8s.io/client-go/tools/cache.ResourceEventHandlerFuncs":
			for _, ref := range *alloc.Referrers() {
				if fa, ok := ref.(*ssa.FieldAddr); ok && fa.Referrers() != nil {
					fName := fieldNameOf(fa.X.Type(), fa.Field)
					for _, faRef := range *fa.Referrers() {
						if st, ok := faRef.(*ssa.Store); ok && st.Addr == fa {
							if targetFn := resolveSSAClosure(st.Val, nil).Fn; targetFn != nil {
								switch fName {
								case "AddFunc", "DeleteFunc":
									addSeed(targetFn, 0)
									count++
								case "UpdateFunc":
									addSeed(targetFn, 0, 1)
									count++
								}
							}
						}
					}
				}
			}
		case "k8s.io/client-go/tools/cache.FilteringResourceEventHandler":
			for _, ref := range *alloc.Referrers() {
				if fa, ok := ref.(*ssa.FieldAddr); ok && fa.Referrers() != nil {
					fName := fieldNameOf(fa.X.Type(), fa.Field)
					for _, faRef := range *fa.Referrers() {
						if st, ok := faRef.(*ssa.Store); ok && st.Addr == fa {
							if fName == "FilterFunc" {
								if targetFn := resolveSSAClosure(st.Val, nil).Fn; targetFn != nil {
									addSeed(targetFn, 0)
									count++
								}
							} else if fName == "Handler" {
								count += extractEventHandlerFuncs(prog, st.Val, addSeed)
							}
						}
					}
				}
			}
		}
	}
	if count == 0 && v != nil && !types.IsInterface(v.Type()) {
		mset := prog.MethodSets.MethodSet(v.Type())
		for _, m := range []struct {
			name string
			idxs []int
		}{
			{"OnAdd", []int{1}},
			{"OnUpdate", []int{1, 2}},
			{"OnDelete", []int{1}},
		} {
			if sel := mset.Lookup(nil, m.name); sel != nil {
				if fn := prog.MethodValue(sel); fn != nil {
					addSeed(fn, m.idxs...)
					count++
				}
			}
		}
	}
	return count
}

func extractIndexerFuncsFromFn(fn *ssa.Function, addSeed func(fn *ssa.Function, paramIdxs ...int)) {
	if fn == nil {
		return
	}
	if isCacheIndexFuncSig(fn.Signature) {
		addSeed(fn, 0)
	}
	for _, anon := range fn.AnonFuncs {
		extractIndexerFuncsFromFn(anon, addSeed)
	}
	forEachInstr(fn, func(instr ssa.Instruction) {
		if mu, ok := instr.(*ssa.MapUpdate); ok {
			if targetFn := resolveSSAClosure(mu.Value, nil).Fn; targetFn != nil && isCacheIndexFuncSig(targetFn.Signature) {
				addSeed(targetFn, 0)
			}
		}
	})
}

func isCacheIndexFuncSig(sig *types.Signature) bool {
	if sig == nil || sig.Recv() != nil || sig.Params().Len() != 1 || sig.Results().Len() != 2 {
		return false
	}
	if iface, ok := sig.Params().At(0).Type().Underlying().(*types.Interface); !ok || iface.NumMethods() != 0 {
		return false
	}
	sl, ok := sig.Results().At(0).Type().Underlying().(*types.Slice)
	if !ok {
		return false
	}
	basic, ok := sl.Elem().Underlying().(*types.Basic)
	return ok && basic.Kind() == types.String && sig.Results().At(1).Type().String() == "error"
}

func structFieldKey(fa *ssa.FieldAddr) string {
	if fa == nil {
		return ""
	}
	baseKey, fName := namedTypeKey(fa.X.Type()), fieldNameOf(fa.X.Type(), fa.Field)
	if baseKey == "" || fName == "" {
		return ""
	}
	return baseKey + "." + fName
}

func calleeCanReturnRootType(sig *types.Signature, rootType string) bool {
	if sig == nil || sig.Results() == nil {
		return false
	}
	for i := 0; i < sig.Results().Len(); i++ {
		if typeCanCarryRoot(sig.Results().At(i).Type(), rootType) {
			return true
		}
	}
	return false
}

func typeCanCarryRoot(t types.Type, rootType string) bool {
	if t == nil {
		return false
	}
	if namedTypeKey(t) == rootType {
		return true
	}
	if m, ok := t.Underlying().(*types.Map); ok {
		return typeCanCarryRoot(m.Elem(), rootType)
	}
	return false
}

func resolveSSAClosure(v ssa.Value, valProv map[ssa.Value]ssaProvenance) ssaProvenance {
	switch val := v.(type) {
	case *ssa.Function:
		return ssaProvenance{Fn: val}
	case *ssa.MakeClosure:
		if fn, ok := val.Fn.(*ssa.Function); ok {
			var freeProv map[int]ssaProvenance
			for i, b := range val.Bindings {
				p := valProv[b]
				if p.Fn == nil {
					c := resolveSSAClosure(b, valProv)
					p.Fn, p.Free = c.Fn, c.Free
				}
				if p.Root != "" || p.Fn != nil {
					if freeProv == nil {
						freeProv = make(map[int]ssaProvenance)
					}
					freeProv[-(i + 1)] = p
				}
			}
			return ssaProvenance{Fn: fn, Free: freeProv}
		}
	case *ssa.ChangeType:
		return resolveSSAClosure(val.X, valProv)
	}
	if p, ok := valProv[v]; ok && p.Fn != nil {
		return p
	}
	return ssaProvenance{}
}

func encodeSSAParamProv(paramProv map[int]ssaProvenance) string {
	var b strings.Builder
	for _, k := range slices.Sorted(maps.Keys(paramProv)) {
		p := paramProv[k]
		fmt.Fprintf(&b, "%d:%s/%s/%s->%s/%v[%s];", k, p.Root, p.Path, p.OrigStruct, p.TargetType, p.Fn, encodeSSAParamProv(p.Free))
	}
	return b.String()
}

func shouldSkipSSAFunction(fn *ssa.Function) bool {
	if fn == nil {
		return true
	}
	n := fn.Name()
	if strings.HasPrefix(n, "DeepCopy") || strings.HasPrefix(n, "Marshal") || strings.HasPrefix(n, "Unmarshal") ||
		strings.HasPrefix(n, "Size") || strings.HasPrefix(n, "Proto") || n == "Descriptor" || n == "SwaggerDoc" ||
		n == "trimInformerObject" || (strings.HasPrefix(n, "trim") && strings.HasSuffix(n, "InPlace")) {
		return true
	}
	p := fnPkgPath(fn)
	return strings.HasPrefix(p, "k8s.io/klog") || strings.HasPrefix(p, "github.com/go-logr/logr") || strings.HasPrefix(p, "k8s.io/component-base/metrics")
}

func isStoreOnlySSAFieldAddr(fa *ssa.FieldAddr) bool {
	if fa.Referrers() == nil || len(*fa.Referrers()) == 0 {
		return true
	}
	for _, r := range *fa.Referrers() {
		if st, ok := r.(*ssa.Store); !ok || st.Addr != fa {
			return false
		}
	}
	return true
}

func isSSALoggingOnlyValue(v ssa.Value, depth int) bool {
	if depth > 6 || v.Referrers() == nil || len(*v.Referrers()) == 0 {
		return false
	}
	for _, r := range *v.Referrers() {
		switch ins := r.(type) {
		case *ssa.Call:
			cc := ins.Common()
			if cc.IsInvoke() {
				recv, m := cc.Value.Type().String(), cc.Method.Name()
				if strings.Contains(recv, "EventRecorder") || strings.Contains(recv, "LogSink") ||
					((m == "Info" || m == "Error" || m == "Event" || m == "Eventf" || m == "AnnotatedEventf") && (strings.Contains(recv, "klog") || strings.Contains(recv, "logr") || strings.Contains(recv, "record"))) {
					continue
				}
				return false
			}
			callee := cc.StaticCallee()
			if callee == nil {
				return false
			}
			pkg, name := fnPkgPath(callee), callee.Name()
			if strings.HasPrefix(pkg, "k8s.io/klog") || strings.HasPrefix(pkg, "github.com/go-logr/logr") ||
				pkg == "k8s.io/apimachinery/pkg/util/diff" ||
				(pkg == "fmt" && (name == "Errorf" || name == "Sprint" || name == "Sprintf")) ||
				(pkg == "k8s.io/apimachinery/pkg/util/runtime" && strings.HasPrefix(name, "HandleError")) {
				continue
			}
			return false
		case *ssa.Store:
			if ins.Addr == v {
				continue
			}
			if idxAddr, ok := ins.Addr.(*ssa.IndexAddr); !ok || !isSSALoggingOnlyValue(idxAddr.X, depth+1) {
				return false
			}
		case ssa.Value:
			if !isSSALoggingOnlyValue(ins, depth+1) {
				return false
			}
		default:
			return false
		}
	}
	return true
}

func isApprovedAnyBoxing(v ssa.Value, depth int) bool {
	if isSSALoggingOnlyValue(v, depth) {
		return true
	}
	if depth > 6 || v.Referrers() == nil || len(*v.Referrers()) == 0 {
		return false
	}
	for _, r := range *v.Referrers() {
		switch ins := r.(type) {
		case *ssa.Call:
			cc := ins.Common()
			if cc.IsInvoke() {
				m := cc.Method.Name()
				recv := cc.Value.Type().String()
				if m == "OnAdd" || m == "OnUpdate" || m == "OnDelete" || m == "Add" || m == "AddAfter" || m == "AddRateLimited" || m == "Done" || m == "Forget" ||
					strings.Contains(recv, "EventRecorder") || strings.Contains(recv, "LogSink") {
					continue
				}
				return false
			}
			callee := cc.StaticCallee()
			if callee == nil {
				if unop, ok := cc.Value.(*ssa.UnOp); ok && unop.Op == token.MUL {
					if glob, ok := unop.X.(*ssa.Global); ok && glob.Name() == "KeyFunc" {
						continue
					}
				}
				return false
			}
			pkg, name := fnPkgPath(callee), callee.Name()
			if name == "MetaNamespaceKeyFunc" || name == "DeletionHandlingMetaNamespaceKeyFunc" || name == "KeyFunc" || name == "NamespacedKey" || name == "PodKey" ||
				name == "DeepEqual" || name == "DeepDerivative" ||
				(pkg == "sort" && (name == "Slice" || name == "SliceStable")) ||
				strings.HasPrefix(pkg, "k8s.io/klog") || strings.HasPrefix(pkg, "github.com/go-logr/logr") || pkg == "fmt" || pkg == "reflect" ||
				pkg == "k8s.io/apimachinery/pkg/util/diff" ||
				strings.HasPrefix(pkg, "k8s.io/apimachinery/pkg/conversion") ||
				strings.HasPrefix(pkg, "k8s.io/apimachinery/pkg/util/runtime") ||
				strings.HasPrefix(pkg, "k8s.io/kubernetes/pkg/controller/") {
				continue
			}
			return false
		case *ssa.Return:
			continue
		case *ssa.Store:
			if ins.Addr == v {
				continue
			}
			if idxAddr, ok := ins.Addr.(*ssa.IndexAddr); !ok || !isApprovedAnyBoxing(idxAddr.X, depth+1) {
				return false
			}
		case ssa.Value:
			if !isApprovedAnyBoxing(ins, depth+1) {
				return false
			}
		default:
			return false
		}
	}
	return true
}

func consumerPkgShortName(pkgPath string) string {
	if rest, ok := strings.CutPrefix(pkgPath, "k8s.io/kubernetes/pkg/controller/"); ok {
		sub := strings.Split(rest, "/")
		if (sub[0] == "volume" || sub[0] == "scheduling") && len(sub) >= 2 {
			return sub[0] + "/" + sub[1]
		}
		return sub[0]
	}
	parts := strings.Split(pkgPath, "/")
	if len(parts) >= 2 && parts[0] == "k8s.io" && parts[1] != "kubernetes" {
		return strings.ReplaceAll(strings.TrimSuffix(parts[1], "-admission"), "-", "")
	}
	for i, p := range parts {
		if (p == "v1" || p == "v1alpha1" || p == "v1beta1") && i > 0 {
			return parts[i-1]
		}
	}
	return path.Base(pkgPath)
}

func isNilableType(t types.Type) bool {
	if t == nil {
		return false
	}
	switch t.Underlying().(type) {
	case *types.Pointer, *types.Slice, *types.Map, *types.Interface, *types.Signature, *types.Chan:
		return true
	default:
		return false
	}
}

func namedTypeKey(t types.Type) string {
	if named, ok := unwrapType(t).(*types.Named); ok && named.Obj() != nil && named.Obj().Pkg() != nil {
		return named.Obj().Pkg().Path() + "." + named.Obj().Name()
	}
	return ""
}

func unwrapType(t types.Type) types.Type {
	for t != nil {
		switch t.(type) {
		case *types.Pointer, *types.Slice, *types.Array:
			t = t.(interface{ Elem() types.Type }).Elem()
		default:
			return t
		}
	}
	return nil
}

func fieldNameOf(t types.Type, idx int) string {
	if st, ok := unwrapType(t).Underlying().(*types.Struct); ok && idx >= 0 && idx < st.NumFields() {
		return st.Field(idx).Name()
	}
	return ""
}

func pluralize(name string) string {
	if strings.HasSuffix(name, "Status") {
		return name + "es"
	}
	return name + "s"
}

func splitCamelWords(s string) []string {
	var words []string
	start := 0
	runes := []rune(s)
	for i := 1; i < len(runes); i++ {
		if unicode.IsUpper(runes[i]) && (unicode.IsLower(runes[i-1]) || (i+1 < len(runes) && unicode.IsLower(runes[i+1]))) {
			words = append(words, string(runes[start:i]))
			start = i
		}
	}
	return append(words, string(runes[start:]))
}

func camelInitials(s string) string {
	words := splitCamelWords(s)
	var b strings.Builder
	for _, w := range words {
		b.WriteRune(unicode.ToLower(rune(w[0])))
	}
	return b.String()
}
