package lint

import (
	"context"
	"fmt"
	"go/build"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/pkg/errors"
	"golang.org/x/tools/go/packages"

	"github.com/golangci/golangci-lint/internal/pkgcache"
	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/exitcodes"
	"github.com/golangci/golangci-lint/pkg/fsutils"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis/load"
	"github.com/golangci/golangci-lint/pkg/goutil"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/logutils"
)

type ContextLoader struct {
	cfg         *config.Config
	log         logutils.Log
	debugf      logutils.DebugFunc
	goenv       *goutil.Env
	pkgTestIDRe *regexp.Regexp
	lineCache   *fsutils.LineCache
	fileCache   *fsutils.FileCache
	pkgCache    *pkgcache.Cache
	loadGuard   *load.Guard
}

func NewContextLoader(cfg *config.Config, log logutils.Log, goenv *goutil.Env,
	lineCache *fsutils.LineCache, fileCache *fsutils.FileCache, pkgCache *pkgcache.Cache, loadGuard *load.Guard) *ContextLoader {
	return &ContextLoader{
		cfg:         cfg,
		log:         log,
		debugf:      logutils.Debug("loader"),
		goenv:       goenv,
		pkgTestIDRe: regexp.MustCompile(`^(.*) \[(.*)\.test\]`),
		lineCache:   lineCache,
		fileCache:   fileCache,
		pkgCache:    pkgCache,
		loadGuard:   loadGuard,
	}
}

func (cl *ContextLoader) prepareBuildContext() {
	// Set GOROOT to have working cross-compilation: cross-compiled binaries
	// have invalid GOROOT. XXX: can't use runtime.GOROOT().
	goroot := cl.goenv.Get(goutil.EnvGoRoot)
	if goroot == "" {
		return
	}

	os.Setenv("GOROOT", goroot)
	build.Default.GOROOT = goroot
	build.Default.BuildTags = cl.cfg.Run.BuildTags
}

func (cl *ContextLoader) findLoadMode(linters []*linter.Config) packages.LoadMode {
	loadMode := packages.LoadMode(0)
	for _, lc := range linters {
		loadMode |= lc.LoadMode
	}

	return loadMode
}

func (cl *ContextLoader) buildArgs() []string {
	args := cl.cfg.Run.Args
	if len(args) == 0 {
		return []string{"./..."}
	}

	var retArgs []string
	for _, arg := range args {
		if strings.HasPrefix(arg, ".") || filepath.IsAbs(arg) {
			retArgs = append(retArgs, arg)
		} else {
			// go/packages doesn't work well if we don't have the prefix ./ for local packages
			retArgs = append(retArgs, fmt.Sprintf(".%c%s", filepath.Separator, arg))
		}
	}

	return retArgs
}

func (cl *ContextLoader) makeBuildFlags() ([]string, error) {
	var buildFlags []string

	if len(cl.cfg.Run.BuildTags) != 0 {
		// go help build
		buildFlags = append(buildFlags, "-tags", strings.Join(cl.cfg.Run.BuildTags, " "))
		cl.log.Infof("Using build tags: %v", cl.cfg.Run.BuildTags)
	}

	mod := cl.cfg.Run.ModulesDownloadMode
	if mod != "" {
		// go help modules
		allowedMods := []string{"mod", "readonly", "vendor"}
		var ok bool
		for _, am := range allowedMods {
			if am == mod {
				ok = true
				break
			}
		}
		if !ok {
			return nil, fmt.Errorf("invalid modules download path %s, only (%s) allowed", mod, strings.Join(allowedMods, "|"))
		}

		buildFlags = append(buildFlags, fmt.Sprintf("-mod=%s", cl.cfg.Run.ModulesDownloadMode))
	}

	return buildFlags, nil
}

func stringifyLoadMode(mode packages.LoadMode) string {
	m := map[packages.LoadMode]string{
		packages.NeedCompiledGoFiles: "compiled_files",
		packages.NeedDeps:            "deps",
		packages.NeedExportsFile:     "exports_file",
		packages.NeedFiles:           "files",
		packages.NeedImports:         "imports",
		packages.NeedName:            "name",
		packages.NeedSyntax:          "syntax",
		packages.NeedTypes:           "types",
		packages.NeedTypesInfo:       "types_info",
		packages.NeedTypesSizes:      "types_sizes",
	}

	var flags []string
	for flag, flagStr := range m {
		if mode&flag != 0 {
			flags = append(flags, flagStr)
		}
	}

	return fmt.Sprintf("%d (%s)", mode, strings.Join(flags, "|"))
}

func (cl *ContextLoader) debugPrintLoadedPackages(pkgs []*packages.Package) {
	cl.debugf("loaded %d pkgs", len(pkgs))
	for i, pkg := range pkgs {
		var syntaxFiles []string
		for _, sf := range pkg.Syntax {
			syntaxFiles = append(syntaxFiles, pkg.Fset.Position(sf.Pos()).Filename)
		}
		cl.debugf("Loaded pkg #%d: ID=%s GoFiles=%s CompiledGoFiles=%s Syntax=%s",
			i, pkg.ID, pkg.GoFiles, pkg.CompiledGoFiles, syntaxFiles)
	}
}

func (cl *ContextLoader) parseLoadedPackagesErrors(pkgs []*packages.Package) error {
	for _, pkg := range pkgs {
		for _, err := range pkg.Errors {
			if strings.Contains(err.Msg, "no Go files") {
				return errors.Wrapf(exitcodes.ErrNoGoFiles, "package %s", pkg.PkgPath)
			}
			if strings.Contains(err.Msg, "cannot find package") {
				// when analyzing not existing directory
				return errors.Wrap(exitcodes.ErrFailure, err.Msg)
			}
		}
	}

	return nil
}

func (cl *ContextLoader) loadPackages(ctx context.Context, loadMode packages.LoadMode) ([]*packages.Package, error) {
	defer func(startedAt time.Time) {
		cl.log.Infof("Go packages loading at mode %s took %s", stringifyLoadMode(loadMode), time.Since(startedAt))
	}(time.Now())

	cl.prepareBuildContext()

	buildFlags, err := cl.makeBuildFlags()
	if err != nil {
		return nil, errors.Wrap(err, "failed to make build flags for go list")
	}

	conf := &packages.Config{
		Mode:       loadMode,
		Tests:      cl.cfg.Run.AnalyzeTests,
		Context:    ctx,
		BuildFlags: buildFlags,
		Logf:       cl.debugf,
		//TODO: use fset, parsefile, overlay
	}

	args := cl.buildArgs()
	cl.debugf("Built loader args are %s", args)
	pkgs, err := packages.Load(conf, args...)
	if err != nil {
		return nil, errors.Wrap(err, "failed to load with go/packages")
	}

	// Currently, go/packages doesn't guarantee that error will be returned
	// if context was canceled. See
	// https://github.com/golang/tools/commit/c5cec6710e927457c3c29d6c156415e8539a5111#r39261855
	if ctx.Err() != nil {
		return nil, errors.Wrap(ctx.Err(), "timed out to load packages")
	}

	if loadMode&packages.NeedSyntax == 0 {
		// Needed e.g. for go/analysis loading.
		fset := token.NewFileSet()
		packages.Visit(pkgs, nil, func(pkg *packages.Package) {
			pkg.Fset = fset
			cl.loadGuard.AddMutexForPkg(pkg)
		})
	}

	cl.debugPrintLoadedPackages(pkgs)

	if err := cl.parseLoadedPackagesErrors(pkgs); err != nil {
		return nil, err
	}

	return cl.filterTestMainPackages(pkgs), nil
}

func (cl *ContextLoader) tryParseTestPackage(pkg *packages.Package) (name string, isTest bool) {
	matches := cl.pkgTestIDRe.FindStringSubmatch(pkg.ID)
	if matches == nil {
		return "", false
	}

	return matches[1], true
}

func (cl *ContextLoader) filterTestMainPackages(pkgs []*packages.Package) []*packages.Package {
	var retPkgs []*packages.Package
	for _, pkg := range pkgs {
		if pkg.Name == "main" && strings.HasSuffix(pkg.PkgPath, ".test") {
			// it's an implicit testmain package
			cl.debugf("skip pkg ID=%s", pkg.ID)
			continue
		}

		retPkgs = append(retPkgs, pkg)
	}

	return retPkgs
}

func (cl *ContextLoader) filterDuplicatePackages(pkgs []*packages.Package) []*packages.Package {
	packagesWithTests := map[string]bool{}
	for _, pkg := range pkgs {
		name, isTest := cl.tryParseTestPackage(pkg)
		if !isTest {
			continue
		}
		packagesWithTests[name] = true
	}

	cl.debugf("package with tests: %#v", packagesWithTests)

	var retPkgs []*packages.Package
	for _, pkg := range pkgs {
		_, isTest := cl.tryParseTestPackage(pkg)
		if !isTest && packagesWithTests[pkg.PkgPath] {
			// If tests loading is enabled,
			// for package with files a.go and a_test.go go/packages loads two packages:
			// 1. ID=".../a" GoFiles=[a.go]
			// 2. ID=".../a [.../a.test]" GoFiles=[a.go a_test.go]
			// We need only the second package, otherwise we can get warnings about unused variables/fields/functions
			// in a.go if they are used only in a_test.go.
			cl.debugf("skip pkg ID=%s because we load it with test package", pkg.ID)
			continue
		}

		retPkgs = append(retPkgs, pkg)
	}

	return retPkgs
}

func (cl *ContextLoader) Load(ctx context.Context, linters []*linter.Config) (*linter.Context, error) {
	loadMode := cl.findLoadMode(linters)
	pkgs, err := cl.loadPackages(ctx, loadMode)
	if err != nil {
		return nil, errors.Wrap(err, "failed to load packages")
	}

	deduplicatedPkgs := cl.filterDuplicatePackages(pkgs)

	if len(deduplicatedPkgs) == 0 {
		return nil, exitcodes.ErrNoGoFiles
	}

	ret := &linter.Context{
		Packages: deduplicatedPkgs,

		// At least `unused` linters works properly only on original (not deduplicated) packages,
		// see https://github.com/golangci/golangci-lint/pull/585.
		OriginalPackages: pkgs,

		Cfg:       cl.cfg,
		Log:       cl.log,
		FileCache: cl.fileCache,
		LineCache: cl.lineCache,
		PkgCache:  cl.pkgCache,
		LoadGuard: cl.loadGuard,
	}

	return ret, nil
}
