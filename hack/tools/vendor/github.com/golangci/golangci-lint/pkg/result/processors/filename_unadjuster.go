package processors

import (
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/go/packages"

	"github.com/golangci/golangci-lint/pkg/logutils"
	"github.com/golangci/golangci-lint/pkg/result"
)

type posMapper func(pos token.Position) token.Position

type adjustMap struct {
	sync.Mutex
	m map[string]posMapper
}

// FilenameUnadjuster is needed because a lot of linters use fset.Position(f.Pos())
// to get filename. And they return adjusted filename (e.g. *.qtpl) for an issue. We need
// restore real .go filename to properly output it, parse it, etc.
type FilenameUnadjuster struct {
	m                   map[string]posMapper // map from adjusted filename to position mapper: adjusted -> unadjusted position
	log                 logutils.Log
	loggedUnadjustments map[string]bool
}

var _ Processor = &FilenameUnadjuster{}

func processUnadjusterPkg(m *adjustMap, pkg *packages.Package, log logutils.Log) {
	fset := token.NewFileSet() // it's more memory efficient to not store all in one fset

	for _, filename := range pkg.CompiledGoFiles {
		// It's important to call func here to run GC
		processUnadjusterFile(filename, m, log, fset)
	}
}

func processUnadjusterFile(filename string, m *adjustMap, log logutils.Log, fset *token.FileSet) {
	syntax, err := parser.ParseFile(fset, filename, nil, parser.ParseComments)
	if err != nil {
		// Error will be reported by typecheck
		return
	}

	adjustedFilename := fset.PositionFor(syntax.Pos(), true).Filename
	if adjustedFilename == "" {
		return
	}

	unadjustedFilename := fset.PositionFor(syntax.Pos(), false).Filename
	if unadjustedFilename == "" || unadjustedFilename == adjustedFilename {
		return
	}

	if !strings.HasSuffix(unadjustedFilename, ".go") {
		return // file.go -> /caches/cgo-xxx
	}

	m.Lock()
	defer m.Unlock()
	m.m[adjustedFilename] = func(adjustedPos token.Position) token.Position {
		tokenFile := fset.File(syntax.Pos())
		if tokenFile == nil {
			log.Warnf("Failed to get token file for %s", adjustedFilename)
			return adjustedPos
		}
		return fset.PositionFor(tokenFile.Pos(adjustedPos.Offset), false)
	}
}

func NewFilenameUnadjuster(pkgs []*packages.Package, log logutils.Log) *FilenameUnadjuster {
	m := adjustMap{m: map[string]posMapper{}}

	startedAt := time.Now()
	var wg sync.WaitGroup
	wg.Add(len(pkgs))
	for _, pkg := range pkgs {
		go func(pkg *packages.Package) {
			// It's important to call func here to run GC
			processUnadjusterPkg(&m, pkg, log)
			wg.Done()
		}(pkg)
	}
	wg.Wait()
	log.Infof("Pre-built %d adjustments in %s", len(m.m), time.Since(startedAt))

	return &FilenameUnadjuster{
		m:                   m.m,
		log:                 log,
		loggedUnadjustments: map[string]bool{},
	}
}

func (p FilenameUnadjuster) Name() string {
	return "filename_unadjuster"
}

func (p *FilenameUnadjuster) Process(issues []result.Issue) ([]result.Issue, error) {
	return transformIssues(issues, func(i *result.Issue) *result.Issue {
		issueFilePath := i.FilePath()
		if !filepath.IsAbs(i.FilePath()) {
			absPath, err := filepath.Abs(i.FilePath())
			if err != nil {
				p.log.Warnf("failed to build abs path for %q: %s", i.FilePath(), err)
				return i
			}
			issueFilePath = absPath
		}

		mapper := p.m[issueFilePath]
		if mapper == nil {
			return i
		}

		newI := *i
		newI.Pos = mapper(i.Pos)
		if !p.loggedUnadjustments[i.Pos.Filename] {
			p.log.Infof("Unadjusted from %v to %v", i.Pos, newI.Pos)
			p.loggedUnadjustments[i.Pos.Filename] = true
		}
		return &newI
	}), nil
}

func (FilenameUnadjuster) Finish() {}
