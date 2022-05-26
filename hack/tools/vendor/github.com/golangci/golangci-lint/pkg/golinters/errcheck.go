package golinters

import (
	"bufio"
	"fmt"
	"os"
	"os/user"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"github.com/kisielk/errcheck/errcheck"
	"github.com/pkg/errors"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/fsutils"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

func NewErrcheck() *goanalysis.Linter {
	const linterName = "errcheck"

	var mu sync.Mutex
	var res []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: linterName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}

	return goanalysis.NewLinter(
		linterName,
		"Errcheck is a program for checking for unchecked errors "+
			"in go programs. These unchecked errors can be critical bugs in some cases",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		// copied from errcheck
		checker, err := getChecker(&lintCtx.Settings().Errcheck)
		if err != nil {
			lintCtx.Log.Errorf("failed to get checker: %v", err)
			return
		}

		checker.Tags = lintCtx.Cfg.Run.BuildTags

		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			pkg := &packages.Package{
				Fset:      pass.Fset,
				Syntax:    pass.Files,
				Types:     pass.Pkg,
				TypesInfo: pass.TypesInfo,
			}

			errcheckIssues := checker.CheckPackage(pkg).Unique()
			if len(errcheckIssues.UncheckedErrors) == 0 {
				return nil, nil
			}

			issues := make([]goanalysis.Issue, len(errcheckIssues.UncheckedErrors))
			for i, err := range errcheckIssues.UncheckedErrors {
				var text string
				if err.FuncName != "" {
					code := err.SelectorName
					if err.SelectorName == "" {
						code = err.FuncName
					}

					text = fmt.Sprintf(
						"Error return value of %s is not checked",
						formatCode(code, lintCtx.Cfg),
					)
				} else {
					text = "Error return value is not checked"
				}

				issues[i] = goanalysis.NewIssue(
					&result.Issue{
						FromLinter: linterName,
						Text:       text,
						Pos:        err.Pos,
					},
					pass,
				)
			}

			mu.Lock()
			res = append(res, issues...)
			mu.Unlock()

			return nil, nil
		}
	}).WithIssuesReporter(func(*linter.Context) []goanalysis.Issue {
		return res
	}).WithLoadMode(goanalysis.LoadModeTypesInfo)
}

// parseIgnoreConfig was taken from errcheck in order to keep the API identical.
// https://github.com/kisielk/errcheck/blob/1787c4bee836470bf45018cfbc783650db3c6501/main.go#L25-L60
func parseIgnoreConfig(s string) (map[string]*regexp.Regexp, error) {
	if s == "" {
		return nil, nil
	}

	cfg := map[string]*regexp.Regexp{}

	for _, pair := range strings.Split(s, ",") {
		colonIndex := strings.Index(pair, ":")
		var pkg, re string
		if colonIndex == -1 {
			pkg = ""
			re = pair
		} else {
			pkg = pair[:colonIndex]
			re = pair[colonIndex+1:]
		}
		regex, err := regexp.Compile(re)
		if err != nil {
			return nil, err
		}
		cfg[pkg] = regex
	}

	return cfg, nil
}

func getChecker(errCfg *config.ErrcheckSettings) (*errcheck.Checker, error) {
	ignoreConfig, err := parseIgnoreConfig(errCfg.Ignore)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse 'ignore' directive")
	}

	checker := errcheck.Checker{
		Exclusions: errcheck.Exclusions{
			BlankAssignments:       !errCfg.CheckAssignToBlank,
			TypeAssertions:         !errCfg.CheckTypeAssertions,
			SymbolRegexpsByPackage: map[string]*regexp.Regexp{},
		},
	}

	if !errCfg.DisableDefaultExclusions {
		checker.Exclusions.Symbols = append(checker.Exclusions.Symbols, errcheck.DefaultExcludedSymbols...)
	}

	for pkg, re := range ignoreConfig {
		checker.Exclusions.SymbolRegexpsByPackage[pkg] = re
	}

	if errCfg.Exclude != "" {
		exclude, err := readExcludeFile(errCfg.Exclude)
		if err != nil {
			return nil, err
		}

		checker.Exclusions.Symbols = append(checker.Exclusions.Symbols, exclude...)
	}

	checker.Exclusions.Symbols = append(checker.Exclusions.Symbols, errCfg.ExcludeFunctions...)

	return &checker, nil
}

func getFirstPathArg() string {
	args := os.Args

	// skip all args ([golangci-lint, run/linters]) before files/dirs list
	for len(args) != 0 {
		if args[0] == "run" {
			args = args[1:]
			break
		}

		args = args[1:]
	}

	// find first file/dir arg
	firstArg := "./..."
	for _, arg := range args {
		if !strings.HasPrefix(arg, "-") {
			firstArg = arg
			break
		}
	}

	return firstArg
}

func setupConfigFileSearch(name string) []string {
	if strings.HasPrefix(name, "~") {
		if u, err := user.Current(); err == nil {
			name = strings.Replace(name, "~", u.HomeDir, 1)
		}
	}

	if filepath.IsAbs(name) {
		return []string{name}
	}

	firstArg := getFirstPathArg()

	absStartPath, err := filepath.Abs(firstArg)
	if err != nil {
		absStartPath = filepath.Clean(firstArg)
	}

	// start from it
	var curDir string
	if fsutils.IsDir(absStartPath) {
		curDir = absStartPath
	} else {
		curDir = filepath.Dir(absStartPath)
	}

	// find all dirs from it up to the root
	configSearchPaths := []string{filepath.Join(".", name)}
	for {
		configSearchPaths = append(configSearchPaths, filepath.Join(curDir, name))
		newCurDir := filepath.Dir(curDir)
		if curDir == newCurDir || newCurDir == "" {
			break
		}
		curDir = newCurDir
	}

	return configSearchPaths
}

func readExcludeFile(name string) ([]string, error) {
	var err error
	var fh *os.File

	for _, path := range setupConfigFileSearch(name) {
		if fh, err = os.Open(path); err == nil {
			break
		}
	}

	if fh == nil {
		return nil, errors.Wrapf(err, "failed reading exclude file: %s", name)
	}

	scanner := bufio.NewScanner(fh)

	var excludes []string
	for scanner.Scan() {
		excludes = append(excludes, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return nil, errors.Wrapf(err, "failed scanning file: %s", name)
	}

	return excludes, nil
}
