package checkers

import (
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/quasilyte/go-ruleguard/ruleguard"

	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "ruleguard"
	info.Tags = []string{"style", "experimental"}
	info.Params = linter.CheckerParams{
		"rules": {
			Value: "",
			Usage: "comma-separated list of gorule file paths. Glob patterns such as 'rules-*.go' may be specified",
		},
		"debug": {
			Value: "",
			Usage: "enable debug for the specified named rules group",
		},
		"failOnError": {
			Value: false,
			Usage: "deprecated, use failOn param; if set to true, identical to failOn='all', otherwise failOn=''",
		},
		"failOn": {
			Value: "",
			Usage: `Determines the behavior when an error occurs while parsing ruleguard files.
If flag is not set, log error and skip rule files that contain an error.
If flag is set, the value must be a comma-separated list of error conditions.
* 'import': rule refers to a package that cannot be loaded.
* 'dsl':    gorule file does not comply with the ruleguard DSL.`,
		},
		"enable": {
			Value: "<all>",
			Usage: "comma-separated list of enabled groups or skip empty to enable everything",
		},
		"disable": {
			Value: "",
			Usage: "comma-separated list of disabled groups or skip empty to enable everything",
		},
	}
	info.Summary = "Runs user-defined rules using ruleguard linter"
	info.Details = "Reads a rules file and turns them into go-critic checkers."
	info.Before = `N/A`
	info.After = `N/A`
	info.Note = "See https://github.com/quasilyte/go-ruleguard."

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return newRuleguardChecker(&info, ctx)
	})
}

// parseErrorHandler is used to determine whether to ignore or fail ruleguard parsing errors.
type parseErrorHandler struct {
	// failureConditions is a map of predicates which are evaluated against a ruleguard parsing error.
	// If at least one predicate returns true, then an error is returned.
	// Otherwise, the ruleguard file is skipped.
	failureConditions map[string]func(err error) bool
}

// failOnParseError returns true if a parseError occurred and that error should be not be ignored.
func (e parseErrorHandler) failOnParseError(parseError error) bool {
	for _, p := range e.failureConditions {
		if p(parseError) {
			return true
		}
	}
	return false
}

func newErrorHandler(failOnErrorFlag string) (*parseErrorHandler, error) {
	h := parseErrorHandler{
		failureConditions: make(map[string]func(err error) bool),
	}
	var failOnErrorPredicates = map[string]func(error) bool{
		"dsl":    func(err error) bool { var e *ruleguard.ImportError; return !errors.As(err, &e) },
		"import": func(err error) bool { var e *ruleguard.ImportError; return errors.As(err, &e) },
		"all":    func(err error) bool { return true },
	}
	for _, k := range strings.Split(failOnErrorFlag, ",") {
		if k == "" {
			continue
		}
		if p, ok := failOnErrorPredicates[k]; ok {
			h.failureConditions[k] = p
		} else {
			// Wrong flag value.
			supportedValues := []string{}
			for key := range failOnErrorPredicates {
				supportedValues = append(supportedValues, key)
			}
			return nil, fmt.Errorf("ruleguard init error: 'failOnError' flag '%s' is invalid. It must be a comma-separated list and supported values are '%s'",
				k, strings.Join(supportedValues, ","))
		}
	}
	return &h, nil
}

func newRuleguardChecker(info *linter.CheckerInfo, ctx *linter.CheckerContext) (*ruleguardChecker, error) {
	c := &ruleguardChecker{
		ctx:        ctx,
		debugGroup: info.Params.String("debug"),
	}
	rulesFlag := info.Params.String("rules")
	if rulesFlag == "" {
		return c, nil
	}
	failOn := info.Params.String("failOn")
	if failOn == "" {
		if info.Params.Bool("failOnError") {
			failOn = "all"
		}
	}
	h, err := newErrorHandler(failOn)
	if err != nil {
		return nil, err
	}

	engine := ruleguard.NewEngine()
	engine.InferBuildContext()
	fset := token.NewFileSet()
	filePatterns := strings.Split(rulesFlag, ",")

	enabledGroups := make(map[string]bool)
	disabledGroups := make(map[string]bool)

	for _, g := range strings.Split(info.Params.String("disable"), ",") {
		g = strings.TrimSpace(g)
		disabledGroups[g] = true
	}
	flagEnable := info.Params.String("enable")
	if flagEnable != "<all>" {
		for _, g := range strings.Split(flagEnable, ",") {
			g = strings.TrimSpace(g)
			enabledGroups[g] = true
		}
	}
	ruleguardDebug := os.Getenv("GOCRITIC_RULEGUARD_DEBUG") != ""

	loadContext := &ruleguard.LoadContext{
		Fset:         fset,
		DebugImports: ruleguardDebug,
		DebugPrint:   debugPrint,
		GroupFilter: func(g string) bool {
			whyDisabled := ""
			enabled := flagEnable == "<all>" || enabledGroups[g]
			switch {
			case !enabled:
				whyDisabled = "not enabled by -enabled flag"
			case disabledGroups[g]:
				whyDisabled = "disabled by -disable flag"
			}
			if ruleguardDebug {
				if whyDisabled != "" {
					debugPrint(fmt.Sprintf("(-) %s is %s", g, whyDisabled))
				} else {
					debugPrint(fmt.Sprintf("(+) %s is enabled", g))
				}
			}
			return whyDisabled == ""
		},
	}

	loaded := 0
	for _, filePattern := range filePatterns {
		filenames, err := filepath.Glob(strings.TrimSpace(filePattern))
		if err != nil {
			// The only possible returned error is ErrBadPattern, when pattern is malformed.
			log.Printf("ruleguard init error: %+v", err)
			continue
		}
		if len(filenames) == 0 {
			return nil, fmt.Errorf("ruleguard init error: no file matching '%s'", strings.TrimSpace(filePattern))
		}
		for _, filename := range filenames {
			data, err := os.ReadFile(filename)
			if err != nil {
				if h.failOnParseError(err) {
					return nil, fmt.Errorf("ruleguard init error: %+v", err)
				}
				log.Printf("ruleguard init error, skip %s: %+v", filename, err)
			}
			if err := engine.Load(loadContext, filename, bytes.NewReader(data)); err != nil {
				if h.failOnParseError(err) {
					return nil, fmt.Errorf("ruleguard init error: %+v", err)
				}
				log.Printf("ruleguard init error, skip %s: %+v", filename, err)
			}
			loaded++
		}
	}

	if loaded != 0 {
		c.engine = engine
	}
	return c, nil
}

type ruleguardChecker struct {
	ctx *linter.CheckerContext

	debugGroup string
	engine     *ruleguard.Engine
}

func (c *ruleguardChecker) WalkFile(f *ast.File) {
	if c.engine == nil {
		return
	}

	runRuleguardEngine(c.ctx, f, c.engine, &ruleguard.RunContext{
		Debug: c.debugGroup,
		DebugPrint: func(s string) {
			fmt.Fprintln(os.Stderr, s)
		},
		Pkg:   c.ctx.Pkg,
		Types: c.ctx.TypesInfo,
		Sizes: c.ctx.SizesInfo,
		Fset:  c.ctx.FileSet,
	})
}

func runRuleguardEngine(ctx *linter.CheckerContext, f *ast.File, e *ruleguard.Engine, runCtx *ruleguard.RunContext) {
	type ruleguardReport struct {
		node    ast.Node
		message string
		fix     linter.QuickFix
	}
	var reports []ruleguardReport

	runCtx.Report = func(data *ruleguard.ReportData) {
		// TODO(quasilyte): investigate whether we should add a rule name as
		// a message prefix here.
		r := ruleguardReport{
			node:    data.Node,
			message: data.Message,
		}
		fix := data.Suggestion
		if fix != nil {
			r.fix = linter.QuickFix{
				From:        fix.From,
				To:          fix.To,
				Replacement: fix.Replacement,
			}
		}
		reports = append(reports, r)
	}

	if err := e.Run(runCtx, f); err != nil {
		// Normally this should never happen, but since
		// we don't have a better mechanism to report errors,
		// emit a warning.
		ctx.Warn(f, "execution error: %v", err)
	}

	sort.Slice(reports, func(i, j int) bool {
		return reports[i].message < reports[j].message
	})
	for _, report := range reports {
		if report.fix.Replacement != nil {
			ctx.WarnFixable(report.node, report.fix, "%s", report.message)
		} else {
			ctx.Warn(report.node, "%s", report.message)
		}
	}
}

func debugPrint(s string) {
	fmt.Println("debug:", s)
}
