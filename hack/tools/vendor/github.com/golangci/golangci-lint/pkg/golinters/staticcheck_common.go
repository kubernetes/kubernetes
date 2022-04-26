package golinters

import (
	"strings"
	"unicode"

	"golang.org/x/tools/go/analysis"
	"honnef.co/go/tools/analysis/lint"
	scconfig "honnef.co/go/tools/config"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/logutils"
)

var debugf = logutils.Debug("megacheck")

func getGoVersion(settings *config.StaticCheckSettings) string {
	var goVersion string
	if settings != nil {
		goVersion = settings.GoVersion
	}

	if goVersion != "" {
		return goVersion
	}

	// TODO: uses "1.13" for backward compatibility, but in the future (v2) must be set by using build.Default.ReleaseTags like staticcheck.
	return "1.13"
}

func setupStaticCheckAnalyzers(src []*lint.Analyzer, goVersion string, checks []string) []*analysis.Analyzer {
	var names []string
	for _, a := range src {
		names = append(names, a.Analyzer.Name)
	}

	filter := filterAnalyzerNames(names, checks)

	var ret []*analysis.Analyzer
	for _, a := range src {
		if filter[a.Analyzer.Name] {
			setAnalyzerGoVersion(a.Analyzer, goVersion)
			ret = append(ret, a.Analyzer)
		}
	}

	return ret
}

func setAnalyzerGoVersion(a *analysis.Analyzer, goVersion string) {
	if v := a.Flags.Lookup("go"); v != nil {
		if err := v.Value.Set(goVersion); err != nil {
			debugf("Failed to set go version: %s", err)
		}
	}
}

func staticCheckConfig(settings *config.StaticCheckSettings) *scconfig.Config {
	var cfg *scconfig.Config

	if settings == nil || !settings.HasConfiguration() {
		return &scconfig.Config{
			Checks:                  []string{"*"}, // override for compatibility reason. Must drop in the next major version.
			Initialisms:             scconfig.DefaultConfig.Initialisms,
			DotImportWhitelist:      scconfig.DefaultConfig.DotImportWhitelist,
			HTTPStatusCodeWhitelist: scconfig.DefaultConfig.HTTPStatusCodeWhitelist,
		}
	}

	cfg = &scconfig.Config{
		Checks:                  settings.Checks,
		Initialisms:             settings.Initialisms,
		DotImportWhitelist:      settings.DotImportWhitelist,
		HTTPStatusCodeWhitelist: settings.HTTPStatusCodeWhitelist,
	}

	if len(cfg.Checks) == 0 {
		cfg.Checks = append(cfg.Checks, "*") // override for compatibility reason. Must drop in the next major version.
	}

	if len(cfg.Initialisms) == 0 {
		cfg.Initialisms = append(cfg.Initialisms, scconfig.DefaultConfig.Initialisms...)
	}

	if len(cfg.DotImportWhitelist) == 0 {
		cfg.DotImportWhitelist = append(cfg.DotImportWhitelist, scconfig.DefaultConfig.DotImportWhitelist...)
	}

	if len(cfg.HTTPStatusCodeWhitelist) == 0 {
		cfg.HTTPStatusCodeWhitelist = append(cfg.HTTPStatusCodeWhitelist, scconfig.DefaultConfig.HTTPStatusCodeWhitelist...)
	}

	cfg.Checks = normalizeList(cfg.Checks)
	cfg.Initialisms = normalizeList(cfg.Initialisms)
	cfg.DotImportWhitelist = normalizeList(cfg.DotImportWhitelist)
	cfg.HTTPStatusCodeWhitelist = normalizeList(cfg.HTTPStatusCodeWhitelist)

	return cfg
}

// https://github.com/dominikh/go-tools/blob/9bf17c0388a65710524ba04c2d821469e639fdc2/lintcmd/lint.go#L437-L477
// nolint // Keep the original source code.
func filterAnalyzerNames(analyzers []string, checks []string) map[string]bool {
	allowedChecks := map[string]bool{}

	for _, check := range checks {
		b := true
		if len(check) > 1 && check[0] == '-' {
			b = false
			check = check[1:]
		}

		if check == "*" || check == "all" {
			// Match all
			for _, c := range analyzers {
				allowedChecks[c] = b
			}
		} else if strings.HasSuffix(check, "*") {
			// Glob
			prefix := check[:len(check)-1]
			isCat := strings.IndexFunc(prefix, func(r rune) bool { return unicode.IsNumber(r) }) == -1

			for _, a := range analyzers {
				idx := strings.IndexFunc(a, func(r rune) bool { return unicode.IsNumber(r) })
				if isCat {
					// Glob is S*, which should match S1000 but not SA1000
					cat := a[:idx]
					if prefix == cat {
						allowedChecks[a] = b
					}
				} else {
					// Glob is S1*
					if strings.HasPrefix(a, prefix) {
						allowedChecks[a] = b
					}
				}
			}
		} else {
			// Literal check name
			allowedChecks[check] = b
		}
	}
	return allowedChecks
}

// https://github.com/dominikh/go-tools/blob/9bf17c0388a65710524ba04c2d821469e639fdc2/config/config.go#L95-L116
func normalizeList(list []string) []string {
	if len(list) > 1 {
		nlist := make([]string, 0, len(list))
		nlist = append(nlist, list[0])
		for i, el := range list[1:] {
			if el != list[i] {
				nlist = append(nlist, el)
			}
		}
		list = nlist
	}

	for _, el := range list {
		if el == "inherit" {
			// This should never happen, because the default config
			// should not use "inherit"
			panic(`unresolved "inherit"`)
		}
	}

	return list
}
