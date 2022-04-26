package gomodguard

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"go/parser"
	"go/token"
	"io/ioutil"
	"os"
	"os/exec"
	"strings"

	"github.com/Masterminds/semver"

	"golang.org/x/mod/modfile"
)

const (
	goModFilename       = "go.mod"
	errReadingGoModFile = "unable to read module file %s: %w"
	errParsingGoModFile = "unable to parse module file %s: %w"
)

var (
	blockReasonNotInAllowedList = "import of package `%s` is blocked because the module is not in the " +
		"allowed modules list."
	blockReasonInBlockedList = "import of package `%s` is blocked because the module is in the " +
		"blocked modules list."
	blockReasonHasLocalReplaceDirective = "import of package `%s` is blocked because the module has a " +
		"local replace directive."
)

// BlockedVersion has a version constraint a reason why the the module version is blocked.
type BlockedVersion struct {
	Version string `yaml:"version"`
	Reason  string `yaml:"reason"`
}

// IsLintedModuleVersionBlocked returns true if a version constraint is specified and the
// linted module version matches the constraint.
func (r *BlockedVersion) IsLintedModuleVersionBlocked(lintedModuleVersion string) bool {
	if r.Version == "" {
		return false
	}

	constraint, err := semver.NewConstraint(r.Version)
	if err != nil {
		return false
	}

	version, err := semver.NewVersion(lintedModuleVersion)
	if err != nil {
		return false
	}

	meet := constraint.Check(version)

	return meet
}

// Message returns the reason why the module version is blocked.
func (r *BlockedVersion) Message(lintedModuleVersion string) string {
	var sb strings.Builder

	// Add version contraint to message.
	_, _ = fmt.Fprintf(&sb, "version `%s` is blocked because it does not meet the version constraint `%s`.",
		lintedModuleVersion, r.Version)

	if r.Reason == "" {
		return sb.String()
	}

	// Add reason to message.
	_, _ = fmt.Fprintf(&sb, " %s.", strings.TrimRight(r.Reason, "."))

	return sb.String()
}

// BlockedModule has alternative modules to use and a reason why the module is blocked.
type BlockedModule struct {
	Recommendations []string `yaml:"recommendations"`
	Reason          string   `yaml:"reason"`
}

// IsCurrentModuleARecommendation returns true if the current module is in the Recommendations list.
//
// If the current go.mod file being linted is a recommended module of a
// blocked module and it imports that blocked module, do not set as blocked.
// This could mean that the linted module is a wrapper for that blocked module.
func (r *BlockedModule) IsCurrentModuleARecommendation(currentModuleName string) bool {
	if r == nil {
		return false
	}

	for n := range r.Recommendations {
		if strings.TrimSpace(currentModuleName) == strings.TrimSpace(r.Recommendations[n]) {
			return true
		}
	}

	return false
}

// Message returns the reason why the module is blocked and a list of recommended modules if provided.
func (r *BlockedModule) Message() string {
	var sb strings.Builder

	// Add recommendations to message
	for i := range r.Recommendations {
		switch {
		case len(r.Recommendations) == 1:
			_, _ = fmt.Fprintf(&sb, "`%s` is a recommended module.", r.Recommendations[i])
		case (i+1) != len(r.Recommendations) && (i+1) == (len(r.Recommendations)-1):
			_, _ = fmt.Fprintf(&sb, "`%s` ", r.Recommendations[i])
		case (i + 1) != len(r.Recommendations):
			_, _ = fmt.Fprintf(&sb, "`%s`, ", r.Recommendations[i])
		default:
			_, _ = fmt.Fprintf(&sb, "and `%s` are recommended modules.", r.Recommendations[i])
		}
	}

	if r.Reason == "" {
		return sb.String()
	}

	// Add reason to message
	if sb.Len() == 0 {
		_, _ = fmt.Fprintf(&sb, "%s.", strings.TrimRight(r.Reason, "."))
	} else {
		_, _ = fmt.Fprintf(&sb, " %s.", strings.TrimRight(r.Reason, "."))
	}

	return sb.String()
}

// HasRecommendations returns true if the blocked package has
// recommended modules.
func (r *BlockedModule) HasRecommendations() bool {
	if r == nil {
		return false
	}

	return len(r.Recommendations) > 0
}

// BlockedVersions a list of blocked modules by a version constraint.
type BlockedVersions []map[string]BlockedVersion

// Get returns the module names that are blocked.
func (b BlockedVersions) Get() []string {
	modules := make([]string, len(b))

	for n := range b {
		for module := range b[n] {
			modules[n] = module
			break
		}
	}

	return modules
}

// GetBlockReason returns a block version if one is set for the provided linted module name.
func (b BlockedVersions) GetBlockReason(lintedModuleName string) *BlockedVersion {
	for _, blockedModule := range b {
		for blockedModuleName, blockedVersion := range blockedModule {
			if strings.TrimSpace(lintedModuleName) == strings.TrimSpace(blockedModuleName) {
				return &blockedVersion
			}
		}
	}

	return nil
}

// BlockedModules a list of blocked modules.
type BlockedModules []map[string]BlockedModule

// Get returns the module names that are blocked.
func (b BlockedModules) Get() []string {
	modules := make([]string, len(b))

	for n := range b {
		for module := range b[n] {
			modules[n] = module
			break
		}
	}

	return modules
}

// GetBlockReason returns a block module if one is set for the provided linted module name.
func (b BlockedModules) GetBlockReason(lintedModuleName string) *BlockedModule {
	for _, blockedModule := range b {
		for blockedModuleName, blockedModule := range blockedModule {
			if strings.TrimSpace(lintedModuleName) == strings.TrimSpace(blockedModuleName) {
				return &blockedModule
			}
		}
	}

	return nil
}

// Allowed is a list of modules and module
// domains that are allowed to be used.
type Allowed struct {
	Modules []string `yaml:"modules"`
	Domains []string `yaml:"domains"`
}

// IsAllowedModule returns true if the given module
// name is in the allowed modules list.
func (a *Allowed) IsAllowedModule(moduleName string) bool {
	allowedModules := a.Modules

	for i := range allowedModules {
		if strings.TrimSpace(moduleName) == strings.TrimSpace(allowedModules[i]) {
			return true
		}
	}

	return false
}

// IsAllowedModuleDomain returns true if the given modules domain is
// in the allowed module domains list.
func (a *Allowed) IsAllowedModuleDomain(moduleName string) bool {
	allowedDomains := a.Domains

	for i := range allowedDomains {
		if strings.HasPrefix(strings.TrimSpace(strings.ToLower(moduleName)),
			strings.TrimSpace(strings.ToLower(allowedDomains[i]))) {
			return true
		}
	}

	return false
}

// Blocked is a list of modules that are
// blocked and not to be used.
type Blocked struct {
	Modules                BlockedModules  `yaml:"modules"`
	Versions               BlockedVersions `yaml:"versions"`
	LocalReplaceDirectives bool            `yaml:"local_replace_directives"`
}

// Configuration of gomodguard allow and block lists.
type Configuration struct {
	Allowed Allowed `yaml:"allowed"`
	Blocked Blocked `yaml:"blocked"`
}

// Issue represents the result of one error.
type Issue struct {
	FileName   string
	LineNumber int
	Position   token.Position
	Reason     string
}

// String returns the filename, line
// number and reason of a Issue.
func (r *Issue) String() string {
	return fmt.Sprintf("%s:%d:1 %s", r.FileName, r.LineNumber, r.Reason)
}

// Processor processes Go files.
type Processor struct {
	Config                    *Configuration
	Modfile                   *modfile.File
	blockedModulesFromModFile map[string][]string
}

// NewProcessor will create a Processor to lint blocked packages.
func NewProcessor(config *Configuration) (*Processor, error) {
	goModFileBytes, err := loadGoModFile()
	if err != nil {
		return nil, fmt.Errorf(errReadingGoModFile, goModFilename, err)
	}

	modFile, err := modfile.Parse(goModFilename, goModFileBytes, nil)
	if err != nil {
		return nil, fmt.Errorf(errParsingGoModFile, goModFilename, err)
	}

	p := &Processor{
		Config:  config,
		Modfile: modFile,
	}

	p.SetBlockedModules()

	return p, nil
}

// ProcessFiles takes a string slice with file names (full paths)
// and lints them.
func (p *Processor) ProcessFiles(filenames []string) (issues []Issue) {
	for _, filename := range filenames {
		data, err := ioutil.ReadFile(filename)
		if err != nil {
			issues = append(issues, Issue{
				FileName:   filename,
				LineNumber: 0,
				Reason:     fmt.Sprintf("unable to read file, file cannot be linted (%s)", err.Error()),
			})

			continue
		}

		issues = append(issues, p.process(filename, data)...)
	}

	return issues
}

// process file imports and add lint error if blocked package is imported.
func (p *Processor) process(filename string, data []byte) (issues []Issue) {
	fileSet := token.NewFileSet()

	file, err := parser.ParseFile(fileSet, filename, data, parser.ParseComments)
	if err != nil {
		issues = append(issues, Issue{
			FileName:   filename,
			LineNumber: 0,
			Reason:     fmt.Sprintf("invalid syntax, file cannot be linted (%s)", err.Error()),
		})

		return
	}

	imports := file.Imports
	for n := range imports {
		importedPkg := strings.TrimSpace(strings.Trim(imports[n].Path.Value, "\""))

		blockReasons := p.isBlockedPackageFromModFile(importedPkg)
		if blockReasons == nil {
			continue
		}

		for _, blockReason := range blockReasons {
			issues = append(issues, p.addError(fileSet, imports[n].Pos(), blockReason))
		}
	}

	return issues
}

// addError adds an error for the file and line number for the current token.Pos
// with the given reason.
func (p *Processor) addError(fileset *token.FileSet, pos token.Pos, reason string) Issue {
	position := fileset.Position(pos)

	return Issue{
		FileName:   position.Filename,
		LineNumber: position.Line,
		Position:   position,
		Reason:     reason,
	}
}

// SetBlockedModules determines and sets which modules are blocked by reading
// the go.mod file of the module that is being linted.
//
// It works by iterating over the dependant modules specified in the require
// directive, checking if the module domain or full name is in the allowed list.
func (p *Processor) SetBlockedModules() { //nolint:gocognit,funlen
	blockedModules := make(map[string][]string, len(p.Modfile.Require))
	currentModuleName := p.Modfile.Module.Mod.Path
	lintedModules := p.Modfile.Require
	replacedModules := p.Modfile.Replace

	for i := range lintedModules {
		if lintedModules[i].Indirect {
			continue // Do not lint indirect modules.
		}

		lintedModuleName := strings.TrimSpace(lintedModules[i].Mod.Path)
		lintedModuleVersion := strings.TrimSpace(lintedModules[i].Mod.Version)

		var isAllowed bool

		switch {
		case len(p.Config.Allowed.Modules) == 0 && len(p.Config.Allowed.Domains) == 0:
			isAllowed = true
		case p.Config.Allowed.IsAllowedModuleDomain(lintedModuleName):
			isAllowed = true
		case p.Config.Allowed.IsAllowedModule(lintedModuleName):
			isAllowed = true
		default:
			isAllowed = false
		}

		blockModuleReason := p.Config.Blocked.Modules.GetBlockReason(lintedModuleName)
		blockVersionReason := p.Config.Blocked.Versions.GetBlockReason(lintedModuleName)

		if !isAllowed && blockModuleReason == nil && blockVersionReason == nil {
			blockedModules[lintedModuleName] = append(blockedModules[lintedModuleName], blockReasonNotInAllowedList)
			continue
		}

		if blockModuleReason != nil && !blockModuleReason.IsCurrentModuleARecommendation(currentModuleName) {
			blockedModules[lintedModuleName] = append(blockedModules[lintedModuleName],
				fmt.Sprintf("%s %s", blockReasonInBlockedList, blockModuleReason.Message()))
		}

		if blockVersionReason != nil && blockVersionReason.IsLintedModuleVersionBlocked(lintedModuleVersion) {
			blockedModules[lintedModuleName] = append(blockedModules[lintedModuleName],
				fmt.Sprintf("%s %s", blockReasonInBlockedList, blockVersionReason.Message(lintedModuleVersion)))
		}
	}

	// Replace directives with local paths are blocked.
	// Filesystem paths found in "replace" directives are represented by a path with an empty version.
	// https://github.com/golang/mod/blob/bc388b264a244501debfb9caea700c6dcaff10e2/module/module.go#L122-L124
	if p.Config.Blocked.LocalReplaceDirectives {
		for i := range replacedModules {
			replacedModuleOldName := strings.TrimSpace(replacedModules[i].Old.Path)
			replacedModuleNewName := strings.TrimSpace(replacedModules[i].New.Path)
			replacedModuleNewVersion := strings.TrimSpace(replacedModules[i].New.Version)

			if replacedModuleNewName != "" && replacedModuleNewVersion == "" {
				blockedModules[replacedModuleOldName] = append(blockedModules[replacedModuleOldName],
					blockReasonHasLocalReplaceDirective)
			}
		}
	}

	p.blockedModulesFromModFile = blockedModules
}

// isBlockedPackageFromModFile returns the block reason if the package is blocked.
func (p *Processor) isBlockedPackageFromModFile(packageName string) []string {
	for blockedModuleName, blockReasons := range p.blockedModulesFromModFile {
		if strings.HasPrefix(strings.TrimSpace(packageName), strings.TrimSpace(blockedModuleName)) {
			formattedReasons := make([]string, 0, len(blockReasons))

			for _, blockReason := range blockReasons {
				formattedReasons = append(formattedReasons, fmt.Sprintf(blockReason, packageName))
			}

			return formattedReasons
		}
	}

	return nil
}

func loadGoModFile() ([]byte, error) {
	cmd := exec.Command("go", "env", "-json")
	stdout, _ := cmd.StdoutPipe()
	_ = cmd.Start()

	if stdout == nil {
		return ioutil.ReadFile(goModFilename)
	}

	buf := new(bytes.Buffer)
	_, _ = buf.ReadFrom(stdout)

	goEnv := make(map[string]string)

	err := json.Unmarshal(buf.Bytes(), &goEnv)
	if err != nil {
		return ioutil.ReadFile(goModFilename)
	}

	if _, ok := goEnv["GOMOD"]; !ok {
		return ioutil.ReadFile(goModFilename)
	}

	if _, err = os.Stat(goEnv["GOMOD"]); os.IsNotExist(err) {
		return ioutil.ReadFile(goModFilename)
	}

	if goEnv["GOMOD"] == "/dev/null" {
		return nil, errors.New("current working directory must have a go.mod file")
	}

	return ioutil.ReadFile(goEnv["GOMOD"])
}
