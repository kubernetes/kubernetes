/*
Copyright 2016 Google Inc. All Rights Reserved.
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
// Buildozer is a tool for programatically editing BUILD files.

package edit

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"

	apipb "github.com/bazelbuild/buildtools/api_proto"
	"github.com/bazelbuild/buildtools/build"
	"github.com/bazelbuild/buildtools/file"
	"github.com/golang/protobuf/proto"
)

// Options represents choices about how buildozer should behave.
type Options struct {
	Stdout            bool     // write changed BUILD file to stdout
	Buildifier        string   // path to buildifier binary
	Parallelism       int      // number of cores to use for concurrent actions
	NumIO             int      // number of concurrent actions
	CommandsFile      string   // file name to read commands from, use '-' for stdin (format:|-separated command line arguments to buildozer, excluding flags
	KeepGoing         bool     // apply all commands, even if there are failures
	FilterRuleTypes   []string // list of rule types to change, empty means all
	PreferEOLComments bool     // when adding a new comment, put it on the same line if possible
	RootDir           string   // If present, use this folder rather than $PWD to find the root dir
	Quiet             bool     // suppress informational messages.
	EditVariables     bool     // for attributes that simply assign a variable (e.g. hdrs = LIB_HDRS), edit the build variable instead of appending to the attribute.
	IsPrintingProto   bool     // output serialized devtools.buildozer.Output protos instead of human-readable strings
}

// Opts represents the options to be used by buildozer, and can be overriden before calling Buildozer.
var Opts = Options{NumIO: 200, PreferEOLComments: true}

// Usage is a user-overriden func to print the program usage.
var Usage = func() {}

var fileModified = false // set to true when a file has been fixed

const stdinPackageName = "-" // the special package name to represent stdin

// CmdEnvironment stores the information the commands below have access to.
type CmdEnvironment struct {
	File   *build.File                  // the AST
	Rule   *build.Rule                  // the rule to modify
	Vars   map[string]*build.BinaryExpr // global variables set in the build file
	Pkg    string                       // the full package name
	Args   []string                     // the command-line arguments
	output *apipb.Output_Record         // output proto, stores whatever a command wants to print
}

// The cmdXXX functions implement the various commands.

func cmdAdd(env CmdEnvironment) (*build.File, error) {
	attr := env.Args[0]
	for _, val := range env.Args[1:] {
		if IsIntList(attr) {
			AddValueToListAttribute(env.Rule, attr, env.Pkg, &build.LiteralExpr{Token: val}, &env.Vars)
			continue
		}
		strVal := &build.StringExpr{Value: ShortenLabel(val, env.Pkg)}
		AddValueToListAttribute(env.Rule, attr, env.Pkg, strVal, &env.Vars)
	}
	return env.File, nil
}

func cmdComment(env CmdEnvironment) (*build.File, error) {
	// The comment string is always the last argument in the list.
	str := env.Args[len(env.Args)-1]
	str = strings.Replace(str, "\\n", "\n", -1)
	// Multiline comments should go on a separate line.
	fullLine := !Opts.PreferEOLComments || strings.Contains(str, "\n")
	str = strings.Replace("# "+str, "\n", "\n# ", -1)
	comment := []build.Comment{build.Comment{Token: str}}

	// The comment might be attached to a rule, an attribute, or a value in a list,
	// depending on how many arguments are passed.
	switch len(env.Args) {
	case 1: // Attach to a rule
		env.Rule.Call.Comments.Before = comment
	case 2: // Attach to an attribute
		if attr := env.Rule.AttrDefn(env.Args[0]); attr != nil {
			if fullLine {
				attr.X.Comment().Before = comment
			} else {
				attr.Y.Comment().Suffix = comment
			}
		}
	case 3: // Attach to a specific value in a list
		if attr := env.Rule.Attr(env.Args[0]); attr != nil {
			if expr := ListFind(attr, env.Args[1], env.Pkg); expr != nil {
				if fullLine {
					expr.Comments.Before = comment
				} else {
					expr.Comments.Suffix = comment
				}
			}
		}
	default:
		panic("cmdComment")
	}
	return env.File, nil
}

// commentsText concatenates comments into a single line.
func commentsText(comments []build.Comment) string {
	var segments []string
	for _, comment := range comments {
		token := comment.Token
		if strings.HasPrefix(token, "#") {
			token = token[1:]
		}
		segments = append(segments, strings.TrimSpace(token))
	}
	return strings.Replace(strings.Join(segments, " "), "\n", " ", -1)
}

func cmdPrintComment(env CmdEnvironment) (*build.File, error) {
	attrError := func() error {
		return fmt.Errorf("rule \"//%s:%s\" has no attribute \"%s\"", env.Pkg, env.Rule.Name(), env.Args[0])
	}

	switch len(env.Args) {
	case 0: // Print rule comment.
		env.output.Fields = []*apipb.Output_Record_Field{
			&apipb.Output_Record_Field{Value: &apipb.Output_Record_Field_Text{commentsText(env.Rule.Call.Comments.Before)}},
		}
	case 1: // Print attribute comment.
		attr := env.Rule.AttrDefn(env.Args[0])
		if attr == nil {
			return nil, attrError()
		}
		comments := append(attr.Before, attr.Suffix...)
		env.output.Fields = []*apipb.Output_Record_Field{
			&apipb.Output_Record_Field{Value: &apipb.Output_Record_Field_Text{commentsText(comments)}},
		}
	case 2: // Print comment of a specific value in a list.
		attr := env.Rule.Attr(env.Args[0])
		if attr == nil {
			return nil, attrError()
		}
		value := env.Args[1]
		expr := ListFind(attr, value, env.Pkg)
		if expr == nil {
			return nil, fmt.Errorf("attribute \"%s\" has no value \"%s\"", env.Args[0], value)
		}
		comments := append(expr.Comments.Before, expr.Comments.Suffix...)
		env.output.Fields = []*apipb.Output_Record_Field{
			&apipb.Output_Record_Field{Value: &apipb.Output_Record_Field_Text{commentsText(comments)}},
		}
	default:
		panic("cmdPrintComment")
	}
	return nil, nil
}

func cmdDelete(env CmdEnvironment) (*build.File, error) {
	return DeleteRule(env.File, env.Rule), nil
}

func cmdMove(env CmdEnvironment) (*build.File, error) {
	oldAttr := env.Args[0]
	newAttr := env.Args[1]
	if len(env.Args) == 3 && env.Args[2] == "*" {
		if err := MoveAllListAttributeValues(env.Rule, oldAttr, newAttr, env.Pkg, &env.Vars); err != nil {
			return nil, err
		}
		return env.File, nil
	}
	fixed := false
	for _, val := range env.Args[2:] {
		if deleted := ListAttributeDelete(env.Rule, oldAttr, val, env.Pkg); deleted != nil {
			AddValueToListAttribute(env.Rule, newAttr, env.Pkg, deleted, &env.Vars)
			fixed = true
		}
	}
	if fixed {
		return env.File, nil
	}
	return nil, nil
}

func cmdNew(env CmdEnvironment) (*build.File, error) {
	kind := env.Args[0]
	name := env.Args[1]
	addAtEOF, insertionIndex, err := findInsertionIndex(env)
	if err != nil {
		return nil, err
	}

	if FindRuleByName(env.File, name) != nil {
		return nil, fmt.Errorf("rule '%s' already exists", name)
	}

	call := &build.CallExpr{X: &build.LiteralExpr{Token: kind}}
	rule := &build.Rule{Call: call}
	rule.SetAttr("name", &build.StringExpr{Value: name})

	if addAtEOF {
		env.File.Stmt = InsertAfterLastOfSameKind(env.File.Stmt, rule.Call)
	} else {
		env.File.Stmt = InsertAfter(insertionIndex, env.File.Stmt, call)
	}
	return env.File, nil
}

// findInsertionIndex is used by cmdNew to find the place at which to insert the new rule.
func findInsertionIndex(env CmdEnvironment) (bool, int, error) {
	if len(env.Args) < 4 {
		return true, 0, nil
	}

	relativeToRuleName := env.Args[3]
	ruleIdx := IndexOfRuleByName(env.File, relativeToRuleName)
	if ruleIdx == -1 {
		return true, 0, nil
	}

	switch env.Args[2] {
	case "before":
		return false, ruleIdx - 1, nil
	case "after":
		return false, ruleIdx, nil
	default:
		return true, 0, fmt.Errorf("Unknown relative operator '%s'; allowed: 'before', 'after'", env.Args[1])
	}
}

func cmdNewLoad(env CmdEnvironment) (*build.File, error) {
	env.File.Stmt = InsertLoad(env.File.Stmt, env.Args)
	return env.File, nil
}

func cmdPrint(env CmdEnvironment) (*build.File, error) {
	format := env.Args
	if len(format) == 0 {
		format = []string{"name", "kind"}
	}
	fields := make([]*apipb.Output_Record_Field, len(format))

	for i, str := range format {
		value := env.Rule.Attr(str)
		if str == "kind" {
			fields[i] = &apipb.Output_Record_Field{Value: &apipb.Output_Record_Field_Text{env.Rule.Kind()}}
		} else if str == "label" {
			if env.Rule.Attr("name") != nil {
				fields[i] = &apipb.Output_Record_Field{Value: &apipb.Output_Record_Field_Text{fmt.Sprintf("//%s:%s", env.Pkg, env.Rule.Name())}}
			} else {
				return nil, nil
			}
		} else if str == "rule" {
			fields[i] = &apipb.Output_Record_Field{
				Value: &apipb.Output_Record_Field_Text{build.FormatString(env.Rule.Call)},
			}
		} else if str == "startline" {
			fields[i] = &apipb.Output_Record_Field{Value: &apipb.Output_Record_Field_Number{int32(env.Rule.Call.ListStart.Line)}}
		} else if str == "endline" {
			fields[i] = &apipb.Output_Record_Field{Value: &apipb.Output_Record_Field_Number{int32(env.Rule.Call.End.Pos.Line)}}
		} else if value == nil {
			fmt.Fprintf(os.Stderr, "rule \"//%s:%s\" has no attribute \"%s\"\n",
				env.Pkg, env.Rule.Name(), str)
			fields[i] = &apipb.Output_Record_Field{Value: &apipb.Output_Record_Field_Error{Error: apipb.Output_Record_Field_MISSING}}
		} else if lit, ok := value.(*build.LiteralExpr); ok {
			fields[i] = &apipb.Output_Record_Field{Value: &apipb.Output_Record_Field_Text{lit.Token}}
		} else if string, ok := value.(*build.StringExpr); ok {
			fields[i] = &apipb.Output_Record_Field{
				Value:             &apipb.Output_Record_Field_Text{string.Value},
				QuoteWhenPrinting: true,
			}
		} else if strList := env.Rule.AttrStrings(str); strList != nil {
			fields[i] = &apipb.Output_Record_Field{Value: &apipb.Output_Record_Field_List{List: &apipb.RepeatedString{Strings: strList}}}
		} else {
			// Some other Expr we haven't listed above. Just print it.
			fields[i] = &apipb.Output_Record_Field{Value: &apipb.Output_Record_Field_Text{build.FormatString(value)}}
		}
	}

	env.output.Fields = fields
	return nil, nil
}

func attrKeysForPattern(rule *build.Rule, pattern string) []string {
	if pattern == "*" {
		return rule.AttrKeys()
	}
	return []string{pattern}
}

func cmdRemove(env CmdEnvironment) (*build.File, error) {
	if len(env.Args) == 1 { // Remove the attribute
		if env.Rule.DelAttr(env.Args[0]) != nil {
			return env.File, nil
		}
	} else { // Remove values in the attribute.
		fixed := false
		for _, key := range attrKeysForPattern(env.Rule, env.Args[0]) {
			for _, val := range env.Args[1:] {
				ListAttributeDelete(env.Rule, key, val, env.Pkg)
				fixed = true
			}
		}
		if fixed {
			return env.File, nil
		}
	}
	return nil, nil
}

func cmdRename(env CmdEnvironment) (*build.File, error) {
	oldAttr := env.Args[0]
	newAttr := env.Args[1]
	if err := RenameAttribute(env.Rule, oldAttr, newAttr); err != nil {
		return nil, err
	}
	return env.File, nil
}

func cmdReplace(env CmdEnvironment) (*build.File, error) {
	oldV := env.Args[1]
	newV := env.Args[2]
	for _, key := range attrKeysForPattern(env.Rule, env.Args[0]) {
		attr := env.Rule.Attr(key)
		if e, ok := attr.(*build.StringExpr); ok {
			if LabelsEqual(e.Value, oldV, env.Pkg) {
				env.Rule.SetAttr(key, getAttrValueExpr(key, []string{newV}))
			}
		} else {
			ListReplace(attr, oldV, newV, env.Pkg)
		}
	}
	return env.File, nil
}

func cmdSet(env CmdEnvironment) (*build.File, error) {
	attr := env.Args[0]
	args := env.Args[1:]
	if attr == "kind" {
		env.Rule.SetKind(args[0])
	} else {
		env.Rule.SetAttr(attr, getAttrValueExpr(attr, args))
	}
	return env.File, nil
}

func cmdSetIfAbsent(env CmdEnvironment) (*build.File, error) {
	attr := env.Args[0]
	args := env.Args[1:]
	if attr == "kind" {
		return nil, fmt.Errorf("setting 'kind' is not allowed for set_if_absent. Got %s", env.Args)
	}
	if env.Rule.Attr(attr) == nil {
		env.Rule.SetAttr(attr, getAttrValueExpr(attr, args))
	}
	return env.File, nil
}

func getAttrValueExpr(attr string, args []string) build.Expr {
	switch {
	case attr == "kind":
		return nil
	case IsIntList(attr):
		var list []build.Expr
		for _, i := range args {
			list = append(list, &build.LiteralExpr{Token: i})
		}
		return &build.ListExpr{List: list}
	case IsList(attr) && !(len(args) == 1 && strings.HasPrefix(args[0], "glob(")):
		var list []build.Expr
		for _, i := range args {
			list = append(list, &build.StringExpr{Value: i})
		}
		return &build.ListExpr{List: list}
	case IsString(attr):
		return &build.StringExpr{Value: args[0]}
	default:
		return &build.LiteralExpr{Token: args[0]}
	}
}

func cmdCopy(env CmdEnvironment) (*build.File, error) {
	attrName := env.Args[0]
	from := env.Args[1]

	return copyAttributeBetweenRules(env, attrName, from)
}

func cmdCopyNoOverwrite(env CmdEnvironment) (*build.File, error) {
	attrName := env.Args[0]
	from := env.Args[1]

	if env.Rule.Attr(attrName) != nil {
		return env.File, nil
	}

	return copyAttributeBetweenRules(env, attrName, from)
}

func copyAttributeBetweenRules(env CmdEnvironment, attrName string, from string) (*build.File, error) {
	fromRule := FindRuleByName(env.File, from)
	if fromRule == nil {
		return nil, fmt.Errorf("could not find rule '%s'", from)
	}
	attr := fromRule.Attr(attrName)
	if attr == nil {
		return nil, fmt.Errorf("rule '%s' does not have attribute '%s'", from, attrName)
	}

	ast, err := build.Parse("" /* filename */, []byte(build.FormatString(attr)))
	if err != nil {
		return nil, fmt.Errorf("could not parse attribute value %v", build.FormatString(attr))
	}

	env.Rule.SetAttr(attrName, ast.Stmt[0])
	return env.File, nil
}

func cmdFix(env CmdEnvironment) (*build.File, error) {
	// Fix the whole file
	if env.Rule.Kind() == "package" {
		return FixFile(env.File, env.Pkg, env.Args), nil
	}
	// Fix a specific rule
	return FixRule(env.File, env.Pkg, env.Rule, env.Args), nil
}

// CommandInfo provides a command function and info on incoming arguments.
type CommandInfo struct {
	Fn       func(CmdEnvironment) (*build.File, error)
	MinArg   int
	MaxArg   int
	Template string
}

// AllCommands associates the command names with their function and number
// of arguments.
var AllCommands = map[string]CommandInfo{
	"add":               {cmdAdd, 2, -1, "<attr> <value(s)>"},
	"new_load":          {cmdNewLoad, 1, -1, "<path> <symbol(s)>"},
	"comment":           {cmdComment, 1, 3, "<attr>? <value>? <comment>"},
	"print_comment":     {cmdPrintComment, 0, 2, "<attr>? <value>?"},
	"delete":            {cmdDelete, 0, 0, ""},
	"fix":               {cmdFix, 0, -1, "<fix(es)>?"},
	"move":              {cmdMove, 3, -1, "<old_attr> <new_attr> <value(s)>"},
	"new":               {cmdNew, 2, 4, "<rule_kind> <rule_name> [(before|after) <relative_rule_name>]"},
	"print":             {cmdPrint, 0, -1, "<attribute(s)>"},
	"remove":            {cmdRemove, 1, -1, "<attr> <value(s)>"},
	"rename":            {cmdRename, 2, 2, "<old_attr> <new_attr>"},
	"replace":           {cmdReplace, 3, 3, "<attr> <old_value> <new_value>"},
	"set":               {cmdSet, 2, -1, "<attr> <value(s)>"},
	"set_if_absent":     {cmdSetIfAbsent, 2, -1, "<attr> <value(s)>"},
	"copy":              {cmdCopy, 2, 2, "<attr> <from_rule>"},
	"copy_no_overwrite": {cmdCopyNoOverwrite, 2, 2, "<attr> <from_rule>"},
}

func expandTargets(f *build.File, rule string) ([]*build.Rule, error) {
	if r := FindRuleByName(f, rule); r != nil {
		return []*build.Rule{r}, nil
	} else if r := FindExportedFile(f, rule); r != nil {
		return []*build.Rule{r}, nil
	} else if rule == "all" || rule == "*" {
		// "all" is a valid name, it is a wildcard only if no such rule is found.
		return f.Rules(""), nil
	} else if strings.HasPrefix(rule, "%") {
		// "%java_library" will match all java_library functions in the package
		// "%<LINENUM>" will match the rule which begins at LINENUM.
		// This is for convenience, "%" is not a valid character in bazel targets.
		kind := rule[1:]
		if linenum, err := strconv.Atoi(kind); err == nil {
			if r := f.RuleAt(linenum); r != nil {
				return []*build.Rule{r}, nil
			}
		} else {
			return f.Rules(kind), nil
		}
	}
	return nil, fmt.Errorf("rule '%s' not found", rule)
}

func filterRules(rules []*build.Rule) (result []*build.Rule) {
	if len(Opts.FilterRuleTypes) == 0 {
		return rules
	}
	for _, rule := range rules {
		acceptableType := false
		for _, filterType := range Opts.FilterRuleTypes {
			if rule.Kind() == filterType {
				acceptableType = true
				break
			}
		}
		if acceptableType || rule.Kind() == "package" {
			result = append(result, rule)
		}
	}
	return
}

// command contains a list of tokens that describe a buildozer command.
type command struct {
	tokens []string
}

// checkCommandUsage checks the number of argument of a command.
// It prints an error and usage when it is not valid.
func checkCommandUsage(name string, cmd CommandInfo, count int) {
	if count >= cmd.MinArg && (cmd.MaxArg == -1 || count <= cmd.MaxArg) {
		return
	}

	if count < cmd.MinArg {
		fmt.Fprintf(os.Stderr, "Too few arguments for command '%s', expected at least %d.\n",
			name, cmd.MinArg)
	} else {
		fmt.Fprintf(os.Stderr, "Too many arguments for command '%s', expected at most %d.\n",
			name, cmd.MaxArg)
	}
	Usage()
}

// Match text that only contains spaces if they're escaped with '\'.
var spaceRegex = regexp.MustCompile(`(\\ |[^ ])+`)

// SplitOnSpaces behaves like strings.Fields, except that spaces can be escaped.
// " some dummy\\ string" -> ["some", "dummy string"]
func SplitOnSpaces(input string) []string {
	result := spaceRegex.FindAllString(input, -1)
	for i, s := range result {
		result[i] = strings.Replace(s, `\ `, " ", -1)
	}
	return result
}

// parseCommands parses commands and targets they should be applied on from
// a list of arguments.
// Each argument can be either:
// - a command (as defined by AllCommands) and its parameters, separated by
//   whitespace
// - a target all commands that are parsed during one call to parseCommands
//   should be applied on
func parseCommands(args []string) (commands []command, targets []string) {
	for _, arg := range args {
		commandTokens := SplitOnSpaces(arg)
		cmd, found := AllCommands[commandTokens[0]]
		if found {
			checkCommandUsage(commandTokens[0], cmd, len(commandTokens)-1)
			commands = append(commands, command{commandTokens})
		} else {
			targets = append(targets, arg)
		}
	}
	return
}

// commandsForTarget contains commands to be executed on the given target.
type commandsForTarget struct {
	target   string
	commands []command
}

// commandsForFile contains the file name and all commands that should be
// applied on that file, indexed by their target.
type commandsForFile struct {
	file     string
	commands []commandsForTarget
}

// commandError returns an error that formats 'err' in the context of the
// commands to be executed on the given target.
func commandError(commands []command, target string, err error) error {
	return fmt.Errorf("error while executing commands %s on target %s: %s", commands, target, err)
}

// rewriteResult contains the outcome of applying fixes to a single file.
type rewriteResult struct {
	file     string
	errs     []error
	modified bool
	records  []*apipb.Output_Record
}

// getGlobalVariables returns the global variable assignments in the provided list of expressions.
// That is, for each variable assignment of the form
//   a = v
// vars["a"] will contain the BinaryExpr whose Y value is the assignment "a = v".
func getGlobalVariables(exprs []build.Expr) (vars map[string]*build.BinaryExpr) {
	vars = make(map[string]*build.BinaryExpr)
	for _, expr := range exprs {
		if binExpr, ok := expr.(*build.BinaryExpr); ok {
			if binExpr.Op != "=" {
				continue
			}
			if lhs, ok := binExpr.X.(*build.LiteralExpr); ok {
				vars[lhs.Token] = binExpr
			}
		}
	}
	return vars
}

// When checking the filesystem, we need to look for any of the
// possible buildFileNames. For historical reasons, the
// parts of the tool that generate paths that we may want to examine
// continue to assume that build files are all named "BUILD".
var buildFileNames = [...]string{"BUILD.bazel", "BUILD", "BUCK"}
var buildFileNamesSet = map[string]bool{
	"BUILD.bazel": true,
	"BUILD":       true,
	"BUCK":        true,
}

// rewrite parses the BUILD file for the given file, transforms the AST,
// and write the changes back in the file (or on stdout).
func rewrite(commandsForFile commandsForFile) *rewriteResult {
	name := commandsForFile.file
	var data []byte
	var err error
	var fi os.FileInfo
	records := []*apipb.Output_Record{}
	if name == stdinPackageName { // read on stdin
		data, err = ioutil.ReadAll(os.Stdin)
		if err != nil {
			return &rewriteResult{file: name, errs: []error{err}}
		}
	} else {
		origName := name
		for _, suffix := range buildFileNames {
			if strings.HasSuffix(name, "/"+suffix) {
				name = strings.TrimSuffix(name, suffix)
				break
			}
		}
		for _, suffix := range buildFileNames {
			name = name + suffix
			data, fi, err = file.ReadFile(name)
			if err == nil {
				break
			}
			name = strings.TrimSuffix(name, suffix)
		}
		if err != nil {
			data, fi, err = file.ReadFile(name)
		}
		if err != nil {
			err = errors.New("file not found or not readable")
			return &rewriteResult{file: origName, errs: []error{err}}
		}
	}

	f, err := build.Parse(name, data)
	if err != nil {
		return &rewriteResult{file: name, errs: []error{err}}
	}

	vars := map[string]*build.BinaryExpr{}
	if Opts.EditVariables {
		vars = getGlobalVariables(f.Stmt)
	}
	var errs []error
	changed := false
	for _, commands := range commandsForFile.commands {
		target := commands.target
		commands := commands.commands
		_, absPkg, rule := InterpretLabelForWorkspaceLocation(Opts.RootDir, target)
		_, pkg, _ := ParseLabel(target)
		if pkg == stdinPackageName { // Special-case: This is already absolute
			absPkg = stdinPackageName
		}

		targets, err := expandTargets(f, rule)
		if err != nil {
			cerr := commandError(commands, target, err)
			errs = append(errs, cerr)
			if !Opts.KeepGoing {
				return &rewriteResult{file: name, errs: errs, records: records}

			}
		}
		targets = filterRules(targets)
		for _, cmd := range commands {
			for _, r := range targets {
				cmdInfo := AllCommands[cmd.tokens[0]]
				record := &apipb.Output_Record{}
				newf, err := cmdInfo.Fn(CmdEnvironment{f, r, vars, absPkg, cmd.tokens[1:], record})
				if len(record.Fields) != 0 {
					records = append(records, record)
				}
				if err != nil {
					cerr := commandError([]command{cmd}, target, err)
					if Opts.KeepGoing {
						errs = append(errs, cerr)
					} else {
						return &rewriteResult{file: name, errs: []error{cerr}, records: records}
					}
				}
				if newf != nil {
					changed = true
					f = newf
				}
			}
		}
	}
	if !changed {
		return &rewriteResult{file: name, errs: errs, records: records}
	}
	f = RemoveEmptyPackage(f)
	ndata, err := runBuildifier(f)
	if err != nil {
		return &rewriteResult{file: name, errs: []error{fmt.Errorf("running buildifier: %v", err)}, records: records}
	}

	if Opts.Stdout || name == stdinPackageName {
		os.Stdout.Write(ndata)
		return &rewriteResult{file: name, errs: errs, records: records}
	}

	if bytes.Equal(data, ndata) {
		return &rewriteResult{file: name, errs: errs, records: records}
	}

	if err := EditFile(fi, name); err != nil {
		return &rewriteResult{file: name, errs: []error{err}, records: records}
	}

	if err := file.WriteFile(name, ndata); err != nil {
		return &rewriteResult{file: name, errs: []error{err}, records: records}
	}

	fileModified = true
	return &rewriteResult{file: name, errs: errs, modified: true, records: records}
}

// EditFile is a function that does any prework needed before editing a file.
// e.g. "checking out for write" from a locking source control repo.
var EditFile = func(fi os.FileInfo, name string) error {
	return nil
}

// runBuildifier formats the build file f.
// Runs Opts.Buildifier if it's non-empty, otherwise uses built-in formatter.
// Opts.Buildifier is useful to force consistency with other tools that call Buildifier.
func runBuildifier(f *build.File) ([]byte, error) {
	if Opts.Buildifier == "" {
		build.Rewrite(f, nil)
		return build.Format(f), nil
	}

	cmd := exec.Command(Opts.Buildifier)
	data := build.Format(f)
	cmd.Stdin = bytes.NewBuffer(data)
	stdout := bytes.NewBuffer(nil)
	stderr := bytes.NewBuffer(nil)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	err := cmd.Run()
	if stderr.Len() > 0 {
		return nil, fmt.Errorf("%s", stderr.Bytes())
	}
	if err != nil {
		return nil, err
	}
	return stdout.Bytes(), nil
}

// Given a target, whose package may contain a trailing "/...", returns all
// extisting BUILD file paths which match the package.
func targetExpressionToBuildFiles(target string) []string {
	file, _, _ := InterpretLabelForWorkspaceLocation(Opts.RootDir, target)
	if Opts.RootDir == "" {
		var err error
		if file, err = filepath.Abs(file); err != nil {
			fmt.Printf("Cannot make path absolute: %s\n", err.Error())
			os.Exit(1)
		}
	}

	if !strings.HasSuffix(file, "/.../BUILD") {
		return []string{file}
	}

	var buildFiles []string
	searchDirs := []string{strings.TrimSuffix(file, "/.../BUILD")}
	for len(searchDirs) != 0 {
		lastIndex := len(searchDirs) - 1
		dir := searchDirs[lastIndex]
		searchDirs = searchDirs[:lastIndex]

		dirFiles, err := ioutil.ReadDir(dir)
		if err != nil {
			continue
		}

		for _, dirFile := range dirFiles {
			if dirFile.IsDir() {
				searchDirs = append(searchDirs, path.Join(dir, dirFile.Name()))
			} else if _, ok := buildFileNamesSet[dirFile.Name()]; ok {
				buildFiles = append(buildFiles, path.Join(dir, dirFile.Name()))
			}
		}
	}

	return buildFiles
}

// appendCommands adds the given commands to be applied to each of the given targets
// via the commandMap.
func appendCommands(commandMap map[string][]commandsForTarget, args []string) {
	commands, targets := parseCommands(args)
	for _, target := range targets {
		if strings.HasSuffix(target, "/BUILD") {
			target = strings.TrimSuffix(target, "/BUILD") + ":__pkg__"
		}
		var buildFiles []string
		_, pkg, _ := ParseLabel(target)
		if pkg == stdinPackageName {
			buildFiles = []string{stdinPackageName}
		} else {
			buildFiles = targetExpressionToBuildFiles(target)
		}

		for _, file := range buildFiles {
			commandMap[file] = append(commandMap[file], commandsForTarget{target, commands})
		}
	}
}

func appendCommandsFromFile(commandsByFile map[string][]commandsForTarget, fileName string) {
	var reader io.Reader
	if Opts.CommandsFile == stdinPackageName {
		reader = os.Stdin
	} else {
		rc := file.OpenReadFile(Opts.CommandsFile)
		reader = rc
		defer rc.Close()
	}
	scanner := bufio.NewScanner(reader)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		args := strings.Split(line, "|")
		appendCommands(commandsByFile, args)
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "Error while reading commands file: %v", scanner.Err())
	}
}

func printRecord(writer io.Writer, record *apipb.Output_Record) {
	fields := record.Fields
	line := make([]string, len(fields))
	for i, field := range fields {
		switch value := field.Value.(type) {
		case *apipb.Output_Record_Field_Text:
			if field.QuoteWhenPrinting && strings.ContainsRune(value.Text, ' ') {
				line[i] = fmt.Sprintf("%q", value.Text)
			} else {
				line[i] = value.Text
			}
			break
		case *apipb.Output_Record_Field_Number:
			line[i] = strconv.Itoa(int(value.Number))
			break
		case *apipb.Output_Record_Field_Error:
			switch value.Error {
			case apipb.Output_Record_Field_UNKNOWN:
				line[i] = "(unknown)"
				break
			case apipb.Output_Record_Field_MISSING:
				line[i] = "(missing)"
				break
			}
			break
		case *apipb.Output_Record_Field_List:
			line[i] = fmt.Sprintf("[%s]", strings.Join(value.List.Strings, " "))
			break
		}
	}

	fmt.Fprint(writer, strings.Join(line, " ")+"\n")
}

// Buildozer loops over all arguments on the command line fixing BUILD files.
func Buildozer(args []string) int {
	commandsByFile := make(map[string][]commandsForTarget)
	if Opts.CommandsFile != "" {
		appendCommandsFromFile(commandsByFile, Opts.CommandsFile)
	} else {
		if len(args) == 0 {
			Usage()
		}
		appendCommands(commandsByFile, args)
	}

	numFiles := len(commandsByFile)
	if Opts.Parallelism > 0 {
		runtime.GOMAXPROCS(Opts.Parallelism)
	}
	results := make(chan *rewriteResult, numFiles)
	data := make(chan commandsForFile)

	for i := 0; i < Opts.NumIO; i++ {
		go func(results chan *rewriteResult, data chan commandsForFile) {
			for commandsForFile := range data {
				results <- rewrite(commandsForFile)
			}
		}(results, data)
	}

	for file, commands := range commandsByFile {
		data <- commandsForFile{file, commands}
	}
	close(data)
	records := []*apipb.Output_Record{}
	hasErrors := false
	for i := 0; i < numFiles; i++ {
		fileResults := <-results
		if fileResults == nil {
			continue
		}
		hasErrors = hasErrors || len(fileResults.errs) > 0
		for _, err := range fileResults.errs {
			fmt.Fprintf(os.Stderr, "%s: %s\n", fileResults.file, err)
		}
		if fileResults.modified && !Opts.Quiet {
			fmt.Fprintf(os.Stderr, "fixed %s\n", fileResults.file)
		}
		if fileResults.records != nil {
			records = append(records, fileResults.records...)
		}
	}

	if Opts.IsPrintingProto {
		data, err := proto.Marshal(&apipb.Output{Records: records})
		if err != nil {
			log.Fatal("marshaling error: ", err)
		}
		fmt.Fprintf(os.Stdout, "%s", data)
	} else {
		for _, record := range records {
			printRecord(os.Stdout, record)
		}
	}

	if hasErrors {
		return 2
	}
	if !fileModified && !Opts.Stdout {
		return 3
	}
	return 0
}
