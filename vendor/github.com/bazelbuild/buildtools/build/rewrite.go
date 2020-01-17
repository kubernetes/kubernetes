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
// Rewriting of high-level (not purely syntactic) BUILD constructs.

package build

import (
	"github.com/bazelbuild/buildtools/tables"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// For debugging: flag to disable certain rewrites.
var DisableRewrites []string

// disabled reports whether the named rewrite is disabled.
func disabled(name string) bool {
	for _, x := range DisableRewrites {
		if name == x {
			return true
		}
	}
	return false
}

// For debugging: allow sorting of these lists even with sorting otherwise disabled.
var AllowSort []string

// allowedSort reports whether sorting is allowed in the named context.
func allowedSort(name string) bool {
	for _, x := range AllowSort {
		if name == x {
			return true
		}
	}
	return false
}

// Rewrite applies the high-level Buildifier rewrites to f, modifying it in place.
// If info is non-nil, Rewrite updates it with information about the rewrite.
func Rewrite(f *File, info *RewriteInfo) {
	// Allocate an info so that helpers can assume it's there.
	if info == nil {
		info = new(RewriteInfo)
	}

	for _, r := range rewrites {
		if !disabled(r.name) {
			if f.Type&r.scope != 0 {
				r.fn(f, info)
			}
		}
	}
}

// RewriteInfo collects information about what Rewrite did.
type RewriteInfo struct {
	EditLabel        int      // number of label strings edited
	NameCall         int      // number of calls with argument names added
	SortCall         int      // number of call argument lists sorted
	SortStringList   int      // number of string lists sorted
	UnsafeSort       int      // number of unsafe string lists sorted
	SortLoad         int      // number of load argument lists sorted
	FormatDocstrings int      // number of reindented docstrings
	ReorderArguments int      // number of reordered function call arguments
	EditOctal        int      // number of edited octals
	Log              []string // log entries - may change
}

// Stats returns a map with statistics about applied rewrites
func (info *RewriteInfo) Stats() map[string]int {
	return map[string]int{
		"label":            info.EditLabel,
		"callname":         info.NameCall,
		"callsort":         info.SortCall,
		"listsort":         info.SortStringList,
		"unsafesort":       info.UnsafeSort,
		"sortload":         info.SortLoad,
		"formatdocstrings": info.FormatDocstrings,
		"reorderarguments": info.ReorderArguments,
		"editoctal":        info.EditOctal,
	}
}

// Each rewrite function can be either applied for BUILD files, other files (such as .bzl),
// or all files.
const (
	scopeDefault = TypeDefault | TypeBzl     // .bzl and generic Starlark files
	scopeBuild   = TypeBuild | TypeWorkspace // BUILD and WORKSPACE files
	scopeBoth    = scopeDefault | scopeBuild
)

// rewrites is the list of all Buildifier rewrites, in the order in which they are applied.
// The order here matters: for example, label canonicalization must happen
// before sorting lists of strings.
var rewrites = []struct {
	name  string
	fn    func(*File, *RewriteInfo)
	scope FileType
}{
	{"callsort", sortCallArgs, scopeBuild},
	{"label", fixLabels, scopeBuild},
	{"listsort", sortStringLists, scopeBoth},
	{"multiplus", fixMultilinePlus, scopeBuild},
	{"loadsort", sortAllLoadArgs, scopeBoth},
	{"formatdocstrings", formatDocstrings, scopeBoth},
	{"reorderarguments", reorderArguments, scopeBoth},
	{"editoctal", editOctals, scopeBoth},
}

// DisableLoadSortForBuildFiles disables the loadsort transformation for BUILD files.
// This is a temporary function for backward compatibility, can be called if there's plenty of
// already formatted BUILD files that shouldn't be changed by the transformation.
func DisableLoadSortForBuildFiles() {
	for i := range rewrites {
		if rewrites[i].name == "loadsort" {
			rewrites[i].scope = scopeDefault
			break
		}
	}
}

// leaveAlone reports whether any of the nodes on the stack are marked
// with a comment containing "buildifier: leave-alone".
func leaveAlone(stk []Expr, final Expr) bool {
	for _, x := range stk {
		if leaveAlone1(x) {
			return true
		}
	}
	if final != nil && leaveAlone1(final) {
		return true
	}
	return false
}

// hasComment reports whether x is marked with a comment that
// after being converted to lower case, contains the specified text.
func hasComment(x Expr, text string) bool {
	for _, com := range x.Comment().Before {
		if strings.Contains(strings.ToLower(com.Token), text) {
			return true
		}
	}
	return false
}

// leaveAlone1 reports whether x is marked with a comment containing
// "buildifier: leave-alone", case-insensitive.
func leaveAlone1(x Expr) bool {
	return hasComment(x, "buildifier: leave-alone")
}

// doNotSort reports whether x is marked with a comment containing
// "do not sort", case-insensitive.
func doNotSort(x Expr) bool {
	return hasComment(x, "do not sort")
}

// keepSorted reports whether x is marked with a comment containing
// "keep sorted", case-insensitive.
func keepSorted(x Expr) bool {
	return hasComment(x, "keep sorted")
}

// fixLabels rewrites labels into a canonical form.
//
// First, it joins labels written as string addition, turning
// "//x" + ":y" (usually split across multiple lines) into "//x:y".
//
// Second, it removes redundant target qualifiers, turning labels like
// "//third_party/m4:m4" into "//third_party/m4" as well as ones like
// "@foo//:foo" into "@foo".
//
func fixLabels(f *File, info *RewriteInfo) {
	joinLabel := func(p *Expr) {
		add, ok := (*p).(*BinaryExpr)
		if !ok || add.Op != "+" {
			return
		}
		str1, ok := add.X.(*StringExpr)
		if !ok || !strings.HasPrefix(str1.Value, "//") || strings.Contains(str1.Value, " ") {
			return
		}
		str2, ok := add.Y.(*StringExpr)
		if !ok || strings.Contains(str2.Value, " ") {
			return
		}
		info.EditLabel++
		str1.Value += str2.Value

		// Deleting nodes add and str2.
		// Merge comments from add, str1, and str2 and save in str1.
		com1 := add.Comment()
		com2 := str1.Comment()
		com3 := str2.Comment()
		com1.Before = append(com1.Before, com2.Before...)
		com1.Before = append(com1.Before, com3.Before...)
		com1.Suffix = append(com1.Suffix, com2.Suffix...)
		com1.Suffix = append(com1.Suffix, com3.Suffix...)
		*str1.Comment() = *com1

		*p = str1
	}

	labelPrefix := "//"
	if tables.StripLabelLeadingSlashes {
		labelPrefix = ""
	}
	// labelRE matches label strings, e.g. @r//x/y/z:abc
	// where $1 is @r//x/y/z, $2 is @r//, $3 is r, $4 is z, $5 is abc.
	labelRE := regexp.MustCompile(`^(((?:@(\w+))?//|` + labelPrefix + `)(?:.+/)?([^:]*))(?::([^:]+))?$`)

	shortenLabel := func(v Expr) {
		str, ok := v.(*StringExpr)
		if !ok {
			return
		}
		editPerformed := false

		if tables.StripLabelLeadingSlashes && strings.HasPrefix(str.Value, "//") {
			if filepath.Dir(f.Path) == "." || !strings.HasPrefix(str.Value, "//:") {
				editPerformed = true
				str.Value = str.Value[2:]
			}
		}

		if tables.ShortenAbsoluteLabelsToRelative {
			thisPackage := labelPrefix + filepath.Dir(f.Path)
			// filepath.Dir on Windows uses backslashes as separators, while labels always have slashes.
			if filepath.Separator != '/' {
				thisPackage = strings.Replace(thisPackage, string(filepath.Separator), "/", -1)
			}

			if str.Value == thisPackage {
				editPerformed = true
				str.Value = ":" + path.Base(str.Value)
			} else if strings.HasPrefix(str.Value, thisPackage+":") {
				editPerformed = true
				str.Value = str.Value[len(thisPackage):]
			}
		}

		m := labelRE.FindStringSubmatch(str.Value)
		if m == nil {
			return
		}
		if m[4] != "" && m[4] == m[5] { // e.g. //foo:foo
			editPerformed = true
			str.Value = m[1]
		} else if m[3] != "" && m[4] == "" && m[3] == m[5] { // e.g. @foo//:foo
			editPerformed = true
			str.Value = "@" + m[3]
		}
		if editPerformed {
			info.EditLabel++
		}
	}

	Walk(f, func(v Expr, stk []Expr) {
		switch v := v.(type) {
		case *CallExpr:
			if leaveAlone(stk, v) {
				return
			}
			for i := range v.List {
				if leaveAlone1(v.List[i]) {
					continue
				}
				as, ok := v.List[i].(*AssignExpr)
				if !ok {
					continue
				}
				key, ok := as.LHS.(*Ident)
				if !ok || !tables.IsLabelArg[key.Name] || tables.LabelBlacklist[callName(v)+"."+key.Name] {
					continue
				}
				if leaveAlone1(as.RHS) {
					continue
				}
				if list, ok := as.RHS.(*ListExpr); ok {
					for i := range list.List {
						if leaveAlone1(list.List[i]) {
							continue
						}
						joinLabel(&list.List[i])
						shortenLabel(list.List[i])
					}
				}
				if set, ok := as.RHS.(*SetExpr); ok {
					for i := range set.List {
						if leaveAlone1(set.List[i]) {
							continue
						}
						joinLabel(&set.List[i])
						shortenLabel(set.List[i])
					}
				} else {
					joinLabel(&as.RHS)
					shortenLabel(as.RHS)
				}
			}
		}
	})
}

// callName returns the name of the rule being called by call.
// If the call is not to a literal rule name, callName returns "".
func callName(call *CallExpr) string {
	rule, ok := call.X.(*Ident)
	if !ok {
		return ""
	}
	return rule.Name
}

// sortCallArgs sorts lists of named arguments to a call.
func sortCallArgs(f *File, info *RewriteInfo) {
	Walk(f, func(v Expr, stk []Expr) {
		call, ok := v.(*CallExpr)
		if !ok {
			return
		}
		if leaveAlone(stk, call) {
			return
		}
		rule := callName(call)
		if rule == "" {
			return
		}

		// Find the tail of the argument list with named arguments.
		start := len(call.List)
		for start > 0 && argName(call.List[start-1]) != "" {
			start--
		}

		// Record information about each arg into a sortable list.
		var args namedArgs
		for i, x := range call.List[start:] {
			name := argName(x)
			args = append(args, namedArg{ruleNamePriority(rule, name), name, i, x})
		}

		// Sort the list and put the args back in the new order.
		if sort.IsSorted(args) {
			return
		}
		info.SortCall++
		sort.Sort(args)
		for i, x := range args {
			call.List[start+i] = x.expr
		}
	})
}

// ruleNamePriority maps a rule argument name to its sorting priority.
// It could use the auto-generated per-rule tables but for now it just
// falls back to the original list.
func ruleNamePriority(rule, arg string) int {
	ruleArg := rule + "." + arg
	if val, ok := tables.NamePriority[ruleArg]; ok {
		return val
	}
	return tables.NamePriority[arg]
	/*
		list := ruleArgOrder[rule]
		if len(list) == 0 {
			return tables.NamePriority[arg]
		}
		for i, x := range list {
			if x == arg {
				return i
			}
		}
		return len(list)
	*/
}

// If x is of the form key=value, argName returns the string key.
// Otherwise argName returns "".
func argName(x Expr) string {
	if as, ok := x.(*AssignExpr); ok {
		if id, ok := as.LHS.(*Ident); ok {
			return id.Name
		}
	}
	return ""
}

// A namedArg records information needed for sorting
// a named call argument into its proper position.
type namedArg struct {
	priority int    // kind of name; first sort key
	name     string // name; second sort key
	index    int    // original index; final sort key
	expr     Expr   // name=value argument
}

// namedArgs is a slice of namedArg that implements sort.Interface
type namedArgs []namedArg

func (x namedArgs) Len() int      { return len(x) }
func (x namedArgs) Swap(i, j int) { x[i], x[j] = x[j], x[i] }

func (x namedArgs) Less(i, j int) bool {
	p := x[i]
	q := x[j]
	if p.priority != q.priority {
		return p.priority < q.priority
	}
	if p.name != q.name {
		return p.name < q.name
	}
	return p.index < q.index
}

// sortStringLists sorts lists of string literals used as specific rule arguments.
func sortStringLists(f *File, info *RewriteInfo) {
	Walk(f, func(v Expr, stk []Expr) {
		switch v := v.(type) {
		case *CallExpr:
			if leaveAlone(stk, v) {
				return
			}
			rule := callName(v)
			for _, arg := range v.List {
				if leaveAlone1(arg) {
					continue
				}
				as, ok := arg.(*AssignExpr)
				if !ok || leaveAlone1(as) || doNotSort(as) {
					continue
				}
				key, ok := as.LHS.(*Ident)
				if !ok {
					continue
				}
				context := rule + "." + key.Name
				if !tables.IsSortableListArg[key.Name] || tables.SortableBlacklist[context] || f.Type == TypeDefault || f.Type == TypeBzl {
					continue
				}
				if disabled("unsafesort") && !tables.SortableWhitelist[context] && !allowedSort(context) {
					continue
				}
				sortStringList(as.RHS, info, context)
			}
		case *AssignExpr:
			if disabled("unsafesort") {
				return
			}
			// "keep sorted" comment on x = list forces sorting of list.
			as := v
			if keepSorted(as) {
				sortStringList(as.RHS, info, "?")
			}
		case *KeyValueExpr:
			if disabled("unsafesort") {
				return
			}
			// "keep sorted" before key: list also forces sorting of list.
			if keepSorted(v) {
				sortStringList(v.Value, info, "?")
			}
		case *ListExpr:
			if disabled("unsafesort") {
				return
			}
			// "keep sorted" comment above first list element also forces sorting of list.
			if len(v.List) > 0 && (keepSorted(v) || keepSorted(v.List[0])) {
				sortStringList(v, info, "?")
			}
		}
	})
}

// SortStringList sorts x, a list of strings.
func SortStringList(x Expr) {
	sortStringList(x, nil, "")
}

// sortStringList sorts x, a list of strings.
// The list is broken by non-strings and by blank lines and comments into chunks.
// Each chunk is sorted in place.
func sortStringList(x Expr, info *RewriteInfo, context string) {
	list, ok := x.(*ListExpr)
	if !ok || len(list.List) < 2 || doNotSort(list.List[0]) {
		return
	}

	forceSort := keepSorted(list) || keepSorted(list.List[0])

	// TODO(bazel-team): Decide how to recognize lists that cannot
	// be sorted. Avoiding all lists with comments avoids sorting
	// lists that say explicitly, in some form or another, why they
	// cannot be sorted. For example, many cc_test rules require
	// certain order in their deps attributes.
	if !forceSort {
		if line, _ := hasComments(list); line {
			return
		}
	}

	// Sort chunks of the list with no intervening blank lines or comments.
	for i := 0; i < len(list.List); {
		if _, ok := list.List[i].(*StringExpr); !ok {
			i++
			continue
		}

		j := i + 1
		for ; j < len(list.List); j++ {
			if str, ok := list.List[j].(*StringExpr); !ok || len(str.Before) > 0 {
				break
			}
		}

		var chunk []stringSortKey
		for index, x := range list.List[i:j] {
			chunk = append(chunk, makeSortKey(index, x.(*StringExpr)))
		}
		if !sort.IsSorted(byStringExpr(chunk)) || !isUniq(chunk) {
			if info != nil {
				info.SortStringList++
				if !tables.SortableWhitelist[context] {
					info.UnsafeSort++
					info.Log = append(info.Log, "sort:"+context)
				}
			}
			before := chunk[0].x.Comment().Before
			chunk[0].x.Comment().Before = nil

			sort.Sort(byStringExpr(chunk))
			chunk = uniq(chunk)

			chunk[0].x.Comment().Before = before
			for offset, key := range chunk {
				list.List[i+offset] = key.x
			}
			list.List = append(list.List[:(i+len(chunk))], list.List[j:]...)
		}

		i = j
	}
}

// uniq removes duplicates from a list, which must already be sorted.
// It edits the list in place.
func uniq(sortedList []stringSortKey) []stringSortKey {
	out := sortedList[:0]
	for _, sk := range sortedList {
		if len(out) == 0 || sk.value != out[len(out)-1].value {
			out = append(out, sk)
		}
	}
	return out
}

// isUniq reports whether the sorted list only contains unique elements.
func isUniq(list []stringSortKey) bool {
	for i := range list {
		if i+1 < len(list) && list[i].value == list[i+1].value {
			return false
		}
	}
	return true
}

// If stk describes a call argument like rule(arg=...), callArgName
// returns the name of that argument, formatted as "rule.arg".
func callArgName(stk []Expr) string {
	n := len(stk)
	if n < 2 {
		return ""
	}
	arg := argName(stk[n-1])
	if arg == "" {
		return ""
	}
	call, ok := stk[n-2].(*CallExpr)
	if !ok {
		return ""
	}
	rule, ok := call.X.(*Ident)
	if !ok {
		return ""
	}
	return rule.Name + "." + arg
}

// A stringSortKey records information about a single string literal to be
// sorted. The strings are first grouped into four phases: most strings,
// strings beginning with ":", strings beginning with "//", and strings
// beginning with "@". The next significant part of the comparison is the list
// of elements in the value, where elements are split at `.' and `:'. Finally
// we compare by value and break ties by original index.
type stringSortKey struct {
	phase    int
	split    []string
	value    string
	original int
	x        Expr
}

func makeSortKey(index int, x *StringExpr) stringSortKey {
	key := stringSortKey{
		value:    x.Value,
		original: index,
		x:        x,
	}

	switch {
	case strings.HasPrefix(x.Value, ":"):
		key.phase = 1
	case strings.HasPrefix(x.Value, "//") || (tables.StripLabelLeadingSlashes && !strings.HasPrefix(x.Value, "@")):
		key.phase = 2
	case strings.HasPrefix(x.Value, "@"):
		key.phase = 3
	}

	key.split = strings.Split(strings.Replace(x.Value, ":", ".", -1), ".")
	return key
}

// byStringExpr implements sort.Interface for a list of stringSortKey.
type byStringExpr []stringSortKey

func (x byStringExpr) Len() int      { return len(x) }
func (x byStringExpr) Swap(i, j int) { x[i], x[j] = x[j], x[i] }

func (x byStringExpr) Less(i, j int) bool {
	xi := x[i]
	xj := x[j]

	if xi.phase != xj.phase {
		return xi.phase < xj.phase
	}
	for k := 0; k < len(xi.split) && k < len(xj.split); k++ {
		if xi.split[k] != xj.split[k] {
			return xi.split[k] < xj.split[k]
		}
	}
	if len(xi.split) != len(xj.split) {
		return len(xi.split) < len(xj.split)
	}
	if xi.value != xj.value {
		return xi.value < xj.value
	}
	return xi.original < xj.original
}

// fixMultilinePlus turns
//
//	... +
//	[ ... ]
//
//	... +
//	call(...)
//
// into
//	... + [
//		...
//	]
//
//	... + call(
//		...
//	)
//
// which typically works better with our aggressively compact formatting.
func fixMultilinePlus(f *File, info *RewriteInfo) {

	// List manipulation helpers.
	// As a special case, we treat f([...]) as a list, mainly
	// for glob.

	// isList reports whether x is a list.
	var isList func(x Expr) bool
	isList = func(x Expr) bool {
		switch x := x.(type) {
		case *ListExpr:
			return true
		case *CallExpr:
			if len(x.List) == 1 {
				return isList(x.List[0])
			}
		}
		return false
	}

	// isMultiLine reports whether x is a multiline list.
	var isMultiLine func(Expr) bool
	isMultiLine = func(x Expr) bool {
		switch x := x.(type) {
		case *ListExpr:
			return x.ForceMultiLine || len(x.List) > 1
		case *CallExpr:
			if x.ForceMultiLine || len(x.List) > 1 && !x.ForceCompact {
				return true
			}
			if len(x.List) == 1 {
				return isMultiLine(x.List[0])
			}
		}
		return false
	}

	// forceMultiLine tries to force the list x to use a multiline form.
	// It reports whether it was successful.
	var forceMultiLine func(Expr) bool
	forceMultiLine = func(x Expr) bool {
		switch x := x.(type) {
		case *ListExpr:
			// Already multi line?
			if x.ForceMultiLine {
				return true
			}
			// If this is a list containing a list, force the
			// inner list to be multiline instead.
			if len(x.List) == 1 && forceMultiLine(x.List[0]) {
				return true
			}
			x.ForceMultiLine = true
			return true

		case *CallExpr:
			if len(x.List) == 1 {
				return forceMultiLine(x.List[0])
			}
		}
		return false
	}

	skip := map[Expr]bool{}
	Walk(f, func(v Expr, stk []Expr) {
		if skip[v] {
			return
		}
		bin, ok := v.(*BinaryExpr)
		if !ok || bin.Op != "+" {
			return
		}

		// Found a +.
		// w + x + y + z parses as ((w + x) + y) + z,
		// so chase down the left side to make a list of
		// all the things being added together, separated
		// by the BinaryExprs that join them.
		// Mark them as "skip" so that when Walk recurses
		// into the subexpressions, we won't reprocess them.
		var all []Expr
		for {
			all = append(all, bin.Y, bin)
			bin1, ok := bin.X.(*BinaryExpr)
			if !ok || bin1.Op != "+" {
				break
			}
			bin = bin1
			skip[bin] = true
		}
		all = append(all, bin.X)

		// Because the outermost expression was the
		// rightmost one, the list is backward. Reverse it.
		for i, j := 0, len(all)-1; i < j; i, j = i+1, j-1 {
			all[i], all[j] = all[j], all[i]
		}

		// The 'all' slice is alternating addends and BinaryExpr +'s:
		//	w, +, x, +, y, +, z
		// If there are no lists involved, don't rewrite anything.
		haveList := false
		for i := 0; i < len(all); i += 2 {
			if isList(all[i]) {
				haveList = true
				break
			}
		}
		if !haveList {
			return
		}

		// Okay, there are lists.
		// Consider each + next to a line break.
		for i := 1; i < len(all); i += 2 {
			bin := all[i].(*BinaryExpr)
			if !bin.LineBreak {
				continue
			}

			// We're going to break the line after the +.
			// If it is followed by a list, force that to be
			// multiline instead.
			if forceMultiLine(all[i+1]) {
				bin.LineBreak = false
				continue
			}

			// If the previous list was multiline already,
			// don't bother with the line break after
			// the +.
			if isMultiLine(all[i-1]) {
				bin.LineBreak = false
				continue
			}
		}
	})
}

// sortAllLoadArgs sorts all load arguments in the file
func sortAllLoadArgs(f *File, info *RewriteInfo) {
	Walk(f, func(v Expr, stk []Expr) {
		if load, ok := v.(*LoadStmt); ok {
			if SortLoadArgs(load) {
				info.SortLoad++
			}
		}
	})
}

// hasComments reports whether any comments are associated with
// the list or its elements.
func hasComments(list *ListExpr) (line, suffix bool) {
	com := list.Comment()
	if len(com.Before) > 0 || len(com.After) > 0 || len(list.End.Before) > 0 {
		line = true
	}
	if len(com.Suffix) > 0 {
		suffix = true
	}
	for _, elem := range list.List {
		com := elem.Comment()
		if len(com.Before) > 0 {
			line = true
		}
		if len(com.Suffix) > 0 {
			suffix = true
		}
	}
	return
}

// A wrapper for a LoadStmt's From and To slices for consistent sorting of their contents.
// It's assumed that the following slices have the same length. The contents are sorted by
// the `To` attribute, but all items with equal "From" and "To" parts are placed before the items
// with different parts.
type loadArgs struct {
	From     []*Ident
	To       []*Ident
	modified bool
}

func (args loadArgs) Len() int {
	return len(args.From)
}

func (args loadArgs) Swap(i, j int) {
	args.From[i], args.From[j] = args.From[j], args.From[i]
	args.To[i], args.To[j] = args.To[j], args.To[i]
	args.modified = true
}

func (args loadArgs) Less(i, j int) bool {
	// Arguments with equal "from" and "to" parts are prioritized
	equalI := args.From[i].Name == args.To[i].Name
	equalJ := args.From[j].Name == args.To[j].Name
	if equalI != equalJ {
		// If equalI and !equalJ, return true, otherwise false.
		// Equivalently, return equalI.
		return equalI
	}
	return args.To[i].Name < args.To[j].Name
}

// SortLoadArgs sorts a load statement arguments (lexicographically, but positional first)
func SortLoadArgs(load *LoadStmt) bool {
	args := loadArgs{From: load.From, To: load.To}
	sort.Sort(args)
	return args.modified
}

// formatDocstrings fixes the indentation and trailing whitespace of docstrings
func formatDocstrings(f *File, info *RewriteInfo) {
	Walk(f, func(v Expr, stk []Expr) {
		def, ok := v.(*DefStmt)
		if !ok || len(def.Body) == 0 {
			return
		}
		docstring, ok := def.Body[0].(*StringExpr)
		if !ok || !docstring.TripleQuote {
			return
		}

		oldIndentation := docstring.Start.LineRune - 1 // LineRune starts with 1
		newIndentation := nestedIndentation * len(stk)

		// Operate on Token, not Value, because their line breaks can be different if a line ends with
		// a backslash.
		updatedToken := formatString(docstring.Token, oldIndentation, newIndentation)
		if updatedToken != docstring.Token {
			docstring.Token = updatedToken
			// Update the value to keep it consistent with Token
			docstring.Value, _, _ = Unquote(updatedToken)
			info.FormatDocstrings++
		}
	})
}

// formatString modifies a string value of a docstring to match the new indentation level and
// to remove trailing whitespace from its lines.
func formatString(value string, oldIndentation, newIndentation int) string {
	difference := newIndentation - oldIndentation
	lines := strings.Split(value, "\n")
	for i, line := range lines {
		if i == 0 {
			// The first line shouldn't be touched because it starts right after ''' or """
			continue
		}
		if difference > 0 {
			line = strings.Repeat(" ", difference) + line
		} else {
			for i, rune := range line {
				if i == -difference || rune != ' ' {
					line = line[i:]
					break
				}
			}
		}
		if i != len(lines)-1 {
			// Remove trailing space from the line unless it's the last line that's responsible
			// for the indentation of the closing `"""`
			line = strings.TrimRight(line, " ")
		}
		lines[i] = line
	}
	return strings.Join(lines, "\n")
}

// argumentType returns an integer by which funcall arguments can be sorted:
// 1 for positional, 2 for named, 3 for *args, 4 for **kwargs
func argumentType(expr Expr) int {
	switch expr := expr.(type) {
	case *UnaryExpr:
		switch expr.Op {
		case "**":
			return 4
		case "*":
			return 3
		}
	case *AssignExpr:
		return 2
	}
	return 1
}

// reorderArguments fixes the order of arguments of a function call
// (positional, named, *args, **kwargs)
func reorderArguments(f *File, info *RewriteInfo) {
	Walk(f, func(expr Expr, stack []Expr) {
		call, ok := expr.(*CallExpr)
		if !ok {
			return
		}
		compare := func(i, j int) bool {
			return argumentType(call.List[i]) < argumentType(call.List[j])
		}
		if !sort.SliceIsSorted(call.List, compare) {
			sort.SliceStable(call.List, compare)
			info.ReorderArguments++
		}
	})
}

// editOctals inserts 'o' into octal numbers to make it more obvious they are octal
// 0123 -> 0o123
func editOctals(f *File, info *RewriteInfo) {
	Walk(f, func(expr Expr, stack []Expr) {
		l, ok := expr.(*LiteralExpr)
		if !ok {
			return
		}
		if len(l.Token) > 1 && l.Token[0] == '0' && l.Token[1] >= '0' && l.Token[1] <= '9' {
			l.Token = "0o" + l.Token[1:]
			info.EditOctal++
		}
	})
}
