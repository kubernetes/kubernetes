package internal

import (
	"regexp"
	"strings"

	"github.com/onsi/ginkgo/v2/types"
)

/*
	If a container marked as focus has a descendant that is also marked as focus, Ginkgo's policy is to
	unmark the container's focus.  This gives developers a more intuitive experience when debugging specs.
	It is common to focus a container to just run a subset of specs, then identify the specific specs within the container to focus -
	this policy allows the developer to simply focus those specific specs and not need to go back and turn the focus off of the container:

	As a common example, consider:

		FDescribe("something to debug", function() {
			It("works", function() {...})
			It("works", function() {...})
			FIt("doesn't work", function() {...})
			It("works", function() {...})
		})

	here the developer's intent is to focus in on the `"doesn't work"` spec and not to run the adjacent specs in the focused `"something to debug"` container.
	The nested policy applied by this function enables this behavior.
*/
func ApplyNestedFocusPolicyToTree(tree *TreeNode) {
	var walkTree func(tree *TreeNode) bool
	walkTree = func(tree *TreeNode) bool {
		if tree.Node.MarkedPending {
			return false
		}
		hasFocusedDescendant := false
		for _, child := range tree.Children {
			childHasFocus := walkTree(child)
			hasFocusedDescendant = hasFocusedDescendant || childHasFocus
		}
		tree.Node.MarkedFocus = tree.Node.MarkedFocus && !hasFocusedDescendant
		return tree.Node.MarkedFocus || hasFocusedDescendant
	}

	walkTree(tree)
}

/*
	Ginkgo supports focussing specs using `FIt`, `FDescribe`, etc. - this is called "programmatic focus"
	It also supports focussing specs using regular expressions on the command line (`-focus=`, `-skip=`) that match against spec text
	and file filters (`-focus-files=`, `-skip-files=`) that match against code locations for nodes in specs.

	If any of the CLI flags are provided they take precedence.  The file filters run first followed by the regex filters.

	This function sets the `Skip` property on specs by applying Ginkgo's focus policy:
	- If there are no CLI arguments and no programmatic focus, do nothing.
	- If there are no CLI arguments but a spec somewhere has programmatic focus, skip any specs that have no programmatic focus.
	- If there are CLI arguments parse them and skip any specs that either don't match the focus filters or do match the skip filters.

	*Note:* specs with pending nodes are Skipped when created by NewSpec.
*/
func ApplyFocusToSpecs(specs Specs, description string, suiteLabels Labels, suiteConfig types.SuiteConfig) (Specs, bool) {
	focusString := strings.Join(suiteConfig.FocusStrings, "|")
	skipString := strings.Join(suiteConfig.SkipStrings, "|")

	hasFocusCLIFlags := focusString != "" || skipString != "" || len(suiteConfig.SkipFiles) > 0 || len(suiteConfig.FocusFiles) > 0 || suiteConfig.LabelFilter != ""

	type SkipCheck func(spec Spec) bool

	// by default, skip any specs marked pending
	skipChecks := []SkipCheck{func(spec Spec) bool { return spec.Nodes.HasNodeMarkedPending() }}
	hasProgrammaticFocus := false

	if !hasFocusCLIFlags {
		// check for programmatic focus
		for _, spec := range specs {
			if spec.Nodes.HasNodeMarkedFocus() && !spec.Nodes.HasNodeMarkedPending() {
				skipChecks = append(skipChecks, func(spec Spec) bool { return !spec.Nodes.HasNodeMarkedFocus() })
				hasProgrammaticFocus = true
				break
			}
		}
	}

	if suiteConfig.LabelFilter != "" {
		labelFilter, _ := types.ParseLabelFilter(suiteConfig.LabelFilter)
		skipChecks = append(skipChecks, func(spec Spec) bool { 
			return !labelFilter(UnionOfLabels(suiteLabels, spec.Nodes.UnionOfLabels())) 
		})
	}

	if len(suiteConfig.FocusFiles) > 0 {
		focusFilters, _ := types.ParseFileFilters(suiteConfig.FocusFiles)
		skipChecks = append(skipChecks, func(spec Spec) bool { return !focusFilters.Matches(spec.Nodes.CodeLocations()) })
	}

	if len(suiteConfig.SkipFiles) > 0 {
		skipFilters, _ := types.ParseFileFilters(suiteConfig.SkipFiles)
		skipChecks = append(skipChecks, func(spec Spec) bool { return skipFilters.Matches(spec.Nodes.CodeLocations()) })
	}

	if focusString != "" {
		// skip specs that don't match the focus string
		re := regexp.MustCompile(focusString)
		skipChecks = append(skipChecks, func(spec Spec) bool { return !re.MatchString(description + " " + spec.Text()) })
	}

	if skipString != "" {
		// skip specs that match the skip string
		re := regexp.MustCompile(skipString)
		skipChecks = append(skipChecks, func(spec Spec) bool { return re.MatchString(description + " " + spec.Text()) })
	}

	// skip specs if shouldSkip() is true.  note that we do nothing if shouldSkip() is false to avoid overwriting skip status established by the node's pending status
	processedSpecs := Specs{}
	for _, spec := range specs {
		for _, skipCheck := range skipChecks {
			if skipCheck(spec) {
				spec.Skip = true
				break
			}
		}
		processedSpecs = append(processedSpecs, spec)
	}

	return processedSpecs, hasProgrammaticFocus
}
