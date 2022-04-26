package linter

import (
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/go-toolsmith/astfmt"
)

type checkerProto struct {
	info        *CheckerInfo
	constructor func(*Context) (*Checker, error)
}

// prototypes is a set of registered checkers that are not yet instantiated.
// Registration should be done with AddChecker function.
// Initialized checkers can be obtained with NewChecker function.
var prototypes = make(map[string]checkerProto)

func getCheckersInfo() []*CheckerInfo {
	infoList := make([]*CheckerInfo, 0, len(prototypes))
	for _, proto := range prototypes {
		infoCopy := *proto.info
		infoList = append(infoList, &infoCopy)
	}
	sort.Slice(infoList, func(i, j int) bool {
		return infoList[i].Name < infoList[j].Name
	})
	return infoList
}

func addChecker(info *CheckerInfo, constructor func(*CheckerContext) (FileWalker, error)) {
	if _, ok := prototypes[info.Name]; ok {
		panic(fmt.Sprintf("checker with name %q already registered", info.Name))
	}

	// Validate param value type.
	for pname, param := range info.Params {
		switch param.Value.(type) {
		case string, int, bool:
			// OK.
		default:
			panic(fmt.Sprintf("unsupported %q param type value: %T",
				pname, param.Value))
		}
	}

	trimDocumentation := func(info *CheckerInfo) {
		fields := []*string{
			&info.Summary,
			&info.Details,
			&info.Before,
			&info.After,
			&info.Note,
		}
		for _, f := range fields {
			*f = strings.TrimSpace(*f)
		}
	}

	trimDocumentation(info)

	if err := validateCheckerInfo(info); err != nil {
		panic(err)
	}

	proto := checkerProto{
		info: info,
		constructor: func(ctx *Context) (*Checker, error) {
			var c Checker
			c.Info = info
			c.ctx = CheckerContext{
				Context: ctx,
				printer: astfmt.NewPrinter(ctx.FileSet),
			}
			var err error
			c.fileWalker, err = constructor(&c.ctx)
			return &c, err
		},
	}

	prototypes[info.Name] = proto
}

func newChecker(ctx *Context, info *CheckerInfo) (*Checker, error) {
	proto, ok := prototypes[info.Name]
	if !ok {
		panic(fmt.Sprintf("checker with name %q not registered", info.Name))
	}
	return proto.constructor(ctx)
}

func validateCheckerInfo(info *CheckerInfo) error {
	steps := []func(*CheckerInfo) error{
		validateCheckerName,
		validateCheckerDocumentation,
		validateCheckerTags,
	}

	for _, step := range steps {
		if err := step(info); err != nil {
			return fmt.Errorf("%q validation error: %v", info.Name, err)
		}
	}
	return nil
}

var validIdentRE = regexp.MustCompile(`^\w+$`)

func validateCheckerName(info *CheckerInfo) error {
	if !validIdentRE.MatchString(info.Name) {
		return fmt.Errorf("checker name contains illegal chars")
	}
	return nil
}

func validateCheckerDocumentation(info *CheckerInfo) error {
	// TODO(quasilyte): validate documentation.
	return nil
}

func validateCheckerTags(info *CheckerInfo) error {
	tagSet := make(map[string]bool)
	for _, tag := range info.Tags {
		if tagSet[tag] {
			return fmt.Errorf("duplicated tag %q", tag)
		}
		if !validIdentRE.MatchString(tag) {
			return fmt.Errorf("checker tag %q contains illegal chars", tag)
		}
		tagSet[tag] = true
	}
	return nil
}
