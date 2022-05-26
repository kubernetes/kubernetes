package sections

import (
	"fmt"
	"strings"

	"github.com/daixiang0/gci/pkg/configuration"
	"github.com/daixiang0/gci/pkg/constants"
	importPkg "github.com/daixiang0/gci/pkg/gci/imports"
	"github.com/daixiang0/gci/pkg/gci/specificity"
)

func init() {
	commentLineType := SectionType{
		generatorFun: func(parameter string, sectionPrefix, sectionSuffix Section) (Section, error) {
			return CommentLine{parameter}, nil
		},
		aliases:       []string{"Comment", "CommentLine"},
		parameterHelp: "your text here",
		description:   "Prints the specified indented comment",
	}.StandAloneSection()
	SectionParserInst.registerSectionWithoutErr(&commentLineType)
}

type CommentLine struct {
	Comment string
}

func (c CommentLine) MatchSpecificity(spec importPkg.ImportDef) specificity.MatchSpecificity {
	return specificity.MisMatch{}
}

func (c CommentLine) Format(imports []importPkg.ImportDef, cfg configuration.FormatterConfiguration) string {
	comment := constants.Indent + "//" + c.Comment
	if !strings.HasSuffix(comment, constants.Linebreak) {
		comment += constants.Linebreak
	}
	return comment
}

func (c CommentLine) sectionPrefix() Section {
	return nil
}

func (c CommentLine) sectionSuffix() Section {
	return nil
}

func (c CommentLine) String() string {
	return fmt.Sprintf("CommentLine(%s)", c.Comment)
}
