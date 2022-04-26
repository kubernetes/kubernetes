package sections

import (
	"fmt"

	"github.com/daixiang0/gci/pkg/configuration"
	importPkg "github.com/daixiang0/gci/pkg/gci/imports"
	"github.com/daixiang0/gci/pkg/gci/specificity"
)

// Section defines a part of the formatted output.
type Section interface {
	// MatchSpecificity returns how well an Import matches to this Section
	MatchSpecificity(spec importPkg.ImportDef) specificity.MatchSpecificity
	// Format receives the array of imports that have matched this section and formats them according to it´s rules
	Format(imports []importPkg.ImportDef, cfg configuration.FormatterConfiguration) string
	// Returns the Section that will be prefixed if this section is rendered
	sectionPrefix() Section
	// Returns the Section that will be suffixed if this section is rendered
	sectionSuffix() Section
	// String Implements the stringer interface
	String() string
}

//Unbound methods that are required until interface methods are supported

// Default method for formatting a section
func inorderSectionFormat(section Section, imports []importPkg.ImportDef, cfg configuration.FormatterConfiguration) string {
	imports = importPkg.SortImportsByPath(imports)
	var output string
	if len(imports) > 0 && section.sectionPrefix() != nil {
		// imports are not passed to a prefix section to prevent rendering them twice
		output += section.sectionPrefix().Format([]importPkg.ImportDef{}, cfg)
	}
	for _, importDef := range imports {
		output += importDef.Format(cfg)
	}
	if len(imports) > 0 && section.sectionSuffix() != nil {
		// imports are not passed to a suffix section to prevent rendering them twice
		output += section.sectionSuffix().Format([]importPkg.ImportDef{}, cfg)
	}
	return output
}

// Default method for converting a section to a String representation
func sectionStringWithPrefixSuffix(mainSectionStr string, section Section) (output string) {
	if section.sectionPrefix() != nil {
		output += fmt.Sprintf("%v:", section.sectionPrefix())
	} else if section.sectionSuffix() != nil {
		// insert empty prefix to make suffix distinguishable from prefix
		output += ":"
	}
	output += mainSectionStr
	if section.sectionSuffix() != nil {
		output += fmt.Sprintf(":%v", section.sectionSuffix())
	}
	return output
}
