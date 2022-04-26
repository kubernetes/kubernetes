package exhaustive

import "golang.org/x/tools/go/analysis"

// NOTE: Fact types must remain gob-coding compatible.
// See TestFactsGob.

var _ analysis.Fact = (*enumMembersFact)(nil)

type enumMembersFact struct{ Members enumMembers }

func (f *enumMembersFact) AFact()         {}
func (f *enumMembersFact) String() string { return f.Members.factString() }

// exportFact exports the enum members for the given enum type.
func exportFact(pass *analysis.Pass, enumTyp enumType, members enumMembers) {
	pass.ExportObjectFact(enumTyp.factObject(), &enumMembersFact{members})
}

// importFact imports the enum members for the given possible enum type.
// An (_, false) return indicates that the enum type is not a known one.
func importFact(pass *analysis.Pass, possibleEnumType enumType) (enumMembers, bool) {
	var f enumMembersFact
	ok := pass.ImportObjectFact(possibleEnumType.factObject(), &f)
	if !ok {
		return enumMembers{}, false
	}
	return f.Members, true
}
