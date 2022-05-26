package specificity

type StandardPackageMatch struct {
}

func (s StandardPackageMatch) IsMoreSpecific(than MatchSpecificity) bool {
	return isMoreSpecific(s, than)
}

func (s StandardPackageMatch) Equal(to MatchSpecificity) bool {
	return equalSpecificity(s, to)
}

func (s StandardPackageMatch) class() specificityClass {
	return StandardPackageClass
}

func (s StandardPackageMatch) String() string {
	return "Standard"
}
