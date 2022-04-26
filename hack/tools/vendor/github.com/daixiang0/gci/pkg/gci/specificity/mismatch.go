package specificity

type MisMatch struct {
}

func (m MisMatch) IsMoreSpecific(than MatchSpecificity) bool {
	return isMoreSpecific(m, than)
}

func (m MisMatch) Equal(to MatchSpecificity) bool {
	return equalSpecificity(m, to)
}

func (m MisMatch) class() specificityClass {
	return MisMatchClass
}

func (m MisMatch) String() string {
	return "Mismatch"

}
