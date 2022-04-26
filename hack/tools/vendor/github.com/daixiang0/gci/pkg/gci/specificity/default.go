package specificity

type Default struct {
}

func (d Default) IsMoreSpecific(than MatchSpecificity) bool {
	return isMoreSpecific(d, than)
}
func (d Default) Equal(to MatchSpecificity) bool {
	return equalSpecificity(d, to)
}

func (d Default) class() specificityClass {
	return DefaultClass
}

func (d Default) String() string {
	return "Default"
}
