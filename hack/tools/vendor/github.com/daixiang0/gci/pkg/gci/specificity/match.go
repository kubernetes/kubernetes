package specificity

import "fmt"

type Match struct {
	Length int
}

func (m Match) IsMoreSpecific(than MatchSpecificity) bool {
	otherMatch, isMatch := than.(Match)
	return isMoreSpecific(m, than) || (isMatch && m.Length > otherMatch.Length)
}

func (m Match) Equal(to MatchSpecificity) bool {
	return equalSpecificity(m, to)
}

func (m Match) class() specificityClass {
	return MatchClass
}

func (m Match) String() string {
	return fmt.Sprintf("Match(length: %d)", m.Length)
}
