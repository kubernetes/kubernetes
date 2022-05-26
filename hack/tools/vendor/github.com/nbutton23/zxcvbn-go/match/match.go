package match

//Matches is an alies for []Match used for sorting
type Matches []Match

func (s Matches) Len() int {
	return len(s)
}
func (s Matches) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
func (s Matches) Less(i, j int) bool {
	if s[i].I < s[j].I {
		return true
	} else if s[i].I == s[j].I {
		return s[i].J < s[j].J
	} else {
		return false
	}
}

// Match represents different matches
type Match struct {
	Pattern        string
	I, J           int
	Token          string
	DictionaryName string
	Entropy        float64
}

//DateMatch is specifilly a match for type date
type DateMatch struct {
	Pattern          string
	I, J             int
	Token            string
	Separator        string
	Day, Month, Year int64
}

//Matcher are a func and ID that can be used to match different passwords
type Matcher struct {
	MatchingFunc func(password string) []Match
	ID           string
}
