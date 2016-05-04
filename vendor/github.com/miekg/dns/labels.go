package dns

// Holds a bunch of helper functions for dealing with labels.

// SplitDomainName splits a name string into it's labels.
// www.miek.nl. returns []string{"www", "miek", "nl"}
// .www.miek.nl. returns []string{"", "www", "miek", "nl"},
// The root label (.) returns nil. Note that using
// strings.Split(s) will work in most cases, but does not handle
// escaped dots (\.) for instance.
// s must be a syntactically valid domain name, see IsDomainName.
func SplitDomainName(s string) (labels []string) {
	if len(s) == 0 {
		return nil
	}
	fqdnEnd := 0 // offset of the final '.' or the length of the name
	idx := Split(s)
	begin := 0
	if s[len(s)-1] == '.' {
		fqdnEnd = len(s) - 1
	} else {
		fqdnEnd = len(s)
	}

	switch len(idx) {
	case 0:
		return nil
	case 1:
		// no-op
	default:
		end := 0
		for i := 1; i < len(idx); i++ {
			end = idx[i]
			labels = append(labels, s[begin:end-1])
			begin = end
		}
	}

	labels = append(labels, s[begin:fqdnEnd])
	return labels
}

// CompareDomainName compares the names s1 and s2 and
// returns how many labels they have in common starting from the *right*.
// The comparison stops at the first inequality. The names are not downcased
// before the comparison.
//
// www.miek.nl. and miek.nl. have two labels in common: miek and nl
// www.miek.nl. and www.bla.nl. have one label in common: nl
//
// s1 and s2 must be syntactically valid domain names.
func CompareDomainName(s1, s2 string) (n int) {
	s1 = Fqdn(s1)
	s2 = Fqdn(s2)
	l1 := Split(s1)
	l2 := Split(s2)

	// the first check: root label
	if l1 == nil || l2 == nil {
		return
	}

	j1 := len(l1) - 1 // end
	i1 := len(l1) - 2 // start
	j2 := len(l2) - 1
	i2 := len(l2) - 2
	// the second check can be done here: last/only label
	// before we fall through into the for-loop below
	if s1[l1[j1]:] == s2[l2[j2]:] {
		n++
	} else {
		return
	}
	for {
		if i1 < 0 || i2 < 0 {
			break
		}
		if s1[l1[i1]:l1[j1]] == s2[l2[i2]:l2[j2]] {
			n++
		} else {
			break
		}
		j1--
		i1--
		j2--
		i2--
	}
	return
}

// CountLabel counts the the number of labels in the string s.
// s must be a syntactically valid domain name.
func CountLabel(s string) (labels int) {
	if s == "." {
		return
	}
	off := 0
	end := false
	for {
		off, end = NextLabel(s, off)
		labels++
		if end {
			return
		}
	}
}

// Split splits a name s into its label indexes.
// www.miek.nl. returns []int{0, 4, 9}, www.miek.nl also returns []int{0, 4, 9}.
// The root name (.) returns nil. Also see SplitDomainName. 
// s must be a syntactically valid domain name.
func Split(s string) []int {
	if s == "." {
		return nil
	}
	idx := make([]int, 1, 3)
	off := 0
	end := false

	for {
		off, end = NextLabel(s, off)
		if end {
			return idx
		}
		idx = append(idx, off)
	}
}

// NextLabel returns the index of the start of the next label in the
// string s starting at offset.
// The bool end is true when the end of the string has been reached.
// Also see PrevLabel.
func NextLabel(s string, offset int) (i int, end bool) {
	quote := false
	for i = offset; i < len(s)-1; i++ {
		switch s[i] {
		case '\\':
			quote = !quote
		default:
			quote = false
		case '.':
			if quote {
				quote = !quote
				continue
			}
			return i + 1, false
		}
	}
	return i + 1, true
}

// PrevLabel returns the index of the label when starting from the right and
// jumping n labels to the left.
// The bool start is true when the start of the string has been overshot.
// Also see NextLabel.
func PrevLabel(s string, n int) (i int, start bool) {
	if n == 0 {
		return len(s), false
	}
	lab := Split(s)
	if lab == nil {
		return 0, true
	}
	if n > len(lab) {
		return 0, true
	}
	return lab[len(lab)-n], false
}
