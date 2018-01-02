package util

import (
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

var level = 0

// func init() { log.SetFlags(0) }
//
// func debugf(f string, vs ...interface{}) {
// 	log.Printf(strings.Repeat("  ", level)+f, vs...)
// }

// ShortRegexpString tries to construct a short regexp that matches exactly the
// provided strings and nothing else.
//
// Warning: the current implementation may use a lot of time of memory.
func ShortRegexpString(vs ...string) (res string) {
	cache := make(map[string][]string)
	return render(shortRegexpString(vs, cache), false)
}

func shortRegexpString(vs []string, cache map[string][]string) (res []string) {
	// Canonicalize (might turn the input into one of the trivial cases below)
	if len(vs) > 1 {
		sort.Strings(vs)
		vs = removeDups(vs)
	}

	// Trivial cases.
	switch len(vs) {
	case 0:
		return nil
	case 1:
		return []string{regexp.QuoteMeta(vs[0])} // Nothing else to do.
	}

	// level++
	// defer func(s string) {
	// 	level--
	// 	debugf("ShortRegexpString(%s) = %#q", s, res)
	// }(fmt.Sprintf("%#q", vs))

	// The one to beat: just put ORs between them (after escaping meta-characters)
	best := make([]string, len(vs))
	for i := range vs {
		best[i] = regexp.QuoteMeta(vs[i])
	}

	cacheKey := render(best, false)
	bestCost := len(cacheKey)

	if cached, ok := cache[cacheKey]; ok {
		return cached
	}
	defer func(key string) {
		// Put clauses in a canonical order and cache them.
		sort.Strings(res)
		cache[key] = res
	}(cacheKey)

	recurse := func(prefix, suffix string, data commonSs) (result []string) {
		// debugf("> recurse(%#q, %#q, %v) on %#q", prefix, suffix, data, vs)
		// defer func() {
		// 	debugf("  recurse(%#q, %#q, %v) on %#q = %#q", prefix, suffix, data, vs, result)
		// }()

		//debugf("%v/%#q/%#q: %v\n", vs, prefix, suffix, data)
		varying := make([]string, data.end-data.start)
		allExist := true
		var preExistingIndices []int
		for i := data.start; i < data.end; i++ {
			substr := vs[i][len(prefix) : len(vs[i])-len(suffix)]
			varying[i-data.start] = substr
			if allExist {
				found := false
				for i := 0; i < len(vs); i++ {
					if i == data.start {
						i = data.end - 1
						continue
					}
					if substr == vs[i] {
						found = true
						preExistingIndices = append(preExistingIndices, i)
						break
					}
				}
				allExist = found
			}
		}

		var others []string
		// combined := make([]string, 0, len(preExistingIndices))
		if allExist && (prefix == "" || suffix == "") {
			others = make([]string, 0, len(vs)-2*len(preExistingIndices))
			sort.Ints(preExistingIndices)
			for i, k := 0, 0; i < len(vs) && k < len(preExistingIndices); i++ {
				if i == data.start {
					i = data.end - 1
					continue
				} else if i == preExistingIndices[k] {
					// combined = append(combined, vs[i])
					// debugf("Eliminating %#q", vs[i])
					k++
				} else {
					others = append(others, vs[i])
				}
			}
		} else {
			others = make([]string, len(vs)-(data.end-data.start))
			copy(others, vs[:data.start])
			copy(others[data.start:], vs[data.end:])
		}

		middle := render(shortRegexpString(varying, cache), true)
		// debugf(">> ShortRegexpString(%#q) = %#q", varying, middle)

		prefix, suffix = regexp.QuoteMeta(prefix), regexp.QuoteMeta(suffix)
		var cur string
		switch {
		case allExist && prefix == "": // M . S | M ==> M . S?
			cur = middle + optional(suffix)
		case allExist && suffix == "": // P . M | M ==> P? . M
			cur = optional(prefix) + middle
		default:
			cur = prefix + middle + suffix
		}
		return append([]string{cur}, shortRegexpString(others, cache)...)
	}

	// Note that vs is still sorted here.
	// debugf("Sorted: %#q", vs)
	for prefix, preLoc := range commonPrefixes(vs, 1) {
		suffix := sharedSuffix(len(prefix), vs[preLoc.start:preLoc.end])
		strs := recurse(prefix, suffix, preLoc)
		if c := cost(strs); c < bestCost { // || (c == len(best) && str < best) {
			best = strs
			bestCost = c
		} else {
			//debugf("! rejected %#q", str)
			//debugf("  because: %#q", best)
		}
	}

	sort.Sort(reverseStrings(vs))
	// debugf("Reverse-sorted: %#q", vs)
	for suffix, sufLoc := range commonSuffixes(vs, 1) {
		// sufLoc := suffixes[suffix]
		prefix := sharedPrefix(len(suffix), vs[sufLoc.start:sufLoc.end])
		strs := recurse(prefix, suffix, sufLoc)
		if c := cost(strs); c < bestCost { //|| (len(str) == len(best) && str < best) {
			best = strs
			bestCost = c
		} else {
			//debugf("! rejected %#q", str)
			//debugf("  because: %#q", best)
		}
	}

	singleChar := true
	optional := ""
	for i := range vs {
		if len(vs[i]) == 0 {
			optional = "?"
		} else if len(vs[i]) != 1 {
			// FIXME: should allow single non-ASCII characters
			singleChar = false
			break
		}
	}
	if singleChar {
		// Construct an array of characters in the right order:
		//   ']' first, '-' last, rest alphabetically
		class := make([]byte, 0, len(vs))
		last := ""
		for i, s := range vs {
			if s == "]" {
				// Must be first
				class = append(class, ']')
				vs[i] = "" // delete
			} else if s == "-" {
				// Must be last
				last = s
				vs[i] = "" // delete
			}
		}
		sortFirst := len(class)
		for _, s := range vs {
			class = append(class, s...)
		}
		sort.Sort(sortBytes(class[sortFirst:]))
		class = append(class, last...)

		// Collapse character ranges
		w := 0
		first := -1
		for i := 0; i < len(class); i++ {
			if first >= 0 {
				// Do we need to finish the range?
				if class[i] != class[i-1]+1 {
					// Does it pay to use a range?
					if i-first > 3 {
						// Build a range
						class[w-(i-first-1)] = '-'
						class[w-(i-first-1)+1] = class[i-1]
						// Rewind the write position
						w = w - (i - first - 1) + 2
						first = i
					}
				}
			} else {
				first = i
			}
			// Write the current character
			class[w] = class[i]
			w++
		}
		class = class[:w]

		if len(class) == 1 {
			str := regexp.QuoteMeta(string(class)) + optional
			if len(str) <= bestCost {
				best = []string{str}
				bestCost = len(str)
			}
		}
		if cost := len(class) + 2 + len(optional); cost <= bestCost {
			best = []string{"[" + string(class) + "]" + optional}
			bestCost = cost
		}
	}

	return best
}

func render(clauses []string, asSingle bool) string {
	switch len(clauses) {
	case 0:
		return "$.^" // Unmatchable?
	case 1:
		return clauses[0]
	default:
		if len(clauses[0]) == 0 {
			clauses = clauses[1:]
			if len(clauses) == 1 {
				return optional(clauses[0])
			}
			return render(clauses, true) + "?"
		}

		result := strings.Join(clauses, "|")
		if asSingle {
			result = "(" + result + ")"
		}
		return result
	}
}

func cost(clauses []string) int {
	// TODO: real implementation
	return len(render(clauses, false))
}

func optional(s string) string {
	if len(s) > 1 {
		s = "(" + s + ")?"
	} else if s != "" {
		s += "?"
	}
	return s
}

// removeDups removes duplicate strings from vs and returns it.
// It assumes that vs has been sorted such that duplicates are next to each
// other.
func removeDups(vs []string) []string {
	insertPos := 1
	for i := 1; i < len(vs); i++ {
		if vs[i-1] != vs[i] {
			vs[insertPos] = vs[i]
			insertPos++
		}
	}
	return vs[:insertPos]
}

func dup(vs []string) []string {
	result := make([]string, len(vs))
	copy(result, vs)
	return result
}

// reverseStrings is a sort.Interface that sort strings by their reverse values.
type reverseStrings []string

func (rs reverseStrings) Less(i, j int) bool {
	for m, n := len(rs[i])-1, len(rs[j])-1; m >= 0 && n >= 0; m, n = m-1, n-1 {
		if rs[i][m] != rs[j][n] {
			// We want to compare runes, not bytes. So find the start of the
			// current runes and decode them.
			for ; m > 0 && !utf8.RuneStart(rs[i][m]); m-- {
			}
			for ; n > 0 && !utf8.RuneStart(rs[j][n]); n-- {
			}
			ri, _ := utf8.DecodeRuneInString(rs[i][m:])
			rj, _ := utf8.DecodeRuneInString(rs[j][n:])
			return ri < rj
		}
	}
	return len(rs[i]) < len(rs[j])
}
func (rs reverseStrings) Swap(i, j int) { rs[i], rs[j] = rs[j], rs[i] }
func (rs reverseStrings) Len() int      { return len(rs) }

// sortBytes is a sort.Interface that sort bytes.
type sortBytes []byte

func (sb sortBytes) Less(i, j int) bool { return sb[i] < sb[j] }
func (sb sortBytes) Swap(i, j int)      { sb[i], sb[j] = sb[j], sb[i] }
func (sb sortBytes) Len() int           { return len(sb) }

// commonSs holds information on where to find a common substring.
type commonSs struct {
	start, end int
}

// commonPrefixes returns a map from prefixes to number of occurrences. Not all
// strings in vs need to have a prefix for it to be returned.
// Assumes vs to have been sorted with sort.Strings()
func commonPrefixes(vs []string, minLength int) (result map[string]commonSs) {
	result = make(map[string]commonSs)
	for i := 0; i < len(vs)-1; i++ {
		j := i + 1
		k := 0
		for ; k < len(vs[i]) && k < len(vs[j]); k++ {
			if vs[i][k] != vs[j][k] {
				break
			}
		}
		if k < minLength {
			continue
		}
		prefix := vs[i][:k]
		if _, exists := result[prefix]; !exists {
			first := prefixStart(vs[:i], prefix)
			//debugf("prefixStart(%#q, %#q) == %v", vs[:i], prefix, first)
			// prefixEnd(vs, prefix) - first + 1
			// == prefixEnd(vs[first:], prefix) + 1
			// == prefixEnd(vs[first+1:], prefix) + 2
			end := first + 1 + prefixEnd(vs[first+1:], prefix)
			result[prefix] = commonSs{
				first, end,
			}
			//debugf("prefixEnd(%#q, %#q) == %v", vs, prefix, result[prefix].end)
		}
	}
	// debugf("# %v..", result)
	return result
}

func prefixStart(vs []string, prefix string) int {
	if prefix == "" {
		return 0
	}
	return findFirst(vs, func(s string) bool {
		return strings.HasPrefix(s, prefix)
	})
}

func prefixEnd(vs []string, prefix string) int {
	if prefix == "" {
		return len(vs)
	}
	//debugf("prefixEnd(%v, %#q)", vs, prefix)
	return findFirst(vs, func(s string) bool {
		return !strings.HasPrefix(s, prefix)
	})
}

// commonSuffixes returns a map from suffixes to number of occurrences. Not all
// strings in vs need to have a suffix for it to be returned.
// Assumes vs to have been sorted using sort.Sort(reverseStrings(vs))
func commonSuffixes(vs []string, minLength int) (result map[string]commonSs) {
	result = make(map[string]commonSs)
	for i := 0; i < len(vs)-1; i++ {
		j := i + 1
		k := 0
		for ; k < len(vs[i]) && k < len(vs[j]); k++ {
			if vs[i][len(vs[i])-k-1] != vs[j][len(vs[j])-k-1] {
				break
			}
		}
		if k < minLength {
			continue
		}
		suffix := vs[i][len(vs[i])-k:]
		if _, exists := result[suffix]; !exists {
			first := suffixStart(vs[:i], suffix)
			//debugf("suffixStart<%#q>(%#q) == %v", suffix, vs[:i], first)
			// suffixEnd(vs, suffix) - first + 1
			// == suffixEnd(vs[first:], suffix) + 1
			// == suffixEnd(vs[first+1:], suffix) + 2
			end := first + 1 + suffixEnd(vs[first+1:], suffix)
			result[suffix] = commonSs{
				first, end,
			}
			//debugf("suffixEnd  <%#q>(%#q) == %v", suffix, vs, result[suffix].end)
			//debugf("selected(%#q): %q\n\n", suffix, vs[first:result[suffix].end])
		}
	}
	// debugf("# ..%v", result)
	return result
}

func suffixStart(vs []string, suffix string) int {
	// //debugf("suffixStart(%#q, %#q)", vs, suffix)
	if suffix == "" {
		return 0
	}
	return findFirst(vs, func(s string) bool {
		return strings.HasSuffix(s, suffix)
	})
}

func suffixEnd(vs []string, suffix string) int {
	// //debugf("suffixEnd  (%#q, %#q)", vs, suffix)
	if suffix == "" {
		return len(vs)
	}
	return findFirst(vs, func(s string) bool {
		return !strings.HasSuffix(s, suffix)
	})
}

// findFirst finds the first element of vs that satisfies the predicate.
// It assumes that the first N strings don't match the predicate, and the rest
// do. If all of the strings satisfy the predicate, it returns 0, and if none
// do it returns len(vs).
func findFirst(vs []string, predicate func(string) bool) int {
	l, h := -1, len(vs)
	// Invariant: vs[l] does not match, vs[h] does.
	// -1 and len(vs) are sentinal values, never tested but assumed to mismatch and match, respectively.
	for l+1 < h {
		m := (l + h) / 2 // Must now be a valid value
		// //debugf("%d %d %d", l, m, h)
		if predicate(vs[m]) {
			h = m
		} else {
			l = m
		}
	}
	//debugf("==> %d", h)
	return h
}

// sharedPrefix returns the longest prefix which all the parameters share but
// ignores a number of characters at the end of each string.
func sharedPrefix(ignore int, vs []string) (result string) {
	//debugf("sharedPrefix(%d, %#q)", ignore, vs)
	// defer func() {
	//debugf("==> %#q", result)
	// }()
	switch len(vs) {
	case 0:
		return ""
	case 1:
		return vs[0]
	}
	for i := 0; i < len(vs[0])-ignore; i++ {
		for n := 1; n < len(vs); n++ {
			if i >= len(vs[n])-ignore || vs[0][i] != vs[n][i] {
				return vs[0][:i]
			}
		}
	}
	return vs[0][:len(vs[0])-ignore]
}

// sharedSuffix returns the longest suffix which all the parameters share but
// ignores a number of characters at the start of each string.
func sharedSuffix(ignore int, vs []string) (result string) {
	//debugf("sharedSuffix(%d, %#q)", ignore, vs)
	// defer func() {
	//debugf("==> %#q", result)
	// }()
	switch len(vs) {
	case 0:
		return ""
	case 1:
		return vs[0]
	}
	first := vs[0]
	for i := 0; i < len(first)-ignore; i++ {
		for n := 1; n < len(vs); n++ {
			cur := vs[n]
			if i == len(cur)-ignore {
				return cur[ignore:]
			}
			if first[len(first)-i-1] != cur[len(cur)-i-1] {
				return first[len(first)-i:]
			}
		}
	}
	return first[ignore:]
}
