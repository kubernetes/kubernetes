// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package env

import (
	"math"
	"sort"
	"strings"
)

// SymbolKind indicates the kind of declaration provided in an environment.
type SymbolKind string

const (
	// FunctionKind represents a function declaration.
	FunctionKind SymbolKind = "function"

	// MacroKind represents a macro declaration.
	MacroKind SymbolKind = "macro"

	// TypeKind represents a type declaration.
	TypeKind SymbolKind = "type"

	// NamespaceKind represents a namespace declaration.
	NamespaceKind SymbolKind = "namespace"

	// VariableKind represents a variable declaration.
	VariableKind SymbolKind = "variable"
)

// CatalogSymbol represents a symbol in the environment catalog.
type CatalogSymbol struct {
	Name       string     `yaml:"name" json:"name"`
	Kind       SymbolKind `yaml:"kind" json:"kind"`
	Library    string     `yaml:"library,omitempty" json:"library,omitempty"`
	Option     string     `yaml:"option,omitempty" json:"option,omitempty"`
	MinVersion uint32     `yaml:"min_version,omitempty" json:"minVersion,omitempty"`
}

// Catalog holds an index of symbols from the active environment and available libraries.
type Catalog struct {
	symbols []*CatalogSymbol
	byName  map[string]*CatalogSymbol
}

// NewCatalog creates a new Catalog initialized with the given symbols.
func NewCatalog(symbols ...*CatalogSymbol) *Catalog {
	c := &Catalog{
		symbols: make([]*CatalogSymbol, 0, len(symbols)),
		byName:  make(map[string]*CatalogSymbol, len(symbols)),
	}
	for _, sym := range symbols {
		c.Add(sym)
	}
	return c
}

// Add adds one or more symbols to the catalog.
func (c *Catalog) Add(syms ...*CatalogSymbol) {
	for _, sym := range syms {
		if sym == nil || sym.Name == "" {
			continue
		}
		c.symbols = append(c.symbols, sym)
		c.byName[sym.Name] = sym
	}
}

// Find searches for symbols in the catalog matching the given name.
//
// If an exact match exists, a single-element slice containing that symbol is returned.
// Otherwise, candidate symbols matching the namespace structure (namespaced vs non-namespaced)
// are evaluated using Damerau-Levenshtein edit distance within the following thresholds:
//   - length <= 3: max distance 1
//   - 4 <= length <= 8: max distance 2
//   - length > 8: max distance length / 3
//
// If candidates meet the edit distance threshold, up to two entries with the lowest edit distance
// are returned, ordered from shortest to longest suggestion (and alphabetically on tie).
//
// If no edit distance matches are found, prefix matching is performed as a fallback. When a prefix
// matches multiple symbols, up to two candidates are returned, ordered from shortest to longest.
func (c *Catalog) Find(name string) []*CatalogSymbol {
	if c == nil || name == "" {
		return nil
	}
	if sym, ok := c.byName[name]; ok {
		return []*CatalogSymbol{sym}
	}
	if len(c.symbols) == 0 {
		return nil
	}

	hasDot := strings.Contains(name, ".")
	maxDist := 2
	if len(name) <= 3 {
		maxDist = 1
	} else if len(name) > 8 {
		maxDist = len(name) / 3
	}

	type match struct {
		sym  *CatalogSymbol
		dist int
	}
	var editMatches []match
	minDist := math.MaxInt32
	seen := make(map[string]bool)

	for _, sym := range c.symbols {
		if seen[sym.Name] {
			continue
		}
		// If the input might be namespaced, only consider other namespaced symbols.
		if strings.Contains(sym.Name, ".") != hasDot {
			continue
		}
		seen[sym.Name] = true
		dist := EditDistance(name, sym.Name)
		if dist <= maxDist {
			editMatches = append(editMatches, match{sym: sym, dist: dist})
			if dist < minDist {
				minDist = dist
			}
		}
	}

	if len(editMatches) > 0 {
		var best []match
		for _, m := range editMatches {
			if m.dist == minDist {
				best = append(best, m)
			}
		}
		sort.Slice(best, func(i, j int) bool {
			if len(best[i].sym.Name) != len(best[j].sym.Name) {
				return len(best[i].sym.Name) < len(best[j].sym.Name)
			}
			return best[i].sym.Name < best[j].sym.Name
		})
		resultCount := min(len(best), 2)
		res := make([]*CatalogSymbol, resultCount)
		for i := range res {
			res[i] = best[i].sym
		}
		return res
	}

	// Fallback: prefix match (require at least 2 characters)
	if len(name) >= 2 {
		seenPrefix := make(map[string]bool)
		var prefixMatches []*CatalogSymbol
		for _, sym := range c.symbols {
			if seenPrefix[sym.Name] {
				continue
			}
			if strings.Contains(sym.Name, ".") != hasDot {
				continue
			}
			if strings.HasPrefix(sym.Name, name) {
				seenPrefix[sym.Name] = true
				prefixMatches = append(prefixMatches, sym)
			}
		}

		if len(prefixMatches) > 0 {
			sort.Slice(prefixMatches, func(i, j int) bool {
				if len(prefixMatches[i].Name) != len(prefixMatches[j].Name) {
					return len(prefixMatches[i].Name) < len(prefixMatches[j].Name)
				}
				return prefixMatches[i].Name < prefixMatches[j].Name
			})
			if len(prefixMatches) > 2 {
				prefixMatches = prefixMatches[:2]
			}
			return prefixMatches
		}
	}

	return nil
}

const (
	maxEditDistanceIterations = 64 * 64
)

// EditDistance calculates the Damerau-Levenshtein distance between two strings,
// taking into account insertions, deletions, substitutions, and adjacent transpositions.
//
// When evaluating string similarity for suggestions, the following distance thresholds are recommended:
//   - length <= 3: max distance 1
//   - 4 <= length <= 8: max distance 2
//   - length > 8: max distance length / 3
func EditDistance(a, b string) int {
	ra, rb := []rune(a), []rune(b)
	la, lb := len(ra), len(rb)
	if la == 0 {
		return lb
	}
	if lb == 0 {
		return la
	}
	if la >= 16 && la*2 >= lb*3 {
		return la - lb
	}
	if lb >= 16 && lb*2 >= la*3 {
		return lb - la
	}
	if la*lb > maxEditDistanceIterations {
		return math.MaxInt32
	}

	maxLen := max(la, lb)
	threshold := maxLen / 3

	d := make([][]int, la+1)
	for i := range d {
		d[i] = make([]int, lb+1)
		d[i][0] = i
	}
	for j := 0; j <= lb; j++ {
		d[0][j] = j
	}

	for i := 1; i <= la; i++ {
		rowMin := d[i][0]
		for j := 1; j <= lb; j++ {
			cost := 1
			if ra[i-1] == rb[j-1] {
				cost = 0
			}
			d[i][j] = min(
				d[i-1][j]+1,      // deletion
				d[i][j-1]+1,      // insertion
				d[i-1][j-1]+cost, // substitution
			)
			if i > 1 && j > 1 && ra[i-1] == rb[j-2] && ra[i-2] == rb[j-1] {
				d[i][j] = min(d[i][j], d[i-2][j-2]+1) // transposition
			}
			if d[i][j] < rowMin {
				rowMin = d[i][j]
			}
		}
		if rowMin > threshold {
			return math.MaxInt32
		}
	}
	return d[la][lb]
}

// ToCatalog converts the Config to a Catalog containing its functions, variables, container, and imports.
func (c *Config) ToCatalog() *Catalog {
	cat := NewCatalog()
	if c == nil {
		return cat
	}
	for _, v := range c.Variables {
		cat.Add(&CatalogSymbol{
			Name: v.Name,
			Kind: VariableKind,
		})
	}
	for _, f := range c.Functions {
		cat.Add(&CatalogSymbol{
			Name: f.Name,
			Kind: FunctionKind,
		})
	}
	if c.Container != "" {
		cat.Add(&CatalogSymbol{
			Name: c.Container,
			Kind: NamespaceKind,
		})
	}
	for _, imp := range c.Imports {
		cat.Add(&CatalogSymbol{
			Name: imp.Name,
			Kind: NamespaceKind,
		})
	}
	return cat
}
