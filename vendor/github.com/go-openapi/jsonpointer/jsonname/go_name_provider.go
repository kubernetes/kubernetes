// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package jsonname

import (
	"reflect"
	"strings"
	"sync"
)

var _ providerIface = (*GoNameProvider)(nil)

// GoNameProvider resolves json property names to go struct field names following the same rules as
// the standard library's [encoding/json] package.
//
// Contrary to [NameProvider], it considers exported fields without a json tag, and promotes fields
// from anonymous embedded struct types.
//
// Rules (aligned with encoding/json):
//
//   - unexported fields are ignored;
//   - a field tagged `json:"-"` is ignored;
//   - a field tagged `json:"-,"` is kept under the json name "-" (stdlib quirk);
//   - a field tagged `json:""` or with no json tag at all keeps its Go name as json name;
//   - anonymous struct fields without an explicit json tag have their fields
//     promoted into the parent, following breadth-first depth rules:
//     a shallower field wins over a deeper one; at equal depth, a conflict
//     discards all conflicting fields unless exactly one has an explicit json tag.
//
// This type is safe for concurrent use.
type GoNameProvider struct {
	lock  sync.Mutex
	index map[reflect.Type]nameIndex
}

// NewGoNameProvider creates a new [GoNameProvider].
func NewGoNameProvider() *GoNameProvider {
	return &GoNameProvider{
		index: make(map[reflect.Type]nameIndex),
	}
}

// GetJSONNames gets all the json property names for a type.
func (n *GoNameProvider) GetJSONNames(subject any) []string {
	n.lock.Lock()
	defer n.lock.Unlock()

	tpe := reflect.Indirect(reflect.ValueOf(subject)).Type()
	names := n.nameIndexFor(tpe)

	res := make([]string, 0, len(names.jsonNames))
	for k := range names.jsonNames {
		res = append(res, k)
	}

	return res
}

// GetJSONName gets the json name for a go property name.
func (n *GoNameProvider) GetJSONName(subject any, name string) (string, bool) {
	tpe := reflect.Indirect(reflect.ValueOf(subject)).Type()

	return n.GetJSONNameForType(tpe, name)
}

// GetJSONNameForType gets the json name for a go property name on a given type.
func (n *GoNameProvider) GetJSONNameForType(tpe reflect.Type, name string) (string, bool) {
	n.lock.Lock()
	defer n.lock.Unlock()

	names := n.nameIndexFor(tpe)
	nme, ok := names.goNames[name]

	return nme, ok
}

// GetGoName gets the go name for a json property name.
func (n *GoNameProvider) GetGoName(subject any, name string) (string, bool) {
	tpe := reflect.Indirect(reflect.ValueOf(subject)).Type()

	return n.GetGoNameForType(tpe, name)
}

// GetGoNameForType gets the go name for a given type for a json property name.
func (n *GoNameProvider) GetGoNameForType(tpe reflect.Type, name string) (string, bool) {
	n.lock.Lock()
	defer n.lock.Unlock()

	names := n.nameIndexFor(tpe)
	nme, ok := names.jsonNames[name]

	return nme, ok
}

func (n *GoNameProvider) nameIndexFor(tpe reflect.Type) nameIndex {
	if names, ok := n.index[tpe]; ok {
		return names
	}

	names := buildGoNameIndex(tpe)
	n.index[tpe] = names

	return names
}

// fieldEntry captures a candidate field discovered while walking a struct along with the
// indirection path from the root type (used to resolve conflicts by depth in the same way
// encoding/json does).
type fieldEntry struct {
	goName   string
	jsonName string
	index    []int
	tagged   bool
}

func buildGoNameIndex(tpe reflect.Type) nameIndex {
	fields := collectGoFields(tpe)

	idx := make(map[string]string, len(fields))
	reverseIdx := make(map[string]string, len(fields))
	for _, f := range fields {
		idx[f.jsonName] = f.goName
		reverseIdx[f.goName] = f.jsonName
	}

	return nameIndex{jsonNames: idx, goNames: reverseIdx}
}

// collectGoFields walks tpe breadth-first along anonymous struct fields,
// reproducing the field selection performed by encoding/json.typeFields.
//
//nolint:gocognit // everything is inlined to help the compiler determine what escapes and what doesn't
func collectGoFields(tpe reflect.Type) []fieldEntry {
	if tpe.Kind() != reflect.Struct {
		return nil
	}

	type queued struct {
		typ   reflect.Type
		index []int
	}

	current := []queued{}
	next := []queued{{typ: tpe}}
	visited := map[reflect.Type]bool{tpe: true}

	var (
		candidates []fieldEntry
		count      = map[string]int{}
		nextCount  = map[string]int{}
	)

	for len(next) > 0 {
		current, next = next, current[:0]
		count, nextCount = nextCount, count
		for k := range nextCount {
			delete(nextCount, k)
		}

		for _, q := range current {
			for i := range q.typ.NumField() {
				sf := q.typ.Field(i)

				if sf.Anonymous {
					ft := sf.Type
					if ft.Kind() == reflect.Pointer {
						ft = ft.Elem()
					}
					if !sf.IsExported() && ft.Kind() != reflect.Struct {
						continue
					}
				} else if !sf.IsExported() {
					continue
				}

				tag := sf.Tag.Get("json")
				if tag == "-" {
					continue
				}
				jsonName, _ := parseJSONTag(tag)
				tagged := jsonName != ""

				ft := sf.Type
				if ft.Kind() == reflect.Pointer {
					ft = ft.Elem()
				}

				if sf.Anonymous && ft.Kind() == reflect.Struct && !tagged {
					if visited[ft] {
						continue
					}
					visited[ft] = true

					index := make([]int, len(q.index)+1)
					copy(index, q.index)
					index[len(q.index)] = i
					next = append(next, queued{typ: ft, index: index})

					continue
				}

				name := jsonName
				if name == "" {
					name = sf.Name
				}

				index := make([]int, len(q.index)+1)
				copy(index, q.index)
				index[len(q.index)] = i

				candidates = append(candidates, fieldEntry{
					goName:   sf.Name,
					jsonName: name,
					index:    index,
					tagged:   tagged,
				})
				nextCount[name]++
			}
		}
	}

	return dominantFields(candidates)
}

// dominantFields applies the Go encoding/json conflict resolution rules: at each JSON name, the
// shallowest field wins; at equal depth, a uniquely tagged candidate wins; otherwise all candidates
// for that name are dropped.
func dominantFields(candidates []fieldEntry) []fieldEntry {
	byName := make(map[string][]fieldEntry, len(candidates))
	for _, c := range candidates {
		byName[c.jsonName] = append(byName[c.jsonName], c)
	}

	out := make([]fieldEntry, 0, len(byName))
	for _, group := range byName {
		if len(group) == 1 {
			out = append(out, group[0])

			continue
		}

		minDepth := len(group[0].index)
		for _, c := range group[1:] {
			if len(c.index) < minDepth {
				minDepth = len(c.index)
			}
		}

		var shallow []fieldEntry
		for _, c := range group {
			if len(c.index) == minDepth {
				shallow = append(shallow, c)
			}
		}

		if len(shallow) == 1 {
			out = append(out, shallow[0])

			continue
		}

		var tagged []fieldEntry
		for _, c := range shallow {
			if c.tagged {
				tagged = append(tagged, c)
			}
		}
		if len(tagged) == 1 {
			out = append(out, tagged[0])
		}
	}

	return out
}

// parseJSONTag returns the name component of a json struct tag and whether it carried any non-name
// option (kept for future-proofing, e.g. "omitempty").
func parseJSONTag(tag string) (string, string) {
	if tag == "" {
		return "", ""
	}
	if before, after, ok := strings.Cut(tag, ","); ok {
		return before, after
	}

	return tag, ""
}
