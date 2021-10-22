package dynamodbattribute

import (
	"reflect"
	"sort"
	"strings"
)

type field struct {
	tag

	Name        string
	NameFromTag bool

	Index []int
	Type  reflect.Type
}

type cachedFields struct {
	fields       []field
	fieldsByName map[string]int
}

func (f *cachedFields) All() []field {
	return f.fields
}

func (f *cachedFields) FieldByName(name string) (field, bool) {
	if i, ok := f.fieldsByName[name]; ok {
		return f.fields[i], ok
	}
	for _, f := range f.fields {
		if strings.EqualFold(f.Name, name) {
			return f, true
		}
	}
	return field{}, false
}

func buildField(pIdx []int, i int, sf reflect.StructField, fieldTag tag) field {
	f := field{
		Name: sf.Name,
		Type: sf.Type,
		tag:  fieldTag,
	}
	if len(fieldTag.Name) != 0 {
		f.NameFromTag = true
		f.Name = fieldTag.Name
	}

	f.Index = make([]int, len(pIdx)+1)
	copy(f.Index, pIdx)
	f.Index[len(pIdx)] = i

	return f
}

// unionStructFields returns a list of fields for the given type. Type info is cached
// to avoid repeated calls into the reflect package
func unionStructFields(t reflect.Type, opts MarshalOptions) *cachedFields {
	if cached, ok := fieldCache.Load(t); ok {
		return cached
	}

	f := enumFields(t, opts)
	sort.Sort(fieldsByName(f))
	f = visibleFields(f)

	fs := &cachedFields{
		fields:       f,
		fieldsByName: make(map[string]int, len(f)),
	}
	for i, f := range fs.fields {
		fs.fieldsByName[f.Name] = i
	}

	cached, _ := fieldCache.LoadOrStore(t, fs)
	return cached
}

// enumFields will recursively iterate through a structure and its nested
// anonymous fields.
//
// Based on the enoding/json struct field enumeration of the Go Stdlib
// https://golang.org/src/encoding/json/encode.go typeField func.
func enumFields(t reflect.Type, opts MarshalOptions) []field {
	// Fields to explore
	current := []field{}
	next := []field{{Type: t}}

	// count of queued names
	count := map[reflect.Type]int{}
	nextCount := map[reflect.Type]int{}

	visited := map[reflect.Type]struct{}{}
	fields := []field{}

	for len(next) > 0 {
		current, next = next, current[:0]
		count, nextCount = nextCount, map[reflect.Type]int{}

		for _, f := range current {
			if _, ok := visited[f.Type]; ok {
				continue
			}
			visited[f.Type] = struct{}{}

			for i := 0; i < f.Type.NumField(); i++ {
				sf := f.Type.Field(i)
				if sf.PkgPath != "" && !sf.Anonymous {
					// Ignore unexported and non-anonymous fields
					// unexported but anonymous field may still be used if
					// the type has exported nested fields
					continue
				}

				fieldTag := tag{}
				fieldTag.parseAVTag(sf.Tag)
				// Because MarshalOptions.TagKey must be explicitly set, use it
				// over JSON, which is enabled by default.
				if opts.TagKey != "" && fieldTag == (tag{}) {
					fieldTag.parseStructTag(opts.TagKey, sf.Tag)
				} else if opts.SupportJSONTags && fieldTag == (tag{}) {
					fieldTag.parseStructTag("json", sf.Tag)
				}

				if fieldTag.Ignore {
					continue
				}

				ft := sf.Type
				if ft.Name() == "" && ft.Kind() == reflect.Ptr {
					ft = ft.Elem()
				}

				structField := buildField(f.Index, i, sf, fieldTag)
				structField.Type = ft

				if !sf.Anonymous || ft.Kind() != reflect.Struct {
					fields = append(fields, structField)
					if count[f.Type] > 1 {
						// If there were multiple instances, add a second,
						// so that the annihilation code will see a duplicate.
						// It only cares about the distinction between 1 or 2,
						// so don't bother generating any more copies.
						fields = append(fields, structField)
					}
					continue
				}

				// Record new anon struct to explore next round
				nextCount[ft]++
				if nextCount[ft] == 1 {
					next = append(next, structField)
				}
			}
		}
	}

	return fields
}

// visibleFields will return a slice of fields which are visible based on
// Go's standard visiblity rules with the exception of ties being broken
// by depth and struct tag naming.
//
// Based on the enoding/json field filtering of the Go Stdlib
// https://golang.org/src/encoding/json/encode.go typeField func.
func visibleFields(fields []field) []field {
	// Delete all fields that are hidden by the Go rules for embedded fields,
	// except that fields with JSON tags are promoted.

	// The fields are sorted in primary order of name, secondary order
	// of field index length. Loop over names; for each name, delete
	// hidden fields by choosing the one dominant field that survives.
	out := fields[:0]
	for advance, i := 0, 0; i < len(fields); i += advance {
		// One iteration per name.
		// Find the sequence of fields with the name of this first field.
		fi := fields[i]
		name := fi.Name
		for advance = 1; i+advance < len(fields); advance++ {
			fj := fields[i+advance]
			if fj.Name != name {
				break
			}
		}
		if advance == 1 { // Only one field with this name
			out = append(out, fi)
			continue
		}
		dominant, ok := dominantField(fields[i : i+advance])
		if ok {
			out = append(out, dominant)
		}
	}

	fields = out
	sort.Sort(fieldsByIndex(fields))

	return fields
}

// dominantField looks through the fields, all of which are known to
// have the same name, to find the single field that dominates the
// others using Go's embedding rules, modified by the presence of
// JSON tags. If there are multiple top-level fields, the boolean
// will be false: This condition is an error in Go and we skip all
// the fields.
//
// Based on the enoding/json field filtering of the Go Stdlib
// https://golang.org/src/encoding/json/encode.go dominantField func.
func dominantField(fields []field) (field, bool) {
	// The fields are sorted in increasing index-length order. The winner
	// must therefore be one with the shortest index length. Drop all
	// longer entries, which is easy: just truncate the slice.
	length := len(fields[0].Index)
	tagged := -1 // Index of first tagged field.
	for i, f := range fields {
		if len(f.Index) > length {
			fields = fields[:i]
			break
		}
		if f.NameFromTag {
			if tagged >= 0 {
				// Multiple tagged fields at the same level: conflict.
				// Return no field.
				return field{}, false
			}
			tagged = i
		}
	}
	if tagged >= 0 {
		return fields[tagged], true
	}
	// All remaining fields have the same length. If there's more than one,
	// we have a conflict (two fields named "X" at the same level) and we
	// return no field.
	if len(fields) > 1 {
		return field{}, false
	}
	return fields[0], true
}

// fieldsByName sorts field by name, breaking ties with depth,
// then breaking ties with "name came from json tag", then
// breaking ties with index sequence.
//
// Based on the enoding/json field filtering of the Go Stdlib
// https://golang.org/src/encoding/json/encode.go fieldsByName type.
type fieldsByName []field

func (x fieldsByName) Len() int { return len(x) }

func (x fieldsByName) Swap(i, j int) { x[i], x[j] = x[j], x[i] }

func (x fieldsByName) Less(i, j int) bool {
	if x[i].Name != x[j].Name {
		return x[i].Name < x[j].Name
	}
	if len(x[i].Index) != len(x[j].Index) {
		return len(x[i].Index) < len(x[j].Index)
	}
	if x[i].NameFromTag != x[j].NameFromTag {
		return x[i].NameFromTag
	}
	return fieldsByIndex(x).Less(i, j)
}

// fieldsByIndex sorts field by index sequence.
//
// Based on the enoding/json field filtering of the Go Stdlib
// https://golang.org/src/encoding/json/encode.go fieldsByIndex type.
type fieldsByIndex []field

func (x fieldsByIndex) Len() int { return len(x) }

func (x fieldsByIndex) Swap(i, j int) { x[i], x[j] = x[j], x[i] }

func (x fieldsByIndex) Less(i, j int) bool {
	for k, xik := range x[i].Index {
		if k >= len(x[j].Index) {
			return false
		}
		if xik != x[j].Index[k] {
			return xik < x[j].Index[k]
		}
	}
	return len(x[i].Index) < len(x[j].Index)
}
