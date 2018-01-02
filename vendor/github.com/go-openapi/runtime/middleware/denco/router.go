// Package denco provides fast URL router.
package denco

import (
	"fmt"
	"sort"
	"strings"
)

const (
	// ParamCharacter is a special character for path parameter.
	ParamCharacter = ':'

	// WildcardCharacter is a special character for wildcard path parameter.
	WildcardCharacter = '*'

	// TerminationCharacter is a special character for end of path.
	TerminationCharacter = '#'

	// MaxSize is max size of records and internal slice.
	MaxSize = (1 << 22) - 1
)

// Router represents a URL router.
type Router struct {
	// SizeHint expects the maximum number of path parameters in records to Build.
	// SizeHint will be used to determine the capacity of the memory to allocate.
	// By default, SizeHint will be determined from given records to Build.
	SizeHint int

	static map[string]interface{}
	param  *doubleArray
}

// New returns a new Router.
func New() *Router {
	return &Router{
		SizeHint: -1,
		static:   make(map[string]interface{}),
		param:    newDoubleArray(),
	}
}

// Lookup returns data and path parameters that associated with path.
// params is a slice of the Param that arranged in the order in which parameters appeared.
// e.g. when built routing path is "/path/:id/:name" and given path is "/path/to/1/alice". params order is [{"id": "1"}, {"name": "alice"}], not [{"name": "alice"}, {"id": "1"}].
func (rt *Router) Lookup(path string) (data interface{}, params Params, found bool) {
	if data, found := rt.static[path]; found {
		return data, nil, true
	}
	if len(rt.param.node) == 1 {
		return nil, nil, false
	}
	nd, params, found := rt.param.lookup(path, make([]Param, 0, rt.SizeHint), 1)
	if !found {
		return nil, nil, false
	}
	for i := 0; i < len(params); i++ {
		params[i].Name = nd.paramNames[i]
	}
	return nd.data, params, true
}

// Build builds URL router from records.
func (rt *Router) Build(records []Record) error {
	statics, params := makeRecords(records)
	if len(params) > MaxSize {
		return fmt.Errorf("denco: too many records")
	}
	if rt.SizeHint < 0 {
		rt.SizeHint = 0
		for _, p := range params {
			size := 0
			for _, k := range p.Key {
				if k == ParamCharacter || k == WildcardCharacter {
					size++
				}
			}
			if size > rt.SizeHint {
				rt.SizeHint = size
			}
		}
	}
	for _, r := range statics {
		rt.static[r.Key] = r.Value
	}
	if err := rt.param.build(params, 1, 0, make(map[int]struct{})); err != nil {
		return err
	}
	return nil
}

// Param represents name and value of path parameter.
type Param struct {
	Name  string
	Value string
}

// Params represents the name and value of path parameters.
type Params []Param

// Get gets the first value associated with the given name.
// If there are no values associated with the key, Get returns "".
func (ps Params) Get(name string) string {
	for _, p := range ps {
		if p.Name == name {
			return p.Value
		}
	}
	return ""
}

type doubleArray struct {
	bc   []baseCheck
	node []*node
}

func newDoubleArray() *doubleArray {
	return &doubleArray{
		bc:   []baseCheck{0},
		node: []*node{nil}, // A start index is adjusting to 1 because 0 will be used as a mark of non-existent node.
	}
}

// baseCheck contains BASE, CHECK and Extra flags.
// From the top, 22bits of BASE, 2bits of Extra flags and 8bits of CHECK.
//
//  BASE (22bit) | Extra flags (2bit) | CHECK (8bit)
// |----------------------|--|--------|
// 32                    10  8         0
type baseCheck uint32

func (bc baseCheck) Base() int {
	return int(bc >> 10)
}

func (bc *baseCheck) SetBase(base int) {
	*bc |= baseCheck(base) << 10
}

func (bc baseCheck) Check() byte {
	return byte(bc)
}

func (bc *baseCheck) SetCheck(check byte) {
	*bc |= baseCheck(check)
}

func (bc baseCheck) IsEmpty() bool {
	return bc&0xfffffcff == 0
}

func (bc baseCheck) IsSingleParam() bool {
	return bc&paramTypeSingle == paramTypeSingle
}

func (bc baseCheck) IsWildcardParam() bool {
	return bc&paramTypeWildcard == paramTypeWildcard
}

func (bc baseCheck) IsAnyParam() bool {
	return bc&paramTypeAny != 0
}

func (bc *baseCheck) SetSingleParam() {
	*bc |= (1 << 8)
}

func (bc *baseCheck) SetWildcardParam() {
	*bc |= (1 << 9)
}

const (
	paramTypeSingle   = 0x0100
	paramTypeWildcard = 0x0200
	paramTypeAny      = 0x0300
)

func (da *doubleArray) lookup(path string, params []Param, idx int) (*node, []Param, bool) {
	indices := make([]uint64, 0, 1)
	for i := 0; i < len(path); i++ {
		if da.bc[idx].IsAnyParam() {
			indices = append(indices, (uint64(i)<<32)|(uint64(idx)&0xffffffff))
		}
		c := path[i]
		if idx = nextIndex(da.bc[idx].Base(), c); idx >= len(da.bc) || da.bc[idx].Check() != c {
			goto BACKTRACKING
		}
	}
	if next := nextIndex(da.bc[idx].Base(), TerminationCharacter); next < len(da.bc) && da.bc[next].Check() == TerminationCharacter {
		return da.node[da.bc[next].Base()], params, true
	}
BACKTRACKING:
	for j := len(indices) - 1; j >= 0; j-- {
		i, idx := int(indices[j]>>32), int(indices[j]&0xffffffff)
		if da.bc[idx].IsSingleParam() {
			idx := nextIndex(da.bc[idx].Base(), ParamCharacter)
			if idx >= len(da.bc) {
				break
			}
			next := NextSeparator(path, i)
			params := append(params, Param{Value: path[i:next]})
			if nd, params, found := da.lookup(path[next:], params, idx); found {
				return nd, params, true
			}
		}
		if da.bc[idx].IsWildcardParam() {
			idx := nextIndex(da.bc[idx].Base(), WildcardCharacter)
			params := append(params, Param{Value: path[i:]})
			return da.node[da.bc[idx].Base()], params, true
		}
	}
	return nil, nil, false
}

// build builds double-array from records.
func (da *doubleArray) build(srcs []*record, idx, depth int, usedBase map[int]struct{}) error {
	sort.Stable(recordSlice(srcs))
	base, siblings, leaf, err := da.arrange(srcs, idx, depth, usedBase)
	if err != nil {
		return err
	}
	if leaf != nil {
		nd, err := makeNode(leaf)
		if err != nil {
			return err
		}
		da.bc[idx].SetBase(len(da.node))
		da.node = append(da.node, nd)
	}
	for _, sib := range siblings {
		da.setCheck(nextIndex(base, sib.c), sib.c)
	}
	for _, sib := range siblings {
		records := srcs[sib.start:sib.end]
		switch sib.c {
		case ParamCharacter:
			for _, r := range records {
				next := NextSeparator(r.Key, depth+1)
				name := r.Key[depth+1 : next]
				r.paramNames = append(r.paramNames, name)
				r.Key = r.Key[next:]
			}
			da.bc[idx].SetSingleParam()
			if err := da.build(records, nextIndex(base, sib.c), 0, usedBase); err != nil {
				return err
			}
		case WildcardCharacter:
			r := records[0]
			name := r.Key[depth+1 : len(r.Key)-1]
			r.paramNames = append(r.paramNames, name)
			r.Key = ""
			da.bc[idx].SetWildcardParam()
			if err := da.build(records, nextIndex(base, sib.c), 0, usedBase); err != nil {
				return err
			}
		default:
			if err := da.build(records, nextIndex(base, sib.c), depth+1, usedBase); err != nil {
				return err
			}
		}
	}
	return nil
}

// setBase sets BASE.
func (da *doubleArray) setBase(i, base int) {
	da.bc[i].SetBase(base)
}

// setCheck sets CHECK.
func (da *doubleArray) setCheck(i int, check byte) {
	da.bc[i].SetCheck(check)
}

// findEmptyIndex returns an index of unused BASE/CHECK node.
func (da *doubleArray) findEmptyIndex(start int) int {
	i := start
	for ; i < len(da.bc); i++ {
		if da.bc[i].IsEmpty() {
			break
		}
	}
	return i
}

// findBase returns good BASE.
func (da *doubleArray) findBase(siblings []sibling, start int, usedBase map[int]struct{}) (base int) {
	for idx, firstChar := start+1, siblings[0].c; ; idx = da.findEmptyIndex(idx + 1) {
		base = nextIndex(idx, firstChar)
		if _, used := usedBase[base]; used {
			continue
		}
		i := 0
		for ; i < len(siblings); i++ {
			next := nextIndex(base, siblings[i].c)
			if len(da.bc) <= next {
				da.bc = append(da.bc, make([]baseCheck, next-len(da.bc)+1)...)
			}
			if !da.bc[next].IsEmpty() {
				break
			}
		}
		if i == len(siblings) {
			break
		}
	}
	usedBase[base] = struct{}{}
	return base
}

func (da *doubleArray) arrange(records []*record, idx, depth int, usedBase map[int]struct{}) (base int, siblings []sibling, leaf *record, err error) {
	siblings, leaf, err = makeSiblings(records, depth)
	if err != nil {
		return -1, nil, nil, err
	}
	if len(siblings) < 1 {
		return -1, nil, leaf, nil
	}
	base = da.findBase(siblings, idx, usedBase)
	if base > MaxSize {
		return -1, nil, nil, fmt.Errorf("denco: too many elements of internal slice")
	}
	da.setBase(idx, base)
	return base, siblings, leaf, err
}

// node represents a node of Double-Array.
type node struct {
	data interface{}

	// Names of path parameters.
	paramNames []string
}

// makeNode returns a new node from record.
func makeNode(r *record) (*node, error) {
	dups := make(map[string]bool)
	for _, name := range r.paramNames {
		if dups[name] {
			return nil, fmt.Errorf("denco: path parameter `%v' is duplicated in the key `%v'", name, r.Key)
		}
		dups[name] = true
	}
	return &node{data: r.Value, paramNames: r.paramNames}, nil
}

// sibling represents an intermediate data of build for Double-Array.
type sibling struct {
	// An index of start of duplicated characters.
	start int

	// An index of end of duplicated characters.
	end int

	// A character of sibling.
	c byte
}

// nextIndex returns a next index of array of BASE/CHECK.
func nextIndex(base int, c byte) int {
	return base ^ int(c)
}

// makeSiblings returns slice of sibling.
func makeSiblings(records []*record, depth int) (sib []sibling, leaf *record, err error) {
	var (
		pc byte
		n  int
	)
	for i, r := range records {
		if len(r.Key) <= depth {
			leaf = r
			continue
		}
		c := r.Key[depth]
		switch {
		case pc < c:
			sib = append(sib, sibling{start: i, c: c})
		case pc == c:
			continue
		default:
			return nil, nil, fmt.Errorf("denco: BUG: routing table hasn't been sorted")
		}
		if n > 0 {
			sib[n-1].end = i
		}
		pc = c
		n++
	}
	if n == 0 {
		return nil, leaf, nil
	}
	sib[n-1].end = len(records)
	return sib, leaf, nil
}

// Record represents a record data for router construction.
type Record struct {
	// Key for router construction.
	Key string

	// Result value for Key.
	Value interface{}
}

// NewRecord returns a new Record.
func NewRecord(key string, value interface{}) Record {
	return Record{
		Key:   key,
		Value: value,
	}
}

// record represents a record that use to build the Double-Array.
type record struct {
	Record
	paramNames []string
}

// makeRecords returns the records that use to build Double-Arrays.
func makeRecords(srcs []Record) (statics, params []*record) {
	spChars := string([]byte{ParamCharacter, WildcardCharacter})
	termChar := string(TerminationCharacter)
	for _, r := range srcs {
		if strings.ContainsAny(r.Key, spChars) {
			r.Key += termChar
			params = append(params, &record{Record: r})
		} else {
			statics = append(statics, &record{Record: r})
		}
	}
	return statics, params
}

// recordSlice represents a slice of Record for sort and implements the sort.Interface.
type recordSlice []*record

// Len implements the sort.Interface.Len.
func (rs recordSlice) Len() int {
	return len(rs)
}

// Less implements the sort.Interface.Less.
func (rs recordSlice) Less(i, j int) bool {
	return rs[i].Key < rs[j].Key
}

// Swap implements the sort.Interface.Swap.
func (rs recordSlice) Swap(i, j int) {
	rs[i], rs[j] = rs[j], rs[i]
}
