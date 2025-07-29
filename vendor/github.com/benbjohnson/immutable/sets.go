package immutable

// Set represents a collection of unique values. The set uses a Hasher
// to generate hashes and check for equality of key values.
//
// Internally, the Set stores values as keys of a Map[T,struct{}]
type Set[T any] struct {
	m *Map[T, struct{}]
}

// NewSet returns a new instance of Set.
//
// If hasher is nil, a default hasher implementation will automatically be chosen based on the first key added.
// Default hasher implementations only exist for int, string, and byte slice types.
// NewSet can also take some initial values as varargs.
func NewSet[T any](hasher Hasher[T], values ...T) Set[T] {
	m := NewMap[T, struct{}](hasher)
	for _, value := range values {
		m = m.set(value, struct{}{}, true)
	}
	return Set[T]{m}
}

// Add returns a set containing the new value.
//
// This function will return a new set even if the set already contains the value.
func (s Set[T]) Add(value T) Set[T] {
	return Set[T]{s.m.Set(value, struct{}{})}
}

// Delete returns a set with the given key removed.
func (s Set[T]) Delete(value T) Set[T] {
	return Set[T]{s.m.Delete(value)}
}

// Has returns true when the set contains the given value
func (s Set[T]) Has(val T) bool {
	_, ok := s.m.Get(val)
	return ok
}

// Len returns the number of elements in the underlying map.
func (s Set[K]) Len() int {
	return s.m.Len()
}

// Items returns a slice of the items inside the set
func (s Set[T]) Items() []T {
	r := make([]T, 0, s.Len())
	itr := s.Iterator()
	for !itr.Done() {
		v, _ := itr.Next()
		r = append(r, v)
	}
	return r
}

// Iterator returns a new iterator for this set positioned at the first value.
func (s Set[T]) Iterator() *SetIterator[T] {
	itr := &SetIterator[T]{mi: s.m.Iterator()}
	itr.mi.First()
	return itr
}

// SetIterator represents an iterator over a set.
// Iteration can occur in natural or reverse order based on use of Next() or Prev().
type SetIterator[T any] struct {
	mi *MapIterator[T, struct{}]
}

// Done returns true if no more values remain in the iterator.
func (itr *SetIterator[T]) Done() bool {
	return itr.mi.Done()
}

// First moves the iterator to the first value.
func (itr *SetIterator[T]) First() {
	itr.mi.First()
}

// Next moves the iterator to the next value.
func (itr *SetIterator[T]) Next() (val T, ok bool) {
	val, _, ok = itr.mi.Next()
	return
}

type SetBuilder[T any] struct {
	s Set[T]
}

func NewSetBuilder[T any](hasher Hasher[T]) *SetBuilder[T] {
	return &SetBuilder[T]{s: NewSet(hasher)}
}

func (s SetBuilder[T]) Set(val T) {
	s.s.m = s.s.m.set(val, struct{}{}, true)
}

func (s SetBuilder[T]) Delete(val T) {
	s.s.m = s.s.m.delete(val, true)
}

func (s SetBuilder[T]) Has(val T) bool {
	return s.s.Has(val)
}

func (s SetBuilder[T]) Len() int {
	return s.s.Len()
}

type SortedSet[T any] struct {
	m *SortedMap[T, struct{}]
}

// NewSortedSet returns a new instance of SortedSet.
//
// If comparer is nil then
// a default comparer is set after the first key is inserted. Default comparers
// exist for int, string, and byte slice keys.
// NewSortedSet can also take some initial values as varargs.
func NewSortedSet[T any](comparer Comparer[T], values ...T) SortedSet[T] {
	m := NewSortedMap[T, struct{}](comparer)
	for _, value := range values {
		m = m.set(value, struct{}{}, true)
	}
	return SortedSet[T]{m}
}

// Add returns a set containing the new value.
//
// This function will return a new set even if the set already contains the value.
func (s SortedSet[T]) Add(value T) SortedSet[T] {
	return SortedSet[T]{s.m.Set(value, struct{}{})}
}

// Delete returns a set with the given key removed.
func (s SortedSet[T]) Delete(value T) SortedSet[T] {
	return SortedSet[T]{s.m.Delete(value)}
}

// Has returns true when the set contains the given value
func (s SortedSet[T]) Has(val T) bool {
	_, ok := s.m.Get(val)
	return ok
}

// Len returns the number of elements in the underlying map.
func (s SortedSet[K]) Len() int {
	return s.m.Len()
}

// Items returns a slice of the items inside the set
func (s SortedSet[T]) Items() []T {
	r := make([]T, 0, s.Len())
	itr := s.Iterator()
	for !itr.Done() {
		v, _ := itr.Next()
		r = append(r, v)
	}
	return r
}

// Iterator returns a new iterator for this set positioned at the first value.
func (s SortedSet[T]) Iterator() *SortedSetIterator[T] {
	itr := &SortedSetIterator[T]{mi: s.m.Iterator()}
	itr.mi.First()
	return itr
}

// SortedSetIterator represents an iterator over a sorted set.
// Iteration can occur in natural or reverse order based on use of Next() or Prev().
type SortedSetIterator[T any] struct {
	mi *SortedMapIterator[T, struct{}]
}

// Done returns true if no more values remain in the iterator.
func (itr *SortedSetIterator[T]) Done() bool {
	return itr.mi.Done()
}

// First moves the iterator to the first value.
func (itr *SortedSetIterator[T]) First() {
	itr.mi.First()
}

// Last moves the iterator to the last value.
func (itr *SortedSetIterator[T]) Last() {
	itr.mi.Last()
}

// Next moves the iterator to the next value.
func (itr *SortedSetIterator[T]) Next() (val T, ok bool) {
	val, _, ok = itr.mi.Next()
	return
}

// Next moves the iterator to the previous value.
func (itr *SortedSetIterator[T]) Prev() (val T, ok bool) {
	val, _, ok = itr.mi.Prev()
	return
}

// Next moves the iterator to the given value.
//
// If the value does not exist then the next value is used. If no more keys exist
// then the iterator is marked as done.
func (itr *SortedSetIterator[T]) Seek(val T) {
	itr.mi.Seek(val)
}

type SortedSetBuilder[T any] struct {
	s SortedSet[T]
}

func NewSortedSetBuilder[T any](comparer Comparer[T]) *SortedSetBuilder[T] {
	return &SortedSetBuilder[T]{s: NewSortedSet(comparer)}
}

func (s SortedSetBuilder[T]) Set(val T) {
	s.s.m = s.s.m.set(val, struct{}{}, true)
}

func (s SortedSetBuilder[T]) Delete(val T) {
	s.s.m = s.s.m.delete(val, true)
}

func (s SortedSetBuilder[T]) Has(val T) bool {
	return s.s.Has(val)
}

func (s SortedSetBuilder[T]) Len() int {
	return s.s.Len()
}
