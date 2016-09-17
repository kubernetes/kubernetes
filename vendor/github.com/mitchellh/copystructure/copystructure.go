package copystructure

import (
	"reflect"
	"sync"

	"github.com/mitchellh/reflectwalk"
)

// Copy returns a deep copy of v.
func Copy(v interface{}) (interface{}, error) {
	return Config{}.Copy(v)
}

// CopierFunc is a function that knows how to deep copy a specific type.
// Register these globally with the Copiers variable.
type CopierFunc func(interface{}) (interface{}, error)

// Copiers is a map of types that behave specially when they are copied.
// If a type is found in this map while deep copying, this function
// will be called to copy it instead of attempting to copy all fields.
//
// The key should be the type, obtained using: reflect.TypeOf(value with type).
//
// It is unsafe to write to this map after Copies have started. If you
// are writing to this map while also copying, wrap all modifications to
// this map as well as to Copy in a mutex.
var Copiers map[reflect.Type]CopierFunc = make(map[reflect.Type]CopierFunc)

type Config struct {
	// Lock any types that are a sync.Locker and are not a mutex while copying.
	// If there is an RLocker method, use that to get the sync.Locker.
	Lock bool

	// Copiers is a map of types associated with a CopierFunc. Use the global
	// Copiers map if this is nil.
	Copiers map[reflect.Type]CopierFunc
}

func (c Config) Copy(v interface{}) (interface{}, error) {
	w := new(walker)
	if c.Lock {
		w.useLocks = true
	}

	if c.Copiers == nil {
		c.Copiers = Copiers
	}

	err := reflectwalk.Walk(v, w)
	if err != nil {
		return nil, err
	}

	// Get the result. If the result is nil, then we want to turn it
	// into a typed nil if we can.
	result := w.Result
	if result == nil {
		val := reflect.ValueOf(v)
		result = reflect.Indirect(reflect.New(val.Type())).Interface()
	}

	return result, nil
}

type walker struct {
	Result interface{}

	depth       int
	ignoreDepth int
	vals        []reflect.Value
	cs          []reflect.Value
	ps          []bool

	// any locks we've taken, indexed by depth
	locks []sync.Locker
	// take locks while walking the structure
	useLocks bool
}

func (w *walker) Enter(l reflectwalk.Location) error {
	w.depth++

	// ensure we have enough elements to index via w.depth
	for w.depth >= len(w.locks) {
		w.locks = append(w.locks, nil)
	}

	return nil
}

func (w *walker) Exit(l reflectwalk.Location) error {
	locker := w.locks[w.depth]
	w.locks[w.depth] = nil
	if locker != nil {
		defer locker.Unlock()
	}

	w.depth--
	if w.ignoreDepth > w.depth {
		w.ignoreDepth = 0
	}

	if w.ignoring() {
		return nil
	}

	switch l {
	case reflectwalk.Map:
		fallthrough
	case reflectwalk.Slice:
		// Pop map off our container
		w.cs = w.cs[:len(w.cs)-1]
	case reflectwalk.MapValue:
		// Pop off the key and value
		mv := w.valPop()
		mk := w.valPop()
		m := w.cs[len(w.cs)-1]

		// If mv is the zero value, SetMapIndex deletes the key form the map,
		// or in this case never adds it. We need to create a properly typed
		// zero value so that this key can be set.
		if !mv.IsValid() {
			mv = reflect.Zero(m.Type().Elem())
		}
		m.SetMapIndex(mk, mv)
	case reflectwalk.SliceElem:
		// Pop off the value and the index and set it on the slice
		v := w.valPop()
		if v.IsValid() {
			i := w.valPop().Interface().(int)
			s := w.cs[len(w.cs)-1]
			se := s.Index(i)
			if se.CanSet() {
				se.Set(v)
			}
		}
	case reflectwalk.Struct:
		w.replacePointerMaybe()

		// Remove the struct from the container stack
		w.cs = w.cs[:len(w.cs)-1]
	case reflectwalk.StructField:
		// Pop off the value and the field
		v := w.valPop()
		f := w.valPop().Interface().(reflect.StructField)
		if v.IsValid() {
			s := w.cs[len(w.cs)-1]
			sf := reflect.Indirect(s).FieldByName(f.Name)
			if sf.CanSet() {
				sf.Set(v)
			}
		}
	case reflectwalk.WalkLoc:
		// Clear out the slices for GC
		w.cs = nil
		w.vals = nil
	}

	return nil
}

func (w *walker) Map(m reflect.Value) error {
	if w.ignoring() {
		return nil
	}
	w.lock(m)

	// Create the map. If the map itself is nil, then just make a nil map
	var newMap reflect.Value
	if m.IsNil() {
		newMap = reflect.Indirect(reflect.New(m.Type()))
	} else {
		newMap = reflect.MakeMap(m.Type())
	}

	w.cs = append(w.cs, newMap)
	w.valPush(newMap)
	return nil
}

func (w *walker) MapElem(m, k, v reflect.Value) error {
	return nil
}

func (w *walker) PointerEnter(v bool) error {
	if w.ignoring() {
		return nil
	}

	w.ps = append(w.ps, v)
	return nil
}

func (w *walker) PointerExit(bool) error {
	if w.ignoring() {
		return nil
	}

	w.ps = w.ps[:len(w.ps)-1]
	return nil
}

func (w *walker) Primitive(v reflect.Value) error {
	if w.ignoring() {
		return nil
	}
	w.lock(v)

	// IsValid verifies the v is non-zero and CanInterface verifies
	// that we're allowed to read this value (unexported fields).
	var newV reflect.Value
	if v.IsValid() && v.CanInterface() {
		newV = reflect.New(v.Type())
		reflect.Indirect(newV).Set(v)
	}

	w.valPush(newV)
	w.replacePointerMaybe()
	return nil
}

func (w *walker) Slice(s reflect.Value) error {
	if w.ignoring() {
		return nil
	}
	w.lock(s)

	var newS reflect.Value
	if s.IsNil() {
		newS = reflect.Indirect(reflect.New(s.Type()))
	} else {
		newS = reflect.MakeSlice(s.Type(), s.Len(), s.Cap())
	}

	w.cs = append(w.cs, newS)
	w.valPush(newS)
	return nil
}

func (w *walker) SliceElem(i int, elem reflect.Value) error {
	if w.ignoring() {
		return nil
	}

	// We don't write the slice here because elem might still be
	// arbitrarily complex. Just record the index and continue on.
	w.valPush(reflect.ValueOf(i))

	return nil
}

func (w *walker) Struct(s reflect.Value) error {
	if w.ignoring() {
		return nil
	}
	w.lock(s)

	var v reflect.Value
	if c, ok := Copiers[s.Type()]; ok {
		// We have a Copier for this struct, so we use that copier to
		// get the copy, and we ignore anything deeper than this.
		w.ignoreDepth = w.depth

		dup, err := c(s.Interface())
		if err != nil {
			return err
		}

		v = reflect.ValueOf(dup)
	} else {
		// No copier, we copy ourselves and allow reflectwalk to guide
		// us deeper into the structure for copying.
		v = reflect.New(s.Type())
	}

	// Push the value onto the value stack for setting the struct field,
	// and add the struct itself to the containers stack in case we walk
	// deeper so that its own fields can be modified.
	w.valPush(v)
	w.cs = append(w.cs, v)

	return nil
}

func (w *walker) StructField(f reflect.StructField, v reflect.Value) error {
	if w.ignoring() {
		return nil
	}

	// Push the field onto the stack, we'll handle it when we exit
	// the struct field in Exit...
	w.valPush(reflect.ValueOf(f))
	return nil
}

func (w *walker) ignoring() bool {
	return w.ignoreDepth > 0 && w.depth >= w.ignoreDepth
}

func (w *walker) pointerPeek() bool {
	return w.ps[len(w.ps)-1]
}

func (w *walker) valPop() reflect.Value {
	result := w.vals[len(w.vals)-1]
	w.vals = w.vals[:len(w.vals)-1]

	// If we're out of values, that means we popped everything off. In
	// this case, we reset the result so the next pushed value becomes
	// the result.
	if len(w.vals) == 0 {
		w.Result = nil
	}

	return result
}

func (w *walker) valPush(v reflect.Value) {
	w.vals = append(w.vals, v)

	// If we haven't set the result yet, then this is the result since
	// it is the first (outermost) value we're seeing.
	if w.Result == nil && v.IsValid() {
		w.Result = v.Interface()
	}
}

func (w *walker) replacePointerMaybe() {
	// Determine the last pointer value. If it is NOT a pointer, then
	// we need to push that onto the stack.
	if !w.pointerPeek() {
		w.valPush(reflect.Indirect(w.valPop()))
	}
}

// if this value is a Locker, lock it and add it to the locks slice
func (w *walker) lock(v reflect.Value) {
	if !w.useLocks {
		return
	}

	if !v.IsValid() || !v.CanInterface() {
		return
	}

	type rlocker interface {
		RLocker() sync.Locker
	}

	var locker sync.Locker

	// first check if we can get a locker from the value
	switch l := v.Interface().(type) {
	case rlocker:
		// don't lock a mutex directly
		if _, ok := l.(*sync.RWMutex); !ok {
			locker = l.RLocker()
		}
	case sync.Locker:
		locker = l
	}

	// the value itself isn't a locker, so check the method on a pointer too
	if locker == nil && v.CanAddr() {
		switch l := v.Addr().Interface().(type) {
		case rlocker:
			// don't lock a mutex directly
			if _, ok := l.(*sync.RWMutex); !ok {
				locker = l.RLocker()
			}
		case sync.Locker:
			locker = l
		}
	}

	// still no callable locker
	if locker == nil {
		return
	}

	// don't lock a mutex directly
	switch locker.(type) {
	case *sync.Mutex, *sync.RWMutex:
		return
	}

	locker.Lock()
	w.locks[w.depth] = locker
}
