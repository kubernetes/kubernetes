package ebpf

import (
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"strings"

	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/btf"
)

// CollectionOptions control loading a collection into the kernel.
//
// Maps and Programs are passed to NewMapWithOptions and NewProgramsWithOptions.
type CollectionOptions struct {
	Maps     MapOptions
	Programs ProgramOptions
}

// CollectionSpec describes a collection.
type CollectionSpec struct {
	Maps     map[string]*MapSpec
	Programs map[string]*ProgramSpec
}

// Copy returns a recursive copy of the spec.
func (cs *CollectionSpec) Copy() *CollectionSpec {
	if cs == nil {
		return nil
	}

	cpy := CollectionSpec{
		Maps:     make(map[string]*MapSpec, len(cs.Maps)),
		Programs: make(map[string]*ProgramSpec, len(cs.Programs)),
	}

	for name, spec := range cs.Maps {
		cpy.Maps[name] = spec.Copy()
	}

	for name, spec := range cs.Programs {
		cpy.Programs[name] = spec.Copy()
	}

	return &cpy
}

// RewriteMaps replaces all references to specific maps.
//
// Use this function to use pre-existing maps instead of creating new ones
// when calling NewCollection. Any named maps are removed from CollectionSpec.Maps.
//
// Returns an error if a named map isn't used in at least one program.
func (cs *CollectionSpec) RewriteMaps(maps map[string]*Map) error {
	for symbol, m := range maps {
		// have we seen a program that uses this symbol / map
		seen := false
		fd := m.FD()
		for progName, progSpec := range cs.Programs {
			err := progSpec.Instructions.RewriteMapPtr(symbol, fd)

			switch {
			case err == nil:
				seen = true

			case asm.IsUnreferencedSymbol(err):
				// Not all programs need to use the map

			default:
				return fmt.Errorf("program %s: %w", progName, err)
			}
		}

		if !seen {
			return fmt.Errorf("map %s not referenced by any programs", symbol)
		}

		// Prevent NewCollection from creating rewritten maps
		delete(cs.Maps, symbol)
	}

	return nil
}

// RewriteConstants replaces the value of multiple constants.
//
// The constant must be defined like so in the C program:
//
//    volatile const type foobar;
//    volatile const type foobar = default;
//
// Replacement values must be of the same length as the C sizeof(type).
// If necessary, they are marshalled according to the same rules as
// map values.
//
// From Linux 5.5 the verifier will use constants to eliminate dead code.
//
// Returns an error if a constant doesn't exist.
func (cs *CollectionSpec) RewriteConstants(consts map[string]interface{}) error {
	rodata := cs.Maps[".rodata"]
	if rodata == nil {
		return errors.New("missing .rodata section")
	}

	if rodata.BTF == nil {
		return errors.New(".rodata section has no BTF")
	}

	if n := len(rodata.Contents); n != 1 {
		return fmt.Errorf("expected one key in .rodata, found %d", n)
	}

	kv := rodata.Contents[0]
	value, ok := kv.Value.([]byte)
	if !ok {
		return fmt.Errorf("first value in .rodata is %T not []byte", kv.Value)
	}

	buf := make([]byte, len(value))
	copy(buf, value)

	err := patchValue(buf, btf.MapValue(rodata.BTF), consts)
	if err != nil {
		return err
	}

	rodata.Contents[0] = MapKV{kv.Key, buf}
	return nil
}

// Assign the contents of a CollectionSpec to a struct.
//
// This function is a short-cut to manually checking the presence
// of maps and programs in a collection spec. Consider using bpf2go if this
// sounds useful.
//
// The argument to must be a pointer to a struct. A field of the
// struct is updated with values from Programs or Maps if it
// has an `ebpf` tag and its type is *ProgramSpec or *MapSpec.
// The tag gives the name of the program or map as found in
// the CollectionSpec.
//
//    struct {
//        Foo     *ebpf.ProgramSpec `ebpf:"xdp_foo"`
//        Bar     *ebpf.MapSpec     `ebpf:"bar_map"`
//        Ignored int
//    }
//
// Returns an error if any of the fields can't be found, or
// if the same map or program is assigned multiple times.
func (cs *CollectionSpec) Assign(to interface{}) error {
	valueOf := func(typ reflect.Type, name string) (reflect.Value, error) {
		switch typ {
		case reflect.TypeOf((*ProgramSpec)(nil)):
			p := cs.Programs[name]
			if p == nil {
				return reflect.Value{}, fmt.Errorf("missing program %q", name)
			}
			return reflect.ValueOf(p), nil
		case reflect.TypeOf((*MapSpec)(nil)):
			m := cs.Maps[name]
			if m == nil {
				return reflect.Value{}, fmt.Errorf("missing map %q", name)
			}
			return reflect.ValueOf(m), nil
		default:
			return reflect.Value{}, fmt.Errorf("unsupported type %s", typ)
		}
	}

	return assignValues(to, valueOf)
}

// LoadAndAssign maps and programs into the kernel and assign them to a struct.
//
// This function is a short-cut to manually checking the presence
// of maps and programs in a collection spec. Consider using bpf2go if this
// sounds useful.
//
// The argument to must be a pointer to a struct. A field of the
// struct is updated with values from Programs or Maps if it
// has an `ebpf` tag and its type is *Program or *Map.
// The tag gives the name of the program or map as found in
// the CollectionSpec.
//
//    struct {
//        Foo     *ebpf.Program `ebpf:"xdp_foo"`
//        Bar     *ebpf.Map     `ebpf:"bar_map"`
//        Ignored int
//    }
//
// opts may be nil.
//
// Returns an error if any of the fields can't be found, or
// if the same map or program is assigned multiple times.
func (cs *CollectionSpec) LoadAndAssign(to interface{}, opts *CollectionOptions) error {
	if opts == nil {
		opts = &CollectionOptions{}
	}

	loadMap, loadProgram, done, cleanup := lazyLoadCollection(cs, opts)
	defer cleanup()

	valueOf := func(typ reflect.Type, name string) (reflect.Value, error) {
		switch typ {
		case reflect.TypeOf((*Program)(nil)):
			p, err := loadProgram(name)
			if err != nil {
				return reflect.Value{}, err
			}
			return reflect.ValueOf(p), nil
		case reflect.TypeOf((*Map)(nil)):
			m, err := loadMap(name)
			if err != nil {
				return reflect.Value{}, err
			}
			return reflect.ValueOf(m), nil
		default:
			return reflect.Value{}, fmt.Errorf("unsupported type %s", typ)
		}
	}

	if err := assignValues(to, valueOf); err != nil {
		return err
	}

	done()
	return nil
}

// Collection is a collection of Programs and Maps associated
// with their symbols
type Collection struct {
	Programs map[string]*Program
	Maps     map[string]*Map
}

// NewCollection creates a Collection from a specification.
func NewCollection(spec *CollectionSpec) (*Collection, error) {
	return NewCollectionWithOptions(spec, CollectionOptions{})
}

// NewCollectionWithOptions creates a Collection from a specification.
func NewCollectionWithOptions(spec *CollectionSpec, opts CollectionOptions) (*Collection, error) {
	loadMap, loadProgram, done, cleanup := lazyLoadCollection(spec, &opts)
	defer cleanup()

	for mapName := range spec.Maps {
		_, err := loadMap(mapName)
		if err != nil {
			return nil, err
		}
	}

	for progName := range spec.Programs {
		_, err := loadProgram(progName)
		if err != nil {
			return nil, err
		}
	}

	maps, progs := done()
	return &Collection{
		progs,
		maps,
	}, nil
}

type handleCache struct {
	btfHandles map[*btf.Spec]*btf.Handle
	btfSpecs   map[io.ReaderAt]*btf.Spec
}

func newHandleCache() *handleCache {
	return &handleCache{
		btfHandles: make(map[*btf.Spec]*btf.Handle),
		btfSpecs:   make(map[io.ReaderAt]*btf.Spec),
	}
}

func (hc handleCache) btfHandle(spec *btf.Spec) (*btf.Handle, error) {
	if hc.btfHandles[spec] != nil {
		return hc.btfHandles[spec], nil
	}

	handle, err := btf.NewHandle(spec)
	if err != nil {
		return nil, err
	}

	hc.btfHandles[spec] = handle
	return handle, nil
}

func (hc handleCache) btfSpec(rd io.ReaderAt) (*btf.Spec, error) {
	if hc.btfSpecs[rd] != nil {
		return hc.btfSpecs[rd], nil
	}

	spec, err := btf.LoadSpecFromReader(rd)
	if err != nil {
		return nil, err
	}

	hc.btfSpecs[rd] = spec
	return spec, nil
}

func (hc handleCache) close() {
	for _, handle := range hc.btfHandles {
		handle.Close()
	}
	hc.btfHandles = nil
	hc.btfSpecs = nil
}

func lazyLoadCollection(coll *CollectionSpec, opts *CollectionOptions) (
	loadMap func(string) (*Map, error),
	loadProgram func(string) (*Program, error),
	done func() (map[string]*Map, map[string]*Program),
	cleanup func(),
) {
	var (
		maps             = make(map[string]*Map)
		progs            = make(map[string]*Program)
		handles          = newHandleCache()
		skipMapsAndProgs = false
	)

	cleanup = func() {
		handles.close()

		if skipMapsAndProgs {
			return
		}

		for _, m := range maps {
			m.Close()
		}

		for _, p := range progs {
			p.Close()
		}
	}

	done = func() (map[string]*Map, map[string]*Program) {
		skipMapsAndProgs = true
		return maps, progs
	}

	loadMap = func(mapName string) (*Map, error) {
		if m := maps[mapName]; m != nil {
			return m, nil
		}

		mapSpec := coll.Maps[mapName]
		if mapSpec == nil {
			return nil, fmt.Errorf("missing map %s", mapName)
		}

		m, err := newMapWithOptions(mapSpec, opts.Maps, handles)
		if err != nil {
			return nil, fmt.Errorf("map %s: %w", mapName, err)
		}

		maps[mapName] = m
		return m, nil
	}

	loadProgram = func(progName string) (*Program, error) {
		if prog := progs[progName]; prog != nil {
			return prog, nil
		}

		progSpec := coll.Programs[progName]
		if progSpec == nil {
			return nil, fmt.Errorf("unknown program %s", progName)
		}

		progSpec = progSpec.Copy()

		// Rewrite any reference to a valid map.
		for i := range progSpec.Instructions {
			ins := &progSpec.Instructions[i]

			if !ins.IsLoadFromMap() || ins.Reference == "" {
				continue
			}

			if uint32(ins.Constant) != math.MaxUint32 {
				// Don't overwrite maps already rewritten, users can
				// rewrite programs in the spec themselves
				continue
			}

			m, err := loadMap(ins.Reference)
			if err != nil {
				return nil, fmt.Errorf("program %s: %w", progName, err)
			}

			fd := m.FD()
			if fd < 0 {
				return nil, fmt.Errorf("map %s: %w", ins.Reference, internal.ErrClosedFd)
			}
			if err := ins.RewriteMapPtr(m.FD()); err != nil {
				return nil, fmt.Errorf("progam %s: map %s: %w", progName, ins.Reference, err)
			}
		}

		prog, err := newProgramWithOptions(progSpec, opts.Programs, handles)
		if err != nil {
			return nil, fmt.Errorf("program %s: %w", progName, err)
		}

		progs[progName] = prog
		return prog, nil
	}

	return
}

// LoadCollection parses an object file and converts it to a collection.
func LoadCollection(file string) (*Collection, error) {
	spec, err := LoadCollectionSpec(file)
	if err != nil {
		return nil, err
	}
	return NewCollection(spec)
}

// Close frees all maps and programs associated with the collection.
//
// The collection mustn't be used afterwards.
func (coll *Collection) Close() {
	for _, prog := range coll.Programs {
		prog.Close()
	}
	for _, m := range coll.Maps {
		m.Close()
	}
}

// DetachMap removes the named map from the Collection.
//
// This means that a later call to Close() will not affect this map.
//
// Returns nil if no map of that name exists.
func (coll *Collection) DetachMap(name string) *Map {
	m := coll.Maps[name]
	delete(coll.Maps, name)
	return m
}

// DetachProgram removes the named program from the Collection.
//
// This means that a later call to Close() will not affect this program.
//
// Returns nil if no program of that name exists.
func (coll *Collection) DetachProgram(name string) *Program {
	p := coll.Programs[name]
	delete(coll.Programs, name)
	return p
}

// Assign the contents of a collection to a struct.
//
// Deprecated: use CollectionSpec.Assign instead. It provides the same
// functionality but creates only the maps and programs requested.
func (coll *Collection) Assign(to interface{}) error {
	assignedMaps := make(map[string]struct{})
	assignedPrograms := make(map[string]struct{})
	valueOf := func(typ reflect.Type, name string) (reflect.Value, error) {
		switch typ {
		case reflect.TypeOf((*Program)(nil)):
			p := coll.Programs[name]
			if p == nil {
				return reflect.Value{}, fmt.Errorf("missing program %q", name)
			}
			assignedPrograms[name] = struct{}{}
			return reflect.ValueOf(p), nil
		case reflect.TypeOf((*Map)(nil)):
			m := coll.Maps[name]
			if m == nil {
				return reflect.Value{}, fmt.Errorf("missing map %q", name)
			}
			assignedMaps[name] = struct{}{}
			return reflect.ValueOf(m), nil
		default:
			return reflect.Value{}, fmt.Errorf("unsupported type %s", typ)
		}
	}

	if err := assignValues(to, valueOf); err != nil {
		return err
	}

	for name := range assignedPrograms {
		coll.DetachProgram(name)
	}

	for name := range assignedMaps {
		coll.DetachMap(name)
	}

	return nil
}

func assignValues(to interface{}, valueOf func(reflect.Type, string) (reflect.Value, error)) error {
	type structField struct {
		reflect.StructField
		value reflect.Value
	}

	var (
		fields        []structField
		visitedTypes  = make(map[reflect.Type]bool)
		flattenStruct func(reflect.Value) error
	)

	flattenStruct = func(structVal reflect.Value) error {
		structType := structVal.Type()
		if structType.Kind() != reflect.Struct {
			return fmt.Errorf("%s is not a struct", structType)
		}

		if visitedTypes[structType] {
			return fmt.Errorf("recursion on type %s", structType)
		}

		for i := 0; i < structType.NumField(); i++ {
			field := structField{structType.Field(i), structVal.Field(i)}

			name := field.Tag.Get("ebpf")
			if name != "" {
				fields = append(fields, field)
				continue
			}

			var err error
			switch field.Type.Kind() {
			case reflect.Ptr:
				if field.Type.Elem().Kind() != reflect.Struct {
					continue
				}

				if field.value.IsNil() {
					return fmt.Errorf("nil pointer to %s", structType)
				}

				err = flattenStruct(field.value.Elem())

			case reflect.Struct:
				err = flattenStruct(field.value)

			default:
				continue
			}

			if err != nil {
				return fmt.Errorf("field %s: %w", field.Name, err)
			}
		}

		return nil
	}

	toValue := reflect.ValueOf(to)
	if toValue.Type().Kind() != reflect.Ptr {
		return fmt.Errorf("%T is not a pointer to struct", to)
	}

	if toValue.IsNil() {
		return fmt.Errorf("nil pointer to %T", to)
	}

	if err := flattenStruct(toValue.Elem()); err != nil {
		return err
	}

	type elem struct {
		// Either *Map or *Program
		typ  reflect.Type
		name string
	}

	assignedTo := make(map[elem]string)
	for _, field := range fields {
		name := field.Tag.Get("ebpf")
		if strings.Contains(name, ",") {
			return fmt.Errorf("field %s: ebpf tag contains a comma", field.Name)
		}

		e := elem{field.Type, name}
		if assignedField := assignedTo[e]; assignedField != "" {
			return fmt.Errorf("field %s: %q was already assigned to %s", field.Name, name, assignedField)
		}

		value, err := valueOf(field.Type, name)
		if err != nil {
			return fmt.Errorf("field %s: %w", field.Name, err)
		}

		if !field.value.CanSet() {
			return fmt.Errorf("field %s: can't set value", field.Name)
		}

		field.value.Set(value)
		assignedTo[e] = field.Name
	}

	return nil
}
