package ebpf

import (
	"encoding/binary"
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

	// ByteOrder specifies whether the ELF was compiled for
	// big-endian or little-endian architectures.
	ByteOrder binary.ByteOrder
}

// Copy returns a recursive copy of the spec.
func (cs *CollectionSpec) Copy() *CollectionSpec {
	if cs == nil {
		return nil
	}

	cpy := CollectionSpec{
		Maps:      make(map[string]*MapSpec, len(cs.Maps)),
		Programs:  make(map[string]*ProgramSpec, len(cs.Programs)),
		ByteOrder: cs.ByteOrder,
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

	err := patchValue(buf, rodata.BTF.Value, consts)
	if err != nil {
		return err
	}

	rodata.Contents[0] = MapKV{kv.Key, buf}
	return nil
}

// Assign the contents of a CollectionSpec to a struct.
//
// This function is a shortcut to manually checking the presence
// of maps and programs in a CollectionSpec. Consider using bpf2go
// if this sounds useful.
//
// 'to' must be a pointer to a struct. A field of the
// struct is updated with values from Programs or Maps if it
// has an `ebpf` tag and its type is *ProgramSpec or *MapSpec.
// The tag's value specifies the name of the program or map as
// found in the CollectionSpec.
//
//    struct {
//        Foo     *ebpf.ProgramSpec `ebpf:"xdp_foo"`
//        Bar     *ebpf.MapSpec     `ebpf:"bar_map"`
//        Ignored int
//    }
//
// Returns an error if any of the eBPF objects can't be found, or
// if the same MapSpec or ProgramSpec is assigned multiple times.
func (cs *CollectionSpec) Assign(to interface{}) error {
	// Assign() only supports assigning ProgramSpecs and MapSpecs,
	// so doesn't load any resources into the kernel.
	getValue := func(typ reflect.Type, name string) (interface{}, error) {
		switch typ {

		case reflect.TypeOf((*ProgramSpec)(nil)):
			if p := cs.Programs[name]; p != nil {
				return p, nil
			}
			return nil, fmt.Errorf("missing program %q", name)

		case reflect.TypeOf((*MapSpec)(nil)):
			if m := cs.Maps[name]; m != nil {
				return m, nil
			}
			return nil, fmt.Errorf("missing map %q", name)

		default:
			return nil, fmt.Errorf("unsupported type %s", typ)
		}
	}

	return assignValues(to, getValue)
}

// LoadAndAssign loads Maps and Programs into the kernel and assigns them
// to a struct.
//
// This function is a shortcut to manually checking the presence
// of maps and programs in a CollectionSpec. Consider using bpf2go
// if this sounds useful.
//
// 'to' must be a pointer to a struct. A field of the struct is updated with
// a Program or Map if it has an `ebpf` tag and its type is *Program or *Map.
// The tag's value specifies the name of the program or map as found in the
// CollectionSpec. Before updating the struct, the requested objects and their
// dependent resources are loaded into the kernel and populated with values if
// specified.
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
// if the same Map or Program is assigned multiple times.
func (cs *CollectionSpec) LoadAndAssign(to interface{}, opts *CollectionOptions) error {
	loader := newCollectionLoader(cs, opts)
	defer loader.cleanup()

	// Support assigning Programs and Maps, lazy-loading the required objects.
	assignedMaps := make(map[string]bool)
	getValue := func(typ reflect.Type, name string) (interface{}, error) {
		switch typ {

		case reflect.TypeOf((*Program)(nil)):
			return loader.loadProgram(name)

		case reflect.TypeOf((*Map)(nil)):
			assignedMaps[name] = true
			return loader.loadMap(name)

		default:
			return nil, fmt.Errorf("unsupported type %s", typ)
		}
	}

	// Load the Maps and Programs requested by the annotated struct.
	if err := assignValues(to, getValue); err != nil {
		return err
	}

	// Populate the requested maps. Has a chance of lazy-loading other dependent maps.
	if err := loader.populateMaps(); err != nil {
		return err
	}

	// Evaluate the loader's objects after all (lazy)loading has taken place.
	for n, m := range loader.maps {
		switch m.typ {
		case ProgramArray:
			// Require all lazy-loaded ProgramArrays to be assigned to the given object.
			// Without any references, they will be closed on the first GC and all tail
			// calls into them will miss.
			if !assignedMaps[n] {
				return fmt.Errorf("ProgramArray %s must be assigned to prevent missed tail calls", n)
			}
		}
	}

	loader.finalize()

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
	loader := newCollectionLoader(spec, &opts)
	defer loader.cleanup()

	// Create maps first, as their fds need to be linked into programs.
	for mapName := range spec.Maps {
		if _, err := loader.loadMap(mapName); err != nil {
			return nil, err
		}
	}

	for progName := range spec.Programs {
		if _, err := loader.loadProgram(progName); err != nil {
			return nil, err
		}
	}

	// Maps can contain Program and Map stubs, so populate them after
	// all Maps and Programs have been successfully loaded.
	if err := loader.populateMaps(); err != nil {
		return nil, err
	}

	maps, progs := loader.maps, loader.programs

	loader.finalize()

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
}

type collectionLoader struct {
	coll     *CollectionSpec
	opts     *CollectionOptions
	maps     map[string]*Map
	programs map[string]*Program
	handles  *handleCache
}

func newCollectionLoader(coll *CollectionSpec, opts *CollectionOptions) *collectionLoader {
	if opts == nil {
		opts = &CollectionOptions{}
	}

	return &collectionLoader{
		coll,
		opts,
		make(map[string]*Map),
		make(map[string]*Program),
		newHandleCache(),
	}
}

// finalize should be called when all the collectionLoader's resources
// have been successfully loaded into the kernel and populated with values.
func (cl *collectionLoader) finalize() {
	cl.maps, cl.programs = nil, nil
}

// cleanup cleans up all resources left over in the collectionLoader.
// Call finalize() when Map and Program creation/population is successful
// to prevent them from getting closed.
func (cl *collectionLoader) cleanup() {
	cl.handles.close()
	for _, m := range cl.maps {
		m.Close()
	}
	for _, p := range cl.programs {
		p.Close()
	}
}

func (cl *collectionLoader) loadMap(mapName string) (*Map, error) {
	if m := cl.maps[mapName]; m != nil {
		return m, nil
	}

	mapSpec := cl.coll.Maps[mapName]
	if mapSpec == nil {
		return nil, fmt.Errorf("missing map %s", mapName)
	}

	m, err := newMapWithOptions(mapSpec, cl.opts.Maps, cl.handles)
	if err != nil {
		return nil, fmt.Errorf("map %s: %w", mapName, err)
	}

	cl.maps[mapName] = m
	return m, nil
}

func (cl *collectionLoader) loadProgram(progName string) (*Program, error) {
	if prog := cl.programs[progName]; prog != nil {
		return prog, nil
	}

	progSpec := cl.coll.Programs[progName]
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

		m, err := cl.loadMap(ins.Reference)
		if err != nil {
			return nil, fmt.Errorf("program %s: %w", progName, err)
		}

		fd := m.FD()
		if fd < 0 {
			return nil, fmt.Errorf("map %s: %w", ins.Reference, internal.ErrClosedFd)
		}
		if err := ins.RewriteMapPtr(m.FD()); err != nil {
			return nil, fmt.Errorf("program %s: map %s: %w", progName, ins.Reference, err)
		}
	}

	prog, err := newProgramWithOptions(progSpec, cl.opts.Programs, cl.handles)
	if err != nil {
		return nil, fmt.Errorf("program %s: %w", progName, err)
	}

	cl.programs[progName] = prog
	return prog, nil
}

func (cl *collectionLoader) populateMaps() error {
	for mapName, m := range cl.maps {
		mapSpec, ok := cl.coll.Maps[mapName]
		if !ok {
			return fmt.Errorf("missing map spec %s", mapName)
		}

		mapSpec = mapSpec.Copy()

		// Replace any object stubs with loaded objects.
		for i, kv := range mapSpec.Contents {
			switch v := kv.Value.(type) {
			case programStub:
				// loadProgram is idempotent and could return an existing Program.
				prog, err := cl.loadProgram(string(v))
				if err != nil {
					return fmt.Errorf("loading program %s, for map %s: %w", v, mapName, err)
				}
				mapSpec.Contents[i] = MapKV{kv.Key, prog}

			case mapStub:
				// loadMap is idempotent and could return an existing Map.
				innerMap, err := cl.loadMap(string(v))
				if err != nil {
					return fmt.Errorf("loading inner map %s, for map %s: %w", v, mapName, err)
				}
				mapSpec.Contents[i] = MapKV{kv.Key, innerMap}
			}
		}

		// Populate and freeze the map if specified.
		if err := m.finalize(mapSpec); err != nil {
			return fmt.Errorf("populating map %s: %w", mapName, err)
		}
	}

	return nil
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

// structField represents a struct field containing the ebpf struct tag.
type structField struct {
	reflect.StructField
	value reflect.Value
}

// ebpfFields extracts field names tagged with 'ebpf' from a struct type.
// Keep track of visited types to avoid infinite recursion.
func ebpfFields(structVal reflect.Value, visited map[reflect.Type]bool) ([]structField, error) {
	if visited == nil {
		visited = make(map[reflect.Type]bool)
	}

	structType := structVal.Type()
	if structType.Kind() != reflect.Struct {
		return nil, fmt.Errorf("%s is not a struct", structType)
	}

	if visited[structType] {
		return nil, fmt.Errorf("recursion on type %s", structType)
	}

	fields := make([]structField, 0, structType.NumField())
	for i := 0; i < structType.NumField(); i++ {
		field := structField{structType.Field(i), structVal.Field(i)}

		// If the field is tagged, gather it and move on.
		name := field.Tag.Get("ebpf")
		if name != "" {
			fields = append(fields, field)
			continue
		}

		// If the field does not have an ebpf tag, but is a struct or a pointer
		// to a struct, attempt to gather its fields as well.
		var v reflect.Value
		switch field.Type.Kind() {
		case reflect.Ptr:
			if field.Type.Elem().Kind() != reflect.Struct {
				continue
			}

			if field.value.IsNil() {
				return nil, fmt.Errorf("nil pointer to %s", structType)
			}

			// Obtain the destination type of the pointer.
			v = field.value.Elem()

		case reflect.Struct:
			// Reference the value's type directly.
			v = field.value

		default:
			continue
		}

		inner, err := ebpfFields(v, visited)
		if err != nil {
			return nil, fmt.Errorf("field %s: %w", field.Name, err)
		}

		fields = append(fields, inner...)
	}

	return fields, nil
}

// assignValues attempts to populate all fields of 'to' tagged with 'ebpf'.
//
// getValue is called for every tagged field of 'to' and must return the value
// to be assigned to the field with the given typ and name.
func assignValues(to interface{},
	getValue func(typ reflect.Type, name string) (interface{}, error)) error {

	toValue := reflect.ValueOf(to)
	if toValue.Type().Kind() != reflect.Ptr {
		return fmt.Errorf("%T is not a pointer to struct", to)
	}

	if toValue.IsNil() {
		return fmt.Errorf("nil pointer to %T", to)
	}

	fields, err := ebpfFields(toValue.Elem(), nil)
	if err != nil {
		return err
	}

	type elem struct {
		// Either *Map or *Program
		typ  reflect.Type
		name string
	}

	assigned := make(map[elem]string)
	for _, field := range fields {
		// Get string value the field is tagged with.
		tag := field.Tag.Get("ebpf")
		if strings.Contains(tag, ",") {
			return fmt.Errorf("field %s: ebpf tag contains a comma", field.Name)
		}

		// Check if the eBPF object with the requested
		// type and tag was already assigned elsewhere.
		e := elem{field.Type, tag}
		if af := assigned[e]; af != "" {
			return fmt.Errorf("field %s: object %q was already assigned to %s", field.Name, tag, af)
		}

		// Get the eBPF object referred to by the tag.
		value, err := getValue(field.Type, tag)
		if err != nil {
			return fmt.Errorf("field %s: %w", field.Name, err)
		}

		if !field.value.CanSet() {
			return fmt.Errorf("field %s: can't set value", field.Name)
		}
		field.value.Set(reflect.ValueOf(value))

		assigned[e] = field.Name
	}

	return nil
}
