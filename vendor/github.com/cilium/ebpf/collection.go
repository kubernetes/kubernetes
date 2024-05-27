package ebpf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/btf"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/kconfig"
	"github.com/cilium/ebpf/internal/sysenc"
)

// CollectionOptions control loading a collection into the kernel.
//
// Maps and Programs are passed to NewMapWithOptions and NewProgramsWithOptions.
type CollectionOptions struct {
	Maps     MapOptions
	Programs ProgramOptions

	// MapReplacements takes a set of Maps that will be used instead of
	// creating new ones when loading the CollectionSpec.
	//
	// For each given Map, there must be a corresponding MapSpec in
	// CollectionSpec.Maps, and its type, key/value size, max entries and flags
	// must match the values of the MapSpec.
	//
	// The given Maps are Clone()d before being used in the Collection, so the
	// caller can Close() them freely when they are no longer needed.
	MapReplacements map[string]*Map
}

// CollectionSpec describes a collection.
type CollectionSpec struct {
	Maps     map[string]*MapSpec
	Programs map[string]*ProgramSpec

	// Types holds type information about Maps and Programs.
	// Modifications to Types are currently undefined behaviour.
	Types *btf.Spec

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
		Types:     cs.Types,
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
//
// Deprecated: Pass CollectionOptions.MapReplacements when loading the Collection
// instead.
func (cs *CollectionSpec) RewriteMaps(maps map[string]*Map) error {
	for symbol, m := range maps {
		// have we seen a program that uses this symbol / map
		seen := false
		for progName, progSpec := range cs.Programs {
			err := progSpec.Instructions.AssociateMap(symbol, m)

			switch {
			case err == nil:
				seen = true

			case errors.Is(err, asm.ErrUnreferencedSymbol):
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

// MissingConstantsError is returned by [CollectionSpec.RewriteConstants].
type MissingConstantsError struct {
	// The constants missing from .rodata.
	Constants []string
}

func (m *MissingConstantsError) Error() string {
	return fmt.Sprintf("some constants are missing from .rodata: %s", strings.Join(m.Constants, ", "))
}

// RewriteConstants replaces the value of multiple constants.
//
// The constant must be defined like so in the C program:
//
//	volatile const type foobar;
//	volatile const type foobar = default;
//
// Replacement values must be of the same length as the C sizeof(type).
// If necessary, they are marshalled according to the same rules as
// map values.
//
// From Linux 5.5 the verifier will use constants to eliminate dead code.
//
// Returns an error wrapping [MissingConstantsError] if a constant doesn't exist.
func (cs *CollectionSpec) RewriteConstants(consts map[string]interface{}) error {
	replaced := make(map[string]bool)

	for name, spec := range cs.Maps {
		if !strings.HasPrefix(name, ".rodata") {
			continue
		}

		b, ds, err := spec.dataSection()
		if errors.Is(err, errMapNoBTFValue) {
			// Data sections without a BTF Datasec are valid, but don't support
			// constant replacements.
			continue
		}
		if err != nil {
			return fmt.Errorf("map %s: %w", name, err)
		}

		// MapSpec.Copy() performs a shallow copy. Fully copy the byte slice
		// to avoid any changes affecting other copies of the MapSpec.
		cpy := make([]byte, len(b))
		copy(cpy, b)

		for _, v := range ds.Vars {
			vname := v.Type.TypeName()
			replacement, ok := consts[vname]
			if !ok {
				continue
			}

			if _, ok := v.Type.(*btf.Var); !ok {
				return fmt.Errorf("section %s: unexpected type %T for variable %s", name, v.Type, vname)
			}

			if replaced[vname] {
				return fmt.Errorf("section %s: duplicate variable %s", name, vname)
			}

			if int(v.Offset+v.Size) > len(cpy) {
				return fmt.Errorf("section %s: offset %d(+%d) for variable %s is out of bounds", name, v.Offset, v.Size, vname)
			}

			b, err := sysenc.Marshal(replacement, int(v.Size))
			if err != nil {
				return fmt.Errorf("marshaling constant replacement %s: %w", vname, err)
			}

			b.CopyTo(cpy[v.Offset : v.Offset+v.Size])

			replaced[vname] = true
		}

		spec.Contents[0] = MapKV{Key: uint32(0), Value: cpy}
	}

	var missing []string
	for c := range consts {
		if !replaced[c] {
			missing = append(missing, c)
		}
	}

	if len(missing) != 0 {
		return fmt.Errorf("rewrite constants: %w", &MissingConstantsError{Constants: missing})
	}

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
//	struct {
//	    Foo     *ebpf.ProgramSpec `ebpf:"xdp_foo"`
//	    Bar     *ebpf.MapSpec     `ebpf:"bar_map"`
//	    Ignored int
//	}
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
// Omitting Map/Program.Close() during application shutdown is an error.
// See the package documentation for details around Map and Program lifecycle.
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
//	struct {
//	    Foo     *ebpf.Program `ebpf:"xdp_foo"`
//	    Bar     *ebpf.Map     `ebpf:"bar_map"`
//	    Ignored int
//	}
//
// opts may be nil.
//
// Returns an error if any of the fields can't be found, or
// if the same Map or Program is assigned multiple times.
func (cs *CollectionSpec) LoadAndAssign(to interface{}, opts *CollectionOptions) error {
	loader, err := newCollectionLoader(cs, opts)
	if err != nil {
		return err
	}
	defer loader.close()

	// Support assigning Programs and Maps, lazy-loading the required objects.
	assignedMaps := make(map[string]bool)
	assignedProgs := make(map[string]bool)

	getValue := func(typ reflect.Type, name string) (interface{}, error) {
		switch typ {

		case reflect.TypeOf((*Program)(nil)):
			assignedProgs[name] = true
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
	if err := loader.populateDeferredMaps(); err != nil {
		return err
	}

	// Evaluate the loader's objects after all (lazy)loading has taken place.
	for n, m := range loader.maps {
		switch m.typ {
		case ProgramArray:
			// Require all lazy-loaded ProgramArrays to be assigned to the given object.
			// The kernel empties a ProgramArray once the last user space reference
			// to it closes, which leads to failed tail calls. Combined with the library
			// closing map fds via GC finalizers this can lead to surprising behaviour.
			// Only allow unassigned ProgramArrays when the library hasn't pre-populated
			// any entries from static value declarations. At this point, we know the map
			// is empty and there's no way for the caller to interact with the map going
			// forward.
			if !assignedMaps[n] && len(cs.Maps[n].Contents) > 0 {
				return fmt.Errorf("ProgramArray %s must be assigned to prevent missed tail calls", n)
			}
		}
	}

	// Prevent loader.cleanup() from closing assigned Maps and Programs.
	for m := range assignedMaps {
		delete(loader.maps, m)
	}
	for p := range assignedProgs {
		delete(loader.programs, p)
	}

	return nil
}

// Collection is a collection of Programs and Maps associated
// with their symbols
type Collection struct {
	Programs map[string]*Program
	Maps     map[string]*Map
}

// NewCollection creates a Collection from the given spec, creating and
// loading its declared resources into the kernel.
//
// Omitting Collection.Close() during application shutdown is an error.
// See the package documentation for details around Map and Program lifecycle.
func NewCollection(spec *CollectionSpec) (*Collection, error) {
	return NewCollectionWithOptions(spec, CollectionOptions{})
}

// NewCollectionWithOptions creates a Collection from the given spec using
// options, creating and loading its declared resources into the kernel.
//
// Omitting Collection.Close() during application shutdown is an error.
// See the package documentation for details around Map and Program lifecycle.
func NewCollectionWithOptions(spec *CollectionSpec, opts CollectionOptions) (*Collection, error) {
	loader, err := newCollectionLoader(spec, &opts)
	if err != nil {
		return nil, err
	}
	defer loader.close()

	// Create maps first, as their fds need to be linked into programs.
	for mapName := range spec.Maps {
		if _, err := loader.loadMap(mapName); err != nil {
			return nil, err
		}
	}

	for progName, prog := range spec.Programs {
		if prog.Type == UnspecifiedProgram {
			continue
		}

		if _, err := loader.loadProgram(progName); err != nil {
			return nil, err
		}
	}

	// Maps can contain Program and Map stubs, so populate them after
	// all Maps and Programs have been successfully loaded.
	if err := loader.populateDeferredMaps(); err != nil {
		return nil, err
	}

	// Prevent loader.cleanup from closing maps and programs.
	maps, progs := loader.maps, loader.programs
	loader.maps, loader.programs = nil, nil

	return &Collection{
		progs,
		maps,
	}, nil
}

type collectionLoader struct {
	coll     *CollectionSpec
	opts     *CollectionOptions
	maps     map[string]*Map
	programs map[string]*Program
}

func newCollectionLoader(coll *CollectionSpec, opts *CollectionOptions) (*collectionLoader, error) {
	if opts == nil {
		opts = &CollectionOptions{}
	}

	// Check for existing MapSpecs in the CollectionSpec for all provided replacement maps.
	for name, m := range opts.MapReplacements {
		spec, ok := coll.Maps[name]
		if !ok {
			return nil, fmt.Errorf("replacement map %s not found in CollectionSpec", name)
		}

		if err := spec.Compatible(m); err != nil {
			return nil, fmt.Errorf("using replacement map %s: %w", spec.Name, err)
		}
	}

	return &collectionLoader{
		coll,
		opts,
		make(map[string]*Map),
		make(map[string]*Program),
	}, nil
}

// close all resources left over in the collectionLoader.
func (cl *collectionLoader) close() {
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

	if replaceMap, ok := cl.opts.MapReplacements[mapName]; ok {
		// Clone the map to avoid closing user's map later on.
		m, err := replaceMap.Clone()
		if err != nil {
			return nil, err
		}

		cl.maps[mapName] = m
		return m, nil
	}

	m, err := newMapWithOptions(mapSpec, cl.opts.Maps)
	if err != nil {
		return nil, fmt.Errorf("map %s: %w", mapName, err)
	}

	// Finalize 'scalar' maps that don't refer to any other eBPF resources
	// potentially pending creation. This is needed for frozen maps like .rodata
	// that need to be finalized before invoking the verifier.
	if !mapSpec.Type.canStoreMapOrProgram() {
		if err := m.finalize(mapSpec); err != nil {
			return nil, fmt.Errorf("finalizing map %s: %w", mapName, err)
		}
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

	// Bail out early if we know the kernel is going to reject the program.
	// This skips loading map dependencies, saving some cleanup work later.
	if progSpec.Type == UnspecifiedProgram {
		return nil, fmt.Errorf("cannot load program %s: program type is unspecified", progName)
	}

	progSpec = progSpec.Copy()

	// Rewrite any reference to a valid map in the program's instructions,
	// which includes all of its dependencies.
	for i := range progSpec.Instructions {
		ins := &progSpec.Instructions[i]

		if !ins.IsLoadFromMap() || ins.Reference() == "" {
			continue
		}

		// Don't overwrite map loads containing non-zero map fd's,
		// they can be manually included by the caller.
		// Map FDs/IDs are placed in the lower 32 bits of Constant.
		if int32(ins.Constant) > 0 {
			continue
		}

		m, err := cl.loadMap(ins.Reference())
		if err != nil {
			return nil, fmt.Errorf("program %s: %w", progName, err)
		}

		if err := ins.AssociateMap(m); err != nil {
			return nil, fmt.Errorf("program %s: map %s: %w", progName, ins.Reference(), err)
		}
	}

	prog, err := newProgramWithOptions(progSpec, cl.opts.Programs)
	if err != nil {
		return nil, fmt.Errorf("program %s: %w", progName, err)
	}

	cl.programs[progName] = prog
	return prog, nil
}

// populateDeferredMaps iterates maps holding programs or other maps and loads
// any dependencies. Populates all maps in cl and freezes them if specified.
func (cl *collectionLoader) populateDeferredMaps() error {
	for mapName, m := range cl.maps {
		mapSpec, ok := cl.coll.Maps[mapName]
		if !ok {
			return fmt.Errorf("missing map spec %s", mapName)
		}

		// Scalar maps without Map or Program references are finalized during
		// creation. Don't finalize them again.
		if !mapSpec.Type.canStoreMapOrProgram() {
			continue
		}

		mapSpec = mapSpec.Copy()

		// MapSpecs that refer to inner maps or programs within the same
		// CollectionSpec do so using strings. These strings are used as the key
		// to look up the respective object in the Maps or Programs fields.
		// Resolve those references to actual Map or Program resources that
		// have been loaded into the kernel.
		for i, kv := range mapSpec.Contents {
			objName, ok := kv.Value.(string)
			if !ok {
				continue
			}

			switch t := mapSpec.Type; {
			case t.canStoreProgram():
				// loadProgram is idempotent and could return an existing Program.
				prog, err := cl.loadProgram(objName)
				if err != nil {
					return fmt.Errorf("loading program %s, for map %s: %w", objName, mapName, err)
				}
				mapSpec.Contents[i] = MapKV{kv.Key, prog}

			case t.canStoreMap():
				// loadMap is idempotent and could return an existing Map.
				innerMap, err := cl.loadMap(objName)
				if err != nil {
					return fmt.Errorf("loading inner map %s, for map %s: %w", objName, mapName, err)
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

// resolveKconfig resolves all variables declared in .kconfig and populates
// m.Contents. Does nothing if the given m.Contents is non-empty.
func resolveKconfig(m *MapSpec) error {
	ds, ok := m.Value.(*btf.Datasec)
	if !ok {
		return errors.New("map value is not a Datasec")
	}

	type configInfo struct {
		offset uint32
		typ    btf.Type
	}

	configs := make(map[string]configInfo)

	data := make([]byte, ds.Size)
	for _, vsi := range ds.Vars {
		v := vsi.Type.(*btf.Var)
		n := v.TypeName()

		switch n {
		case "LINUX_KERNEL_VERSION":
			if integer, ok := v.Type.(*btf.Int); !ok || integer.Size != 4 {
				return fmt.Errorf("variable %s must be a 32 bits integer, got %s", n, v.Type)
			}

			kv, err := internal.KernelVersion()
			if err != nil {
				return fmt.Errorf("getting kernel version: %w", err)
			}
			internal.NativeEndian.PutUint32(data[vsi.Offset:], kv.Kernel())

		case "LINUX_HAS_SYSCALL_WRAPPER":
			integer, ok := v.Type.(*btf.Int)
			if !ok {
				return fmt.Errorf("variable %s must be an integer, got %s", n, v.Type)
			}
			var value uint64 = 1
			if err := haveSyscallWrapper(); errors.Is(err, ErrNotSupported) {
				value = 0
			} else if err != nil {
				return fmt.Errorf("unable to derive a value for LINUX_HAS_SYSCALL_WRAPPER: %w", err)
			}

			if err := kconfig.PutInteger(data[vsi.Offset:], integer, value); err != nil {
				return fmt.Errorf("set LINUX_HAS_SYSCALL_WRAPPER: %w", err)
			}

		default: // Catch CONFIG_*.
			configs[n] = configInfo{
				offset: vsi.Offset,
				typ:    v.Type,
			}
		}
	}

	// We only parse kconfig file if a CONFIG_* variable was found.
	if len(configs) > 0 {
		f, err := kconfig.Find()
		if err != nil {
			return fmt.Errorf("cannot find a kconfig file: %w", err)
		}
		defer f.Close()

		filter := make(map[string]struct{}, len(configs))
		for config := range configs {
			filter[config] = struct{}{}
		}

		kernelConfig, err := kconfig.Parse(f, filter)
		if err != nil {
			return fmt.Errorf("cannot parse kconfig file: %w", err)
		}

		for n, info := range configs {
			value, ok := kernelConfig[n]
			if !ok {
				return fmt.Errorf("config option %q does not exists for this kernel", n)
			}

			err := kconfig.PutValue(data[info.offset:], info.typ, value)
			if err != nil {
				return fmt.Errorf("problem adding value for %s: %w", n, err)
			}
		}
	}

	m.Contents = []MapKV{{uint32(0), data}}

	return nil
}

// LoadCollection reads an object file and creates and loads its declared
// resources into the kernel.
//
// Omitting Collection.Close() during application shutdown is an error.
// See the package documentation for details around Map and Program lifecycle.
func LoadCollection(file string) (*Collection, error) {
	spec, err := LoadCollectionSpec(file)
	if err != nil {
		return nil, err
	}
	return NewCollection(spec)
}

// Assign the contents of a Collection to a struct.
//
// This function bridges functionality between bpf2go generated
// code and any functionality better implemented in Collection.
//
// 'to' must be a pointer to a struct. A field of the
// struct is updated with values from Programs or Maps if it
// has an `ebpf` tag and its type is *Program or *Map.
// The tag's value specifies the name of the program or map as
// found in the CollectionSpec.
//
//	struct {
//	    Foo     *ebpf.Program `ebpf:"xdp_foo"`
//	    Bar     *ebpf.Map     `ebpf:"bar_map"`
//	    Ignored int
//	}
//
// Returns an error if any of the eBPF objects can't be found, or
// if the same Map or Program is assigned multiple times.
//
// Ownership and Close()ing responsibility is transferred to `to`
// for any successful assigns. On error `to` is left in an undefined state.
func (coll *Collection) Assign(to interface{}) error {
	assignedMaps := make(map[string]bool)
	assignedProgs := make(map[string]bool)

	// Assign() only transfers already-loaded Maps and Programs. No extra
	// loading is done.
	getValue := func(typ reflect.Type, name string) (interface{}, error) {
		switch typ {

		case reflect.TypeOf((*Program)(nil)):
			if p := coll.Programs[name]; p != nil {
				assignedProgs[name] = true
				return p, nil
			}
			return nil, fmt.Errorf("missing program %q", name)

		case reflect.TypeOf((*Map)(nil)):
			if m := coll.Maps[name]; m != nil {
				assignedMaps[name] = true
				return m, nil
			}
			return nil, fmt.Errorf("missing map %q", name)

		default:
			return nil, fmt.Errorf("unsupported type %s", typ)
		}
	}

	if err := assignValues(to, getValue); err != nil {
		return err
	}

	// Finalize ownership transfer
	for p := range assignedProgs {
		delete(coll.Programs, p)
	}
	for m := range assignedMaps {
		delete(coll.Maps, m)
	}

	return nil
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
