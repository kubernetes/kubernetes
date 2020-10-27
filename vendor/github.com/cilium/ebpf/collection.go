package ebpf

import (
	"errors"
	"fmt"
	"math"

	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/btf"
)

// CollectionOptions control loading a collection into the kernel.
type CollectionOptions struct {
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
//    static volatile const type foobar;
//    static volatile const type foobar = default;
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

// Collection is a collection of Programs and Maps associated
// with their symbols
type Collection struct {
	Programs map[string]*Program
	Maps     map[string]*Map
}

// NewCollection creates a Collection from a specification.
//
// Only maps referenced by at least one of the programs are initialized.
func NewCollection(spec *CollectionSpec) (*Collection, error) {
	return NewCollectionWithOptions(spec, CollectionOptions{})
}

// NewCollectionWithOptions creates a Collection from a specification.
//
// Only maps referenced by at least one of the programs are initialized.
func NewCollectionWithOptions(spec *CollectionSpec, opts CollectionOptions) (coll *Collection, err error) {
	var (
		maps  = make(map[string]*Map)
		progs = make(map[string]*Program)
		btfs  = make(map[*btf.Spec]*btf.Handle)
	)

	defer func() {
		for _, btf := range btfs {
			btf.Close()
		}

		if err == nil {
			return
		}

		for _, m := range maps {
			m.Close()
		}

		for _, p := range progs {
			p.Close()
		}
	}()

	loadBTF := func(spec *btf.Spec) (*btf.Handle, error) {
		if btfs[spec] != nil {
			return btfs[spec], nil
		}

		handle, err := btf.NewHandle(spec)
		if err != nil {
			return nil, err
		}

		btfs[spec] = handle
		return handle, nil
	}

	for mapName, mapSpec := range spec.Maps {
		var handle *btf.Handle
		if mapSpec.BTF != nil {
			handle, err = loadBTF(btf.MapSpec(mapSpec.BTF))
			if err != nil && !errors.Is(err, btf.ErrNotSupported) {
				return nil, err
			}
		}

		m, err := newMapWithBTF(mapSpec, handle)
		if err != nil {
			return nil, fmt.Errorf("map %s: %w", mapName, err)
		}
		maps[mapName] = m
	}

	for progName, origProgSpec := range spec.Programs {
		progSpec := origProgSpec.Copy()

		// Rewrite any reference to a valid map.
		for i := range progSpec.Instructions {
			ins := &progSpec.Instructions[i]

			if ins.OpCode != asm.LoadImmOp(asm.DWord) || ins.Reference == "" {
				continue
			}

			if uint32(ins.Constant) != math.MaxUint32 {
				// Don't overwrite maps already rewritten, users can
				// rewrite programs in the spec themselves
				continue
			}

			m := maps[ins.Reference]
			if m == nil {
				return nil, fmt.Errorf("program %s: missing map %s", progName, ins.Reference)
			}

			fd := m.FD()
			if fd < 0 {
				return nil, fmt.Errorf("map %s: %w", ins.Reference, internal.ErrClosedFd)
			}
			if err := ins.RewriteMapPtr(m.FD()); err != nil {
				return nil, fmt.Errorf("progam %s: map %s: %w", progName, ins.Reference, err)
			}
		}

		var handle *btf.Handle
		if progSpec.BTF != nil {
			handle, err = loadBTF(btf.ProgramSpec(progSpec.BTF))
			if err != nil && !errors.Is(err, btf.ErrNotSupported) {
				return nil, err
			}
		}

		prog, err := newProgramWithBTF(progSpec, handle, opts.Programs)
		if err != nil {
			return nil, fmt.Errorf("program %s: %w", progName, err)
		}
		progs[progName] = prog
	}

	return &Collection{
		progs,
		maps,
	}, nil
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
