package ebpf

import (
	"github.com/cilium/ebpf/asm"
	"github.com/pkg/errors"
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
func NewCollectionWithOptions(spec *CollectionSpec, opts CollectionOptions) (*Collection, error) {
	maps := make(map[string]*Map)
	for mapName, mapSpec := range spec.Maps {
		m, err := NewMap(mapSpec)
		if err != nil {
			return nil, errors.Wrapf(err, "map %s", mapName)
		}
		maps[mapName] = m
	}

	progs := make(map[string]*Program)
	for progName, origProgSpec := range spec.Programs {
		progSpec := origProgSpec.Copy()

		// Rewrite any reference to a valid map.
		for i := range progSpec.Instructions {
			var (
				ins = &progSpec.Instructions[i]
				m   = maps[ins.Reference]
			)

			if ins.Reference == "" || m == nil {
				continue
			}

			if ins.Src == asm.R1 {
				// Don't overwrite maps already rewritten, users can
				// rewrite programs in the spec themselves
				continue
			}

			if err := ins.RewriteMapPtr(m.FD()); err != nil {
				return nil, errors.Wrapf(err, "progam %s: map %s", progName, ins.Reference)
			}
		}

		prog, err := NewProgramWithOptions(progSpec, opts.Programs)
		if err != nil {
			return nil, errors.Wrapf(err, "program %s", progName)
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
