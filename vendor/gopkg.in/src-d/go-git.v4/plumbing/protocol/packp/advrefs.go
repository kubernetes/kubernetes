package packp

import (
	"fmt"
	"strings"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/capability"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
	"gopkg.in/src-d/go-git.v4/storage/memory"
)

// AdvRefs values represent the information transmitted on an
// advertised-refs message.  Values from this type are not zero-value
// safe, use the New function instead.
type AdvRefs struct {
	// Prefix stores prefix payloads.
	//
	// When using this message over (smart) HTTP, you have to add a pktline
	// before the whole thing with the following payload:
	//
	// '# service=$servicename" LF
	//
	// Moreover, some (all) git HTTP smart servers will send a flush-pkt
	// just after the first pkt-line.
	//
	// To accommodate both situations, the Prefix field allow you to store
	// any data you want to send before the actual pktlines.  It will also
	// be filled up with whatever is found on the line.
	Prefix [][]byte
	// Head stores the resolved HEAD reference if present.
	// This can be present with git-upload-pack, not with git-receive-pack.
	Head *plumbing.Hash
	// Capabilities are the capabilities.
	Capabilities *capability.List
	// References are the hash references.
	References map[string]plumbing.Hash
	// Peeled are the peeled hash references.
	Peeled map[string]plumbing.Hash
	// Shallows are the shallow object ids.
	Shallows []plumbing.Hash
}

// NewAdvRefs returns a pointer to a new AdvRefs value, ready to be used.
func NewAdvRefs() *AdvRefs {
	return &AdvRefs{
		Prefix:       [][]byte{},
		Capabilities: capability.NewList(),
		References:   make(map[string]plumbing.Hash),
		Peeled:       make(map[string]plumbing.Hash),
		Shallows:     []plumbing.Hash{},
	}
}

func (a *AdvRefs) AddReference(r *plumbing.Reference) error {
	switch r.Type() {
	case plumbing.SymbolicReference:
		v := fmt.Sprintf("%s:%s", r.Name().String(), r.Target().String())
		a.Capabilities.Add(capability.SymRef, v)
	case plumbing.HashReference:
		a.References[r.Name().String()] = r.Hash()
	default:
		return plumbing.ErrInvalidType
	}

	return nil
}

func (a *AdvRefs) AllReferences() (memory.ReferenceStorage, error) {
	s := memory.ReferenceStorage{}
	if err := addRefs(s, a); err != nil {
		return s, plumbing.NewUnexpectedError(err)
	}

	return s, nil
}

func addRefs(s storer.ReferenceStorer, ar *AdvRefs) error {
	for name, hash := range ar.References {
		ref := plumbing.NewReferenceFromStrings(name, hash.String())
		if err := s.SetReference(ref); err != nil {
			return err
		}
	}

	return addSymbolicRefs(s, ar)
}

func addSymbolicRefs(s storer.ReferenceStorer, ar *AdvRefs) error {
	if !hasSymrefs(ar) {
		return nil
	}

	for _, symref := range ar.Capabilities.Get(capability.SymRef) {
		chunks := strings.Split(symref, ":")
		if len(chunks) != 2 {
			err := fmt.Errorf("bad number of `:` in symref value (%q)", symref)
			return plumbing.NewUnexpectedError(err)
		}
		name := plumbing.ReferenceName(chunks[0])
		target := plumbing.ReferenceName(chunks[1])
		ref := plumbing.NewSymbolicReference(name, target)
		if err := s.SetReference(ref); err != nil {
			return nil
		}
	}

	return nil
}

func hasSymrefs(ar *AdvRefs) bool {
	return ar.Capabilities.Supports(capability.SymRef)
}
