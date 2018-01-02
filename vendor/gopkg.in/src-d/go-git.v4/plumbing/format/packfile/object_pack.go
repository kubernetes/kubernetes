package packfile

import (
	"gopkg.in/src-d/go-git.v4/plumbing"
)

// ObjectToPack is a representation of an object that is going to be into a
// pack file.
type ObjectToPack struct {
	// The main object to pack, it could be any object, including deltas
	Object plumbing.EncodedObject
	// Base is the object that a delta is based on (it could be also another delta).
	// If the main object is not a delta, Base will be null
	Base *ObjectToPack
	// Original is the object that we can generate applying the delta to
	// Base, or the same object as EncodedObject in the case of a non-delta
	// object.
	Original plumbing.EncodedObject
	// Depth is the amount of deltas needed to resolve to obtain Original
	// (delta based on delta based on ...)
	Depth int
}

// newObjectToPack creates a correct ObjectToPack based on a non-delta object
func newObjectToPack(o plumbing.EncodedObject) *ObjectToPack {
	return &ObjectToPack{
		Object:   o,
		Original: o,
	}
}

// newDeltaObjectToPack creates a correct ObjectToPack for a delta object, based on
// his base (could be another delta), the delta target (in this case called original),
// and the delta Object itself
func newDeltaObjectToPack(base *ObjectToPack, original, delta plumbing.EncodedObject) *ObjectToPack {
	return &ObjectToPack{
		Object:   delta,
		Base:     base,
		Original: original,
		Depth:    base.Depth + 1,
	}
}

func (o *ObjectToPack) Type() plumbing.ObjectType {
	if o.Original != nil {
		return o.Original.Type()
	}

	if o.Base != nil {
		return o.Base.Type()
	}

	if o.Object != nil {
		return o.Object.Type()
	}

	panic("cannot get type")
}

func (o *ObjectToPack) Hash() plumbing.Hash {
	if o.Original != nil {
		return o.Original.Hash()
	}

	do, ok := o.Object.(plumbing.DeltaObject)
	if ok {
		return do.ActualHash()
	}

	panic("cannot get hash")
}

func (o *ObjectToPack) Size() int64 {
	if o.Original != nil {
		return o.Original.Size()
	}

	do, ok := o.Object.(plumbing.DeltaObject)
	if ok {
		return do.ActualSize()
	}

	panic("cannot get ObjectToPack size")
}

func (o *ObjectToPack) IsDelta() bool {
	if o.Base != nil {
		return true
	}

	return false
}

func (o *ObjectToPack) SetDelta(base *ObjectToPack, delta plumbing.EncodedObject) {
	o.Object = delta
	o.Base = base
	o.Depth = base.Depth + 1
}
