package packfile

import (
	"sort"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
)

const (
	// How far back in the sorted list to search for deltas.  10 is
	// the default in command line git.
	deltaWindowSize = 10
	// deltas based on deltas, how many steps we can do.
	// 50 is the default value used in JGit
	maxDepth = int64(50)
)

// applyDelta is the set of object types that we should apply deltas
var applyDelta = map[plumbing.ObjectType]bool{
	plumbing.BlobObject: true,
	plumbing.TreeObject: true,
}

type deltaSelector struct {
	storer storer.EncodedObjectStorer
}

func newDeltaSelector(s storer.EncodedObjectStorer) *deltaSelector {
	return &deltaSelector{s}
}

// ObjectsToPack creates a list of ObjectToPack from the hashes provided,
// creating deltas if it's suitable, using an specific internal logic
func (dw *deltaSelector) ObjectsToPack(hashes []plumbing.Hash) ([]*ObjectToPack, error) {
	otp, err := dw.objectsToPack(hashes)
	if err != nil {
		return nil, err
	}

	dw.sort(otp)

	if err := dw.walk(otp); err != nil {
		return nil, err
	}

	return otp, nil
}

func (dw *deltaSelector) objectsToPack(hashes []plumbing.Hash) ([]*ObjectToPack, error) {
	var objectsToPack []*ObjectToPack
	for _, h := range hashes {
		o, err := dw.encodedDeltaObject(h)
		if err != nil {
			return nil, err
		}

		otp := newObjectToPack(o)
		if _, ok := o.(plumbing.DeltaObject); ok {
			otp.Original = nil
		}

		objectsToPack = append(objectsToPack, otp)
	}

	if err := dw.fixAndBreakChains(objectsToPack); err != nil {
		return nil, err
	}

	return objectsToPack, nil
}

func (dw *deltaSelector) encodedDeltaObject(h plumbing.Hash) (plumbing.EncodedObject, error) {
	edos, ok := dw.storer.(storer.DeltaObjectStorer)
	if !ok {
		return dw.encodedObject(h)
	}

	return edos.DeltaObject(plumbing.AnyObject, h)
}

func (dw *deltaSelector) encodedObject(h plumbing.Hash) (plumbing.EncodedObject, error) {
	return dw.storer.EncodedObject(plumbing.AnyObject, h)
}

func (dw *deltaSelector) fixAndBreakChains(objectsToPack []*ObjectToPack) error {
	m := make(map[plumbing.Hash]*ObjectToPack, len(objectsToPack))
	for _, otp := range objectsToPack {
		m[otp.Hash()] = otp
	}

	for _, otp := range objectsToPack {
		if err := dw.fixAndBreakChainsOne(m, otp); err != nil {
			return err
		}
	}

	return nil
}

func (dw *deltaSelector) fixAndBreakChainsOne(objectsToPack map[plumbing.Hash]*ObjectToPack, otp *ObjectToPack) error {
	if !otp.Object.Type().IsDelta() {
		return nil
	}

	// Initial ObjectToPack instances might have a delta assigned to Object
	// but no actual base initially. Once Base is assigned to a delta, it means
	// we already fixed it.
	if otp.Base != nil {
		return nil
	}

	do, ok := otp.Object.(plumbing.DeltaObject)
	if !ok {
		// if this is not a DeltaObject, then we cannot retrieve its base,
		// so we have to break the delta chain here.
		return dw.undeltify(otp)
	}

	base, ok := objectsToPack[do.BaseHash()]
	if !ok {
		// The base of the delta is not in our list of objects to pack, so
		// we break the chain.
		return dw.undeltify(otp)
	}

	if base.Size() <= otp.Size() {
		// Bases should be bigger
		return dw.undeltify(otp)
	}

	if err := dw.fixAndBreakChainsOne(objectsToPack, base); err != nil {
		return err
	}

	otp.SetDelta(base, otp.Object)
	return nil
}

func (dw *deltaSelector) restoreOriginal(otp *ObjectToPack) error {
	if otp.Original != nil {
		return nil
	}

	if !otp.Object.Type().IsDelta() {
		return nil
	}

	obj, err := dw.encodedObject(otp.Hash())
	if err != nil {
		return err
	}

	otp.Original = obj
	return nil
}

// undeltify undeltifies an *ObjectToPack by retrieving the original object from
// the storer and resetting it.
func (dw *deltaSelector) undeltify(otp *ObjectToPack) error {
	if err := dw.restoreOriginal(otp); err != nil {
		return err
	}

	otp.Object = otp.Original
	otp.Depth = 0
	return nil
}

func (dw *deltaSelector) sort(objectsToPack []*ObjectToPack) {
	sort.Sort(byTypeAndSize(objectsToPack))
}

func (dw *deltaSelector) walk(objectsToPack []*ObjectToPack) error {
	for i := 0; i < len(objectsToPack); i++ {
		target := objectsToPack[i]

		// If we already have a delta, we don't try to find a new one for this
		// object. This happens when a delta is set to be reused from an existing
		// packfile.
		if target.IsDelta() {
			continue
		}

		// We only want to create deltas from specific types.
		if !applyDelta[target.Type()] {
			continue
		}

		for j := i - 1; j >= 0 && i-j < deltaWindowSize; j-- {
			base := objectsToPack[j]
			// Objects must use only the same type as their delta base.
			// Since objectsToPack is sorted by type and size, once we find
			// a different type, we know we won't find more of them.
			if base.Type() != target.Type() {
				break
			}

			if err := dw.tryToDeltify(base, target); err != nil {
				return err
			}
		}
	}

	return nil
}

func (dw *deltaSelector) tryToDeltify(base, target *ObjectToPack) error {
	// If the sizes are radically different, this is a bad pairing.
	if target.Size() < base.Size()>>4 {
		return nil
	}

	msz := dw.deltaSizeLimit(
		target.Object.Size(),
		base.Depth,
		target.Depth,
		target.IsDelta(),
	)

	// Nearly impossible to fit useful delta.
	if msz <= 8 {
		return nil
	}

	// If we have to insert a lot to make this work, find another.
	if base.Size()-target.Size() > msz {
		return nil
	}

	// Original object might not be present if we're reusing a delta, so we
	// ensure it is restored.
	if err := dw.restoreOriginal(target); err != nil {
		return err
	}

	if err := dw.restoreOriginal(base); err != nil {
		return err
	}

	// Now we can generate the delta using originals
	delta, err := GetDelta(base.Original, target.Original)
	if err != nil {
		return err
	}

	// if delta better than target
	if delta.Size() < msz {
		target.SetDelta(base, delta)
	}

	return nil
}

func (dw *deltaSelector) deltaSizeLimit(targetSize int64, baseDepth int,
	targetDepth int, targetDelta bool) int64 {
	if !targetDelta {
		// Any delta should be no more than 50% of the original size
		// (for text files deflate of whole form should shrink 50%).
		n := targetSize >> 1

		// Evenly distribute delta size limits over allowed depth.
		// If src is non-delta (depth = 0), delta <= 50% of original.
		// If src is almost at limit (9/10), delta <= 10% of original.
		return n * (maxDepth - int64(baseDepth)) / maxDepth
	}

	// With a delta base chosen any new delta must be "better".
	// Retain the distribution described above.
	d := int64(targetDepth)
	n := targetSize

	// If target depth is bigger than maxDepth, this delta is not suitable to be used.
	if d >= maxDepth {
		return 0
	}

	// If src is whole (depth=0) and base is near limit (depth=9/10)
	// any delta using src can be 10x larger and still be better.
	//
	// If src is near limit (depth=9/10) and base is whole (depth=0)
	// a new delta dependent on src must be 1/10th the size.
	return n * (maxDepth - int64(baseDepth)) / (maxDepth - d)
}

type byTypeAndSize []*ObjectToPack

func (a byTypeAndSize) Len() int { return len(a) }

func (a byTypeAndSize) Swap(i, j int) { a[i], a[j] = a[j], a[i] }

func (a byTypeAndSize) Less(i, j int) bool {
	if a[i].Type() < a[j].Type() {
		return false
	}

	if a[i].Type() > a[j].Type() {
		return true
	}

	return a[i].Size() > a[j].Size()
}
