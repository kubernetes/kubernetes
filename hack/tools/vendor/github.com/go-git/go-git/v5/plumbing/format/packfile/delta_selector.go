package packfile

import (
	"sort"
	"sync"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/storer"
)

const (
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

// ObjectsToPack creates a list of ObjectToPack from the hashes
// provided, creating deltas if it's suitable, using an specific
// internal logic.  `packWindow` specifies the size of the sliding
// window used to compare objects for delta compression; 0 turns off
// delta compression entirely.
func (dw *deltaSelector) ObjectsToPack(
	hashes []plumbing.Hash,
	packWindow uint,
) ([]*ObjectToPack, error) {
	otp, err := dw.objectsToPack(hashes, packWindow)
	if err != nil {
		return nil, err
	}

	if packWindow == 0 {
		return otp, nil
	}

	dw.sort(otp)

	var objectGroups [][]*ObjectToPack
	var prev *ObjectToPack
	i := -1
	for _, obj := range otp {
		if prev == nil || prev.Type() != obj.Type() {
			objectGroups = append(objectGroups, []*ObjectToPack{obj})
			i++
			prev = obj
		} else {
			objectGroups[i] = append(objectGroups[i], obj)
		}
	}

	var wg sync.WaitGroup
	var once sync.Once
	for _, objs := range objectGroups {
		objs := objs
		wg.Add(1)
		go func() {
			if walkErr := dw.walk(objs, packWindow); walkErr != nil {
				once.Do(func() {
					err = walkErr
				})
			}
			wg.Done()
		}()
	}
	wg.Wait()

	if err != nil {
		return nil, err
	}

	return otp, nil
}

func (dw *deltaSelector) objectsToPack(
	hashes []plumbing.Hash,
	packWindow uint,
) ([]*ObjectToPack, error) {
	var objectsToPack []*ObjectToPack
	for _, h := range hashes {
		var o plumbing.EncodedObject
		var err error
		if packWindow == 0 {
			o, err = dw.encodedObject(h)
		} else {
			o, err = dw.encodedDeltaObject(h)
		}
		if err != nil {
			return nil, err
		}

		otp := newObjectToPack(o)
		if _, ok := o.(plumbing.DeltaObject); ok {
			otp.CleanOriginal()
		}

		objectsToPack = append(objectsToPack, otp)
	}

	if packWindow == 0 {
		return objectsToPack, nil
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

	otp.SetOriginal(obj)

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

func (dw *deltaSelector) walk(
	objectsToPack []*ObjectToPack,
	packWindow uint,
) error {
	indexMap := make(map[plumbing.Hash]*deltaIndex)
	for i := 0; i < len(objectsToPack); i++ {
		// Clean up the index map and reconstructed delta objects for anything
		// outside our pack window, to save memory.
		if i > int(packWindow) {
			obj := objectsToPack[i-int(packWindow)]

			delete(indexMap, obj.Hash())

			if obj.IsDelta() {
				obj.SaveOriginalMetadata()
				obj.CleanOriginal()
			}
		}

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

		for j := i - 1; j >= 0 && i-j < int(packWindow); j-- {
			base := objectsToPack[j]
			// Objects must use only the same type as their delta base.
			// Since objectsToPack is sorted by type and size, once we find
			// a different type, we know we won't find more of them.
			if base.Type() != target.Type() {
				break
			}

			if err := dw.tryToDeltify(indexMap, base, target); err != nil {
				return err
			}
		}
	}

	return nil
}

func (dw *deltaSelector) tryToDeltify(indexMap map[plumbing.Hash]*deltaIndex, base, target *ObjectToPack) error {
	// Original object might not be present if we're reusing a delta, so we
	// ensure it is restored.
	if err := dw.restoreOriginal(target); err != nil {
		return err
	}

	if err := dw.restoreOriginal(base); err != nil {
		return err
	}

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

	if _, ok := indexMap[base.Hash()]; !ok {
		indexMap[base.Hash()] = new(deltaIndex)
	}

	// Now we can generate the delta using originals
	delta, err := getDelta(indexMap[base.Hash()], base.Original, target.Original)
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
