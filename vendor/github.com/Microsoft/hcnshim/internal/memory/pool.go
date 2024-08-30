package memory

import (
	"github.com/pkg/errors"
)

const (
	minimumClassSize  = MiB
	maximumClassSize  = 4 * GiB
	memoryClassNumber = 7
)

var (
	ErrInvalidMemoryClass = errors.New("invalid memory class")
	ErrEarlyMerge         = errors.New("not all children have been freed")
	ErrEmptyPoolOperation = errors.New("operation on empty pool")
)

// GetMemoryClassType returns the minimum memory class type that can hold a device of
// a given size. The smallest class is 1MB and the largest one is 4GB with 2 bit offset
// intervals in between, for a total of 7 different classes. This function does not
// do a validity check
func GetMemoryClassType(s uint64) classType {
	s = (s - 1) >> 20
	memCls := uint32(0)
	for s > 0 {
		s = s >> 2
		memCls++
	}
	return classType(memCls)
}

// GetMemoryClassSize returns size in bytes for a given memory class
func GetMemoryClassSize(memCls classType) (uint64, error) {
	if memCls >= memoryClassNumber {
		return 0, ErrInvalidMemoryClass
	}
	return minimumClassSize << (2 * memCls), nil
}

// region represents a contiguous memory block
type region struct {
	// parent region that has been split into 4
	parent *region
	class  classType
	// offset represents offset in bytes
	offset uint64
}

// memoryPool tracks free and busy (used) memory regions
type memoryPool struct {
	free map[uint64]*region
	busy map[uint64]*region
}

// PoolAllocator implements a memory allocation strategy similar to buddy-malloc https://github.com/evanw/buddy-malloc/blob/master/buddy-malloc.c
// We borrow the idea of spanning a tree of fixed size regions on top of a contiguous memory
// space.
//
// There are a total of 7 different region sizes that can be allocated, with the smallest
// being 1MB and the largest 4GB (the default maximum size of a Virtual PMem device).
//
// For efficiency and to reduce fragmentation an entire region is allocated when requested.
// When there's no available region of requested size, we try to allocate more memory for
// this particular size by splitting the next available larger region into smaller ones, e.g.
// if there's no region available for size class 0, we try splitting a region from class 1,
// then class 2 etc, until we are able to do so or hit the upper limit.
type PoolAllocator struct {
	pools [memoryClassNumber]*memoryPool
}

var _ MappedRegion = &region{}
var _ Allocator = &PoolAllocator{}

func (r *region) Offset() uint64 {
	return r.offset
}

func (r *region) Size() uint64 {
	sz, err := GetMemoryClassSize(r.class)
	if err != nil {
		panic(err)
	}
	return sz
}

func (r *region) Type() classType {
	return r.class
}

func newEmptyMemoryPool() *memoryPool {
	return &memoryPool{
		free: make(map[uint64]*region),
		busy: make(map[uint64]*region),
	}
}

func NewPoolMemoryAllocator() PoolAllocator {
	pa := PoolAllocator{}
	p := newEmptyMemoryPool()
	// by default we allocate a single region with maximum possible size (class type)
	p.free[0] = &region{
		class:  memoryClassNumber - 1,
		offset: 0,
	}
	pa.pools[memoryClassNumber-1] = p
	return pa
}

// Allocate checks memory region pool for the given `size` and returns a free region with
// minimal offset, if none available tries expanding matched memory pool.
//
// Internally it's done via moving a region from free pool into a busy pool
func (pa *PoolAllocator) Allocate(size uint64) (MappedRegion, error) {
	memCls := GetMemoryClassType(size)
	if memCls >= memoryClassNumber {
		return nil, ErrInvalidMemoryClass
	}

	// find region with the smallest offset
	nextCls, nextOffset, err := pa.findNextOffset(memCls)
	if err != nil {
		return nil, err
	}

	// this means that there are no more regions for the current class, try expanding
	if nextCls != memCls {
		if err := pa.split(memCls); err != nil {
			if errors.Is(err, ErrInvalidMemoryClass) {
				return nil, ErrNotEnoughSpace
			}
			return nil, err
		}
	}

	if err := pa.markBusy(memCls, nextOffset); err != nil {
		return nil, err
	}

	// by this point memory pool for memCls should have been created,
	// either prior or during split call
	if r := pa.pools[memCls].busy[nextOffset]; r != nil {
		return r, nil
	}

	return nil, ErrNotEnoughSpace
}

// Release marks a memory region of class `memCls` and offset `offset` as free and tries to merge smaller regions into
// a bigger one.
func (pa *PoolAllocator) Release(reg MappedRegion) error {
	mp := pa.pools[reg.Type()]
	if mp == nil {
		return ErrEmptyPoolOperation
	}

	err := pa.markFree(reg.Type(), reg.Offset())
	if err != nil {
		return err
	}

	n := mp.free[reg.Offset()]
	if n == nil {
		return ErrNotAllocated
	}
	if err := pa.merge(n.parent); err != nil {
		if !errors.Is(err, ErrEarlyMerge) {
			return err
		}
	}
	return nil
}

// findNextOffset finds next region location for a given memCls
func (pa *PoolAllocator) findNextOffset(memCls classType) (classType, uint64, error) {
	for mc := memCls; mc < memoryClassNumber; mc++ {
		pi := pa.pools[mc]
		if pi == nil || len(pi.free) == 0 {
			continue
		}

		target := uint64(maximumClassSize)
		for offset := range pi.free {
			if offset < target {
				target = offset
			}
		}
		return mc, target, nil
	}
	return 0, 0, ErrNotEnoughSpace
}

// split tries to recursively split a bigger memory region into smaller ones until it succeeds or hits the upper limit
func (pa *PoolAllocator) split(clsType classType) error {
	nextClsType := clsType + 1
	if nextClsType >= memoryClassNumber {
		return ErrInvalidMemoryClass
	}

	nextPool := pa.pools[nextClsType]
	if nextPool == nil {
		nextPool = newEmptyMemoryPool()
		pa.pools[nextClsType] = nextPool
	}

	cls, offset, err := pa.findNextOffset(nextClsType)
	if err != nil {
		return err
	}
	// not enough memory in the next class, try to recursively expand
	if cls != nextClsType {
		if err := pa.split(nextClsType); err != nil {
			return err
		}
	}

	if err := pa.markBusy(nextClsType, offset); err != nil {
		return err
	}

	// memCls validity has been checked already, we can ignore the error
	clsSize, _ := GetMemoryClassSize(clsType)

	nextReg := nextPool.busy[offset]
	if nextReg == nil {
		return ErrNotAllocated
	}

	// expand memCls
	cp := pa.pools[clsType]
	if cp == nil {
		cp = newEmptyMemoryPool()
		pa.pools[clsType] = cp
	}
	// create 4 smaller regions
	for i := uint64(0); i < 4; i++ {
		offset := nextReg.offset + i*clsSize
		reg := &region{
			parent: nextReg,
			class:  clsType,
			offset: offset,
		}
		cp.free[offset] = reg
	}
	return nil
}

func (pa *PoolAllocator) merge(parent *region) error {
	// nothing to merge
	if parent == nil {
		return nil
	}

	childCls := parent.class - 1
	childPool := pa.pools[childCls]
	// no child nodes to merge, try to merge parent
	if childPool == nil {
		return pa.merge(parent.parent)
	}

	childSize, err := GetMemoryClassSize(childCls)
	if err != nil {
		return err
	}

	// check if all the child nodes are free
	var children []*region
	for i := uint64(0); i < 4; i++ {
		child, free := childPool.free[parent.offset+i*childSize]
		if !free {
			return ErrEarlyMerge
		}
		children = append(children, child)
	}

	// at this point all the child nodes will be free and we can merge
	for _, child := range children {
		delete(childPool.free, child.offset)
	}

	if err := pa.markFree(parent.class, parent.offset); err != nil {
		return err
	}

	return pa.merge(parent.parent)
}

// markFree internally moves a region with `offset` from busy to free map
func (pa *PoolAllocator) markFree(memCls classType, offset uint64) error {
	clsPool := pa.pools[memCls]
	if clsPool == nil {
		return ErrEmptyPoolOperation
	}

	if reg, exists := clsPool.busy[offset]; exists {
		clsPool.free[offset] = reg
		delete(clsPool.busy, offset)
		return nil
	}
	return ErrNotAllocated
}

// markBusy internally moves a region with `offset` from free to busy map
func (pa *PoolAllocator) markBusy(memCls classType, offset uint64) error {
	clsPool := pa.pools[memCls]
	if clsPool == nil {
		return ErrEmptyPoolOperation
	}

	if reg, exists := clsPool.free[offset]; exists {
		clsPool.busy[offset] = reg
		delete(clsPool.free, offset)
		return nil
	}
	return ErrNotAllocated
}
