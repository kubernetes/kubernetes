package uid

import (
	"fmt"
	"strings"
)

type Block struct {
	Start uint32
	End   uint32
}

var (
	ErrBlockSlashBadFormat = fmt.Errorf("block not in the format \"<start>/<size>\"")
	ErrBlockDashBadFormat  = fmt.Errorf("block not in the format \"<start>-<end>\"")
)

func ParseBlock(in string) (Block, error) {
	if strings.Contains(in, "/") {
		var start, size uint32
		n, err := fmt.Sscanf(in, "%d/%d", &start, &size)
		if err != nil {
			return Block{}, err
		}
		if n != 2 {
			return Block{}, ErrBlockSlashBadFormat
		}
		return Block{Start: start, End: start + size - 1}, nil
	}

	var start, end uint32
	n, err := fmt.Sscanf(in, "%d-%d", &start, &end)
	if err != nil {
		return Block{}, err
	}
	if n != 2 {
		return Block{}, ErrBlockDashBadFormat
	}
	return Block{Start: start, End: end}, nil
}

func (b Block) String() string {
	return fmt.Sprintf("%d/%d", b.Start, b.Size())
}

func (b Block) RangeString() string {
	return fmt.Sprintf("%d-%d", b.Start, b.End)
}

func (b Block) Size() uint32 {
	return b.End - b.Start + 1
}

type Range struct {
	block Block
	size  uint32
}

func NewRange(start, end, size uint32) (*Range, error) {
	if start > end {
		return nil, fmt.Errorf("start %d must be less than end %d", start, end)
	}
	if size == 0 {
		return nil, fmt.Errorf("block size must be a positive integer")
	}
	if (end - start) < size {
		return nil, fmt.Errorf("block size must be less than or equal to the range")
	}
	return &Range{
		block: Block{start, end},
		size:  size,
	}, nil
}

func ParseRange(in string) (*Range, error) {
	var start, end, block uint32
	n, err := fmt.Sscanf(in, "%d-%d/%d", &start, &end, &block)
	if err != nil {
		return nil, err
	}
	if n != 3 {
		return nil, fmt.Errorf("range not in the format \"<start>-<end>/<blockSize>\"")
	}
	return NewRange(start, end, block)
}

func (r *Range) Size() uint32 {
	return r.block.Size() / r.size
}

func (r *Range) String() string {
	return fmt.Sprintf("%s/%d", r.block.RangeString(), r.size)
}

func (r *Range) BlockAt(offset uint32) (Block, bool) {
	if offset > r.Size() {
		return Block{}, false
	}
	start := r.block.Start + offset*r.size
	return Block{
		Start: start,
		End:   start + r.size - 1,
	}, true
}

func (r *Range) Contains(block Block) bool {
	ok, _ := r.Offset(block)
	return ok
}

func (r *Range) Offset(block Block) (bool, uint32) {
	if block.Start < r.block.Start {
		return false, 0
	}
	if block.End > r.block.End {
		return false, 0
	}
	if block.End-block.Start+1 != r.size {
		return false, 0
	}
	if (block.Start-r.block.Start)%r.size != 0 {
		return false, 0
	}
	return true, (block.Start - r.block.Start) / r.size
}
