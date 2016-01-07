package ioutils

import (
	"bytes"
	"fmt"
	"io"
	"os"
)

type pos struct {
	idx    int
	offset int64
}

type multiReadSeeker struct {
	readers []io.ReadSeeker
	pos     *pos
	posIdx  map[io.ReadSeeker]int
}

func (r *multiReadSeeker) Seek(offset int64, whence int) (int64, error) {
	var tmpOffset int64
	switch whence {
	case os.SEEK_SET:
		for i, rdr := range r.readers {
			// get size of the current reader
			s, err := rdr.Seek(0, os.SEEK_END)
			if err != nil {
				return -1, err
			}

			if offset > tmpOffset+s {
				if i == len(r.readers)-1 {
					rdrOffset := s + (offset - tmpOffset)
					if _, err := rdr.Seek(rdrOffset, os.SEEK_SET); err != nil {
						return -1, err
					}
					r.pos = &pos{i, rdrOffset}
					return offset, nil
				}

				tmpOffset += s
				continue
			}

			rdrOffset := offset - tmpOffset
			idx := i

			rdr.Seek(rdrOffset, os.SEEK_SET)
			// make sure all following readers are at 0
			for _, rdr := range r.readers[i+1:] {
				rdr.Seek(0, os.SEEK_SET)
			}

			if rdrOffset == s && i != len(r.readers)-1 {
				idx += 1
				rdrOffset = 0
			}
			r.pos = &pos{idx, rdrOffset}
			return offset, nil
		}
	case os.SEEK_END:
		for _, rdr := range r.readers {
			s, err := rdr.Seek(0, os.SEEK_END)
			if err != nil {
				return -1, err
			}
			tmpOffset += s
		}
		r.Seek(tmpOffset+offset, os.SEEK_SET)
		return tmpOffset + offset, nil
	case os.SEEK_CUR:
		if r.pos == nil {
			return r.Seek(offset, os.SEEK_SET)
		}
		// Just return the current offset
		if offset == 0 {
			return r.getCurOffset()
		}

		curOffset, err := r.getCurOffset()
		if err != nil {
			return -1, err
		}
		rdr, rdrOffset, err := r.getReaderForOffset(curOffset + offset)
		if err != nil {
			return -1, err
		}

		r.pos = &pos{r.posIdx[rdr], rdrOffset}
		return curOffset + offset, nil
	default:
		return -1, fmt.Errorf("Invalid whence: %d", whence)
	}

	return -1, fmt.Errorf("Error seeking for whence: %d, offset: %d", whence, offset)
}

func (r *multiReadSeeker) getReaderForOffset(offset int64) (io.ReadSeeker, int64, error) {
	var rdr io.ReadSeeker
	var rdrOffset int64

	for i, rdr := range r.readers {
		offsetTo, err := r.getOffsetToReader(rdr)
		if err != nil {
			return nil, -1, err
		}
		if offsetTo > offset {
			rdr = r.readers[i-1]
			rdrOffset = offsetTo - offset
			break
		}

		if rdr == r.readers[len(r.readers)-1] {
			rdrOffset = offsetTo + offset
			break
		}
	}

	return rdr, rdrOffset, nil
}

func (r *multiReadSeeker) getCurOffset() (int64, error) {
	var totalSize int64
	for _, rdr := range r.readers[:r.pos.idx+1] {
		if r.posIdx[rdr] == r.pos.idx {
			totalSize += r.pos.offset
			break
		}

		size, err := getReadSeekerSize(rdr)
		if err != nil {
			return -1, fmt.Errorf("error getting seeker size: %v", err)
		}
		totalSize += size
	}
	return totalSize, nil
}

func (r *multiReadSeeker) getOffsetToReader(rdr io.ReadSeeker) (int64, error) {
	var offset int64
	for _, r := range r.readers {
		if r == rdr {
			break
		}

		size, err := getReadSeekerSize(rdr)
		if err != nil {
			return -1, err
		}
		offset += size
	}
	return offset, nil
}

func (r *multiReadSeeker) Read(b []byte) (int, error) {
	if r.pos == nil {
		r.pos = &pos{0, 0}
	}

	bCap := int64(cap(b))
	buf := bytes.NewBuffer(nil)
	var rdr io.ReadSeeker

	for _, rdr = range r.readers[r.pos.idx:] {
		readBytes, err := io.CopyN(buf, rdr, bCap)
		if err != nil && err != io.EOF {
			return -1, err
		}
		bCap -= readBytes

		if bCap == 0 {
			break
		}
	}

	rdrPos, err := rdr.Seek(0, os.SEEK_CUR)
	if err != nil {
		return -1, err
	}
	r.pos = &pos{r.posIdx[rdr], rdrPos}
	return buf.Read(b)
}

func getReadSeekerSize(rdr io.ReadSeeker) (int64, error) {
	// save the current position
	pos, err := rdr.Seek(0, os.SEEK_CUR)
	if err != nil {
		return -1, err
	}

	// get the size
	size, err := rdr.Seek(0, os.SEEK_END)
	if err != nil {
		return -1, err
	}

	// reset the position
	if _, err := rdr.Seek(pos, os.SEEK_SET); err != nil {
		return -1, err
	}
	return size, nil
}

// MultiReadSeeker returns a ReadSeeker that's the logical concatenation of the provided
// input readseekers. After calling this method the initial position is set to the
// beginning of the first ReadSeeker. At the end of a ReadSeeker, Read always advances
// to the beginning of the next ReadSeeker and returns EOF at the end of the last ReadSeeker.
// Seek can be used over the sum of lengths of all readseekers.
//
// When a MultiReadSeeker is used, no Read and Seek operations should be made on
// its ReadSeeker components. Also, users should make no assumption on the state
// of individual readseekers while the MultiReadSeeker is used.
func MultiReadSeeker(readers ...io.ReadSeeker) io.ReadSeeker {
	if len(readers) == 1 {
		return readers[0]
	}
	idx := make(map[io.ReadSeeker]int)
	for i, rdr := range readers {
		idx[rdr] = i
	}
	return &multiReadSeeker{
		readers: readers,
		posIdx:  idx,
	}
}
