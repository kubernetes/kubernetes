package idxfile

import (
	"bufio"
	"bytes"
	"errors"
	"io"

	"github.com/go-git/go-git/v5/utils/binary"
)

var (
	// ErrUnsupportedVersion is returned by Decode when the idx file version
	// is not supported.
	ErrUnsupportedVersion = errors.New("Unsupported version")
	// ErrMalformedIdxFile is returned by Decode when the idx file is corrupted.
	ErrMalformedIdxFile = errors.New("Malformed IDX file")
)

const (
	fanout         = 256
	objectIDLength = 20
)

// Decoder reads and decodes idx files from an input stream.
type Decoder struct {
	*bufio.Reader
}

// NewDecoder builds a new idx stream decoder, that reads from r.
func NewDecoder(r io.Reader) *Decoder {
	return &Decoder{bufio.NewReader(r)}
}

// Decode reads from the stream and decode the content into the MemoryIndex struct.
func (d *Decoder) Decode(idx *MemoryIndex) error {
	if err := validateHeader(d); err != nil {
		return err
	}

	flow := []func(*MemoryIndex, io.Reader) error{
		readVersion,
		readFanout,
		readObjectNames,
		readCRC32,
		readOffsets,
		readChecksums,
	}

	for _, f := range flow {
		if err := f(idx, d); err != nil {
			return err
		}
	}

	return nil
}

func validateHeader(r io.Reader) error {
	var h = make([]byte, 4)
	if _, err := io.ReadFull(r, h); err != nil {
		return err
	}

	if !bytes.Equal(h, idxHeader) {
		return ErrMalformedIdxFile
	}

	return nil
}

func readVersion(idx *MemoryIndex, r io.Reader) error {
	v, err := binary.ReadUint32(r)
	if err != nil {
		return err
	}

	if v > VersionSupported {
		return ErrUnsupportedVersion
	}

	idx.Version = v
	return nil
}

func readFanout(idx *MemoryIndex, r io.Reader) error {
	for k := 0; k < fanout; k++ {
		n, err := binary.ReadUint32(r)
		if err != nil {
			return err
		}

		idx.Fanout[k] = n
		idx.FanoutMapping[k] = noMapping
	}

	return nil
}

func readObjectNames(idx *MemoryIndex, r io.Reader) error {
	for k := 0; k < fanout; k++ {
		var buckets uint32
		if k == 0 {
			buckets = idx.Fanout[k]
		} else {
			buckets = idx.Fanout[k] - idx.Fanout[k-1]
		}

		if buckets == 0 {
			continue
		}

		idx.FanoutMapping[k] = len(idx.Names)

		nameLen := int(buckets * objectIDLength)
		bin := make([]byte, nameLen)
		if _, err := io.ReadFull(r, bin); err != nil {
			return err
		}

		idx.Names = append(idx.Names, bin)
		idx.Offset32 = append(idx.Offset32, make([]byte, buckets*4))
		idx.CRC32 = append(idx.CRC32, make([]byte, buckets*4))
	}

	return nil
}

func readCRC32(idx *MemoryIndex, r io.Reader) error {
	for k := 0; k < fanout; k++ {
		if pos := idx.FanoutMapping[k]; pos != noMapping {
			if _, err := io.ReadFull(r, idx.CRC32[pos]); err != nil {
				return err
			}
		}
	}

	return nil
}

func readOffsets(idx *MemoryIndex, r io.Reader) error {
	var o64cnt int
	for k := 0; k < fanout; k++ {
		if pos := idx.FanoutMapping[k]; pos != noMapping {
			if _, err := io.ReadFull(r, idx.Offset32[pos]); err != nil {
				return err
			}

			for p := 0; p < len(idx.Offset32[pos]); p += 4 {
				if idx.Offset32[pos][p]&(byte(1)<<7) > 0 {
					o64cnt++
				}
			}
		}
	}

	if o64cnt > 0 {
		idx.Offset64 = make([]byte, o64cnt*8)
		if _, err := io.ReadFull(r, idx.Offset64); err != nil {
			return err
		}
	}

	return nil
}

func readChecksums(idx *MemoryIndex, r io.Reader) error {
	if _, err := io.ReadFull(r, idx.PackfileChecksum[:]); err != nil {
		return err
	}

	if _, err := io.ReadFull(r, idx.IdxChecksum[:]); err != nil {
		return err
	}

	return nil
}
