package godirwalk

import "sort"

type scanner interface {
	Dirent() (*Dirent, error)
	Err() error
	Name() string
	Scan() bool
}

// sortedScanner enumerates through a directory's contents after reading the
// entire directory and sorting the entries by name. Used by walk to simplify
// its implementation.
type sortedScanner struct {
	dd []*Dirent
	de *Dirent
}

func newSortedScanner(osPathname string, scratchBuffer []byte) (*sortedScanner, error) {
	deChildren, err := ReadDirents(osPathname, scratchBuffer)
	if err != nil {
		return nil, err
	}
	sort.Sort(deChildren)
	return &sortedScanner{dd: deChildren}, nil
}

func (d *sortedScanner) Err() error {
	d.dd, d.de = nil, nil
	return nil
}

func (d *sortedScanner) Dirent() (*Dirent, error) { return d.de, nil }

func (d *sortedScanner) Name() string { return d.de.name }

func (d *sortedScanner) Scan() bool {
	if len(d.dd) > 0 {
		d.de, d.dd = d.dd[0], d.dd[1:]
		return true
	}
	return false
}
