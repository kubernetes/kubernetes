package tarsum

import "sort"

// FileInfoSumInterface provides an interface for accessing file checksum
// information within a tar file. This info is accessed through interface
// so the actual name and sum cannot be melded with.
type FileInfoSumInterface interface {
	// File name
	Name() string
	// Checksum of this particular file and its headers
	Sum() string
	// Position of file in the tar
	Pos() int64
}

type fileInfoSum struct {
	name string
	sum  string
	pos  int64
}

func (fis fileInfoSum) Name() string {
	return fis.name
}
func (fis fileInfoSum) Sum() string {
	return fis.sum
}
func (fis fileInfoSum) Pos() int64 {
	return fis.pos
}

// FileInfoSums provides a list of FileInfoSumInterfaces.
type FileInfoSums []FileInfoSumInterface

// GetFile returns the first FileInfoSumInterface with a matching name.
func (fis FileInfoSums) GetFile(name string) FileInfoSumInterface {
	for i := range fis {
		if fis[i].Name() == name {
			return fis[i]
		}
	}
	return nil
}

// GetAllFile returns a FileInfoSums with all matching names.
func (fis FileInfoSums) GetAllFile(name string) FileInfoSums {
	f := FileInfoSums{}
	for i := range fis {
		if fis[i].Name() == name {
			f = append(f, fis[i])
		}
	}
	return f
}

// GetDuplicatePaths returns a FileInfoSums with all duplicated paths.
func (fis FileInfoSums) GetDuplicatePaths() (dups FileInfoSums) {
	seen := make(map[string]int, len(fis)) // allocate earl. no need to grow this map.
	for i := range fis {
		f := fis[i]
		if _, ok := seen[f.Name()]; ok {
			dups = append(dups, f)
		} else {
			seen[f.Name()] = 0
		}
	}
	return dups
}

// Len returns the size of the FileInfoSums.
func (fis FileInfoSums) Len() int { return len(fis) }

// Swap swaps two FileInfoSum values if a FileInfoSums list.
func (fis FileInfoSums) Swap(i, j int) { fis[i], fis[j] = fis[j], fis[i] }

// SortByPos sorts FileInfoSums content by position.
func (fis FileInfoSums) SortByPos() {
	sort.Sort(byPos{fis})
}

// SortByNames sorts FileInfoSums content by name.
func (fis FileInfoSums) SortByNames() {
	sort.Sort(byName{fis})
}

// SortBySums sorts FileInfoSums content by sums.
func (fis FileInfoSums) SortBySums() {
	dups := fis.GetDuplicatePaths()
	if len(dups) > 0 {
		sort.Sort(bySum{fis, dups})
	} else {
		sort.Sort(bySum{fis, nil})
	}
}

// byName is a sort.Sort helper for sorting by file names.
// If names are the same, order them by their appearance in the tar archive
type byName struct{ FileInfoSums }

func (bn byName) Less(i, j int) bool {
	if bn.FileInfoSums[i].Name() == bn.FileInfoSums[j].Name() {
		return bn.FileInfoSums[i].Pos() < bn.FileInfoSums[j].Pos()
	}
	return bn.FileInfoSums[i].Name() < bn.FileInfoSums[j].Name()
}

// bySum is a sort.Sort helper for sorting by the sums of all the fileinfos in the tar archive
type bySum struct {
	FileInfoSums
	dups FileInfoSums
}

func (bs bySum) Less(i, j int) bool {
	if bs.dups != nil && bs.FileInfoSums[i].Name() == bs.FileInfoSums[j].Name() {
		return bs.FileInfoSums[i].Pos() < bs.FileInfoSums[j].Pos()
	}
	return bs.FileInfoSums[i].Sum() < bs.FileInfoSums[j].Sum()
}

// byPos is a sort.Sort helper for sorting by the sums of all the fileinfos by their original order
type byPos struct{ FileInfoSums }

func (bp byPos) Less(i, j int) bool {
	return bp.FileInfoSums[i].Pos() < bp.FileInfoSums[j].Pos()
}
