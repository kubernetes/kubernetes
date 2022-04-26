package object

import (
	"errors"
	"io"
	"sort"
	"strings"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/filemode"
	"github.com/go-git/go-git/v5/utils/ioutil"
	"github.com/go-git/go-git/v5/utils/merkletrie"
)

// DetectRenames detects the renames in the given changes on two trees with
// the given options. It will return the given changes grouping additions and
// deletions into modifications when possible.
// If options is nil, the default diff tree options will be used.
func DetectRenames(
	changes Changes,
	opts *DiffTreeOptions,
) (Changes, error) {
	if opts == nil {
		opts = DefaultDiffTreeOptions
	}

	detector := &renameDetector{
		renameScore: int(opts.RenameScore),
		renameLimit: int(opts.RenameLimit),
		onlyExact:   opts.OnlyExactRenames,
	}

	for _, c := range changes {
		action, err := c.Action()
		if err != nil {
			return nil, err
		}

		switch action {
		case merkletrie.Insert:
			detector.added = append(detector.added, c)
		case merkletrie.Delete:
			detector.deleted = append(detector.deleted, c)
		default:
			detector.modified = append(detector.modified, c)
		}
	}

	return detector.detect()
}

// renameDetector will detect and resolve renames in a set of changes.
// see: https://github.com/eclipse/jgit/blob/master/org.eclipse.jgit/src/org/eclipse/jgit/diff/RenameDetector.java
type renameDetector struct {
	added    []*Change
	deleted  []*Change
	modified []*Change

	renameScore int
	renameLimit int
	onlyExact   bool
}

// detectExactRenames detects matches files that were deleted with files that
// were added where the hash is the same on both. If there are multiple targets
// the one with the most similar path will be chosen as the rename and the
// rest as either deletions or additions.
func (d *renameDetector) detectExactRenames() {
	added := groupChangesByHash(d.added)
	deletes := groupChangesByHash(d.deleted)
	var uniqueAdds []*Change
	var nonUniqueAdds [][]*Change
	var addedLeft []*Change

	for _, cs := range added {
		if len(cs) == 1 {
			uniqueAdds = append(uniqueAdds, cs[0])
		} else {
			nonUniqueAdds = append(nonUniqueAdds, cs)
		}
	}

	for _, c := range uniqueAdds {
		hash := changeHash(c)
		deleted := deletes[hash]

		if len(deleted) == 1 {
			if sameMode(c, deleted[0]) {
				d.modified = append(d.modified, &Change{From: deleted[0].From, To: c.To})
				delete(deletes, hash)
			} else {
				addedLeft = append(addedLeft, c)
			}
		} else if len(deleted) > 1 {
			bestMatch := bestNameMatch(c, deleted)
			if bestMatch != nil && sameMode(c, bestMatch) {
				d.modified = append(d.modified, &Change{From: bestMatch.From, To: c.To})
				delete(deletes, hash)

				var newDeletes = make([]*Change, 0, len(deleted)-1)
				for _, d := range deleted {
					if d != bestMatch {
						newDeletes = append(newDeletes, d)
					}
				}
				deletes[hash] = newDeletes
			}
		} else {
			addedLeft = append(addedLeft, c)
		}
	}

	for _, added := range nonUniqueAdds {
		hash := changeHash(added[0])
		deleted := deletes[hash]

		if len(deleted) == 1 {
			deleted := deleted[0]
			bestMatch := bestNameMatch(deleted, added)
			if bestMatch != nil && sameMode(deleted, bestMatch) {
				d.modified = append(d.modified, &Change{From: deleted.From, To: bestMatch.To})
				delete(deletes, hash)

				for _, c := range added {
					if c != bestMatch {
						addedLeft = append(addedLeft, c)
					}
				}
			} else {
				addedLeft = append(addedLeft, added...)
			}
		} else if len(deleted) > 1 {
			maxSize := len(deleted) * len(added)
			if d.renameLimit > 0 && d.renameLimit < maxSize {
				maxSize = d.renameLimit
			}

			matrix := make(similarityMatrix, 0, maxSize)

			for delIdx, del := range deleted {
				deletedName := changeName(del)

				for addIdx, add := range added {
					addedName := changeName(add)

					score := nameSimilarityScore(addedName, deletedName)
					matrix = append(matrix, similarityPair{added: addIdx, deleted: delIdx, score: score})

					if len(matrix) >= maxSize {
						break
					}
				}

				if len(matrix) >= maxSize {
					break
				}
			}

			sort.Stable(matrix)

			usedAdds := make(map[*Change]struct{})
			usedDeletes := make(map[*Change]struct{})
			for i := len(matrix) - 1; i >= 0; i-- {
				del := deleted[matrix[i].deleted]
				add := added[matrix[i].added]

				if add == nil || del == nil {
					// it was already matched
					continue
				}

				usedAdds[add] = struct{}{}
				usedDeletes[del] = struct{}{}
				d.modified = append(d.modified, &Change{From: del.From, To: add.To})
				added[matrix[i].added] = nil
				deleted[matrix[i].deleted] = nil
			}

			for _, c := range added {
				if _, ok := usedAdds[c]; !ok && c != nil {
					addedLeft = append(addedLeft, c)
				}
			}

			var newDeletes = make([]*Change, 0, len(deleted)-len(usedDeletes))
			for _, c := range deleted {
				if _, ok := usedDeletes[c]; !ok && c != nil {
					newDeletes = append(newDeletes, c)
				}
			}
			deletes[hash] = newDeletes
		} else {
			addedLeft = append(addedLeft, added...)
		}
	}

	d.added = addedLeft
	d.deleted = nil
	for _, dels := range deletes {
		d.deleted = append(d.deleted, dels...)
	}
}

// detectContentRenames detects renames based on the similarity of the content
// in the files by building a matrix of pairs between sources and destinations
// and matching by the highest score.
// see: https://github.com/eclipse/jgit/blob/master/org.eclipse.jgit/src/org/eclipse/jgit/diff/SimilarityRenameDetector.java
func (d *renameDetector) detectContentRenames() error {
	cnt := max(len(d.added), len(d.deleted))
	if d.renameLimit > 0 && cnt > d.renameLimit {
		return nil
	}

	srcs, dsts := d.deleted, d.added
	matrix, err := buildSimilarityMatrix(srcs, dsts, d.renameScore)
	if err != nil {
		return err
	}
	renames := make([]*Change, 0, min(len(matrix), len(dsts)))

	// Match rename pairs on a first come, first serve basis until
	// we have looked at everything that is above the minimum score.
	for i := len(matrix) - 1; i >= 0; i-- {
		pair := matrix[i]
		src := srcs[pair.deleted]
		dst := dsts[pair.added]

		if dst == nil || src == nil {
			// It was already matched before
			continue
		}

		renames = append(renames, &Change{From: src.From, To: dst.To})

		// Claim destination and source as matched
		dsts[pair.added] = nil
		srcs[pair.deleted] = nil
	}

	d.modified = append(d.modified, renames...)
	d.added = compactChanges(dsts)
	d.deleted = compactChanges(srcs)

	return nil
}

func (d *renameDetector) detect() (Changes, error) {
	if len(d.added) > 0 && len(d.deleted) > 0 {
		d.detectExactRenames()

		if !d.onlyExact {
			if err := d.detectContentRenames(); err != nil {
				return nil, err
			}
		}
	}

	result := make(Changes, 0, len(d.added)+len(d.deleted)+len(d.modified))
	result = append(result, d.added...)
	result = append(result, d.deleted...)
	result = append(result, d.modified...)

	sort.Stable(result)

	return result, nil
}

func bestNameMatch(change *Change, changes []*Change) *Change {
	var best *Change
	var bestScore int

	cname := changeName(change)

	for _, c := range changes {
		score := nameSimilarityScore(cname, changeName(c))
		if score > bestScore {
			bestScore = score
			best = c
		}
	}

	return best
}

func nameSimilarityScore(a, b string) int {
	aDirLen := strings.LastIndexByte(a, '/') + 1
	bDirLen := strings.LastIndexByte(b, '/') + 1

	dirMin := min(aDirLen, bDirLen)
	dirMax := max(aDirLen, bDirLen)

	var dirScoreLtr, dirScoreRtl int
	if dirMax == 0 {
		dirScoreLtr = 100
		dirScoreRtl = 100
	} else {
		var dirSim int

		for ; dirSim < dirMin; dirSim++ {
			if a[dirSim] != b[dirSim] {
				break
			}
		}

		dirScoreLtr = dirSim * 100 / dirMax

		if dirScoreLtr == 100 {
			dirScoreRtl = 100
		} else {
			for dirSim = 0; dirSim < dirMin; dirSim++ {
				if a[aDirLen-1-dirSim] != b[bDirLen-1-dirSim] {
					break
				}
			}
			dirScoreRtl = dirSim * 100 / dirMax
		}
	}

	fileMin := min(len(a)-aDirLen, len(b)-bDirLen)
	fileMax := max(len(a)-aDirLen, len(b)-bDirLen)

	fileSim := 0
	for ; fileSim < fileMin; fileSim++ {
		if a[len(a)-1-fileSim] != b[len(b)-1-fileSim] {
			break
		}
	}
	fileScore := fileSim * 100 / fileMax

	return (((dirScoreLtr + dirScoreRtl) * 25) + (fileScore * 50)) / 100
}

func changeName(c *Change) string {
	if c.To != empty {
		return c.To.Name
	}
	return c.From.Name
}

func changeHash(c *Change) plumbing.Hash {
	if c.To != empty {
		return c.To.TreeEntry.Hash
	}

	return c.From.TreeEntry.Hash
}

func changeMode(c *Change) filemode.FileMode {
	if c.To != empty {
		return c.To.TreeEntry.Mode
	}

	return c.From.TreeEntry.Mode
}

func sameMode(a, b *Change) bool {
	return changeMode(a) == changeMode(b)
}

func groupChangesByHash(changes []*Change) map[plumbing.Hash][]*Change {
	var result = make(map[plumbing.Hash][]*Change)
	for _, c := range changes {
		hash := changeHash(c)
		result[hash] = append(result[hash], c)
	}
	return result
}

type similarityMatrix []similarityPair

func (m similarityMatrix) Len() int      { return len(m) }
func (m similarityMatrix) Swap(i, j int) { m[i], m[j] = m[j], m[i] }
func (m similarityMatrix) Less(i, j int) bool {
	if m[i].score == m[j].score {
		if m[i].added == m[j].added {
			return m[i].deleted < m[j].deleted
		}
		return m[i].added < m[j].added
	}
	return m[i].score < m[j].score
}

type similarityPair struct {
	// index of the added file
	added int
	// index of the deleted file
	deleted int
	// similarity score
	score int
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func buildSimilarityMatrix(srcs, dsts []*Change, renameScore int) (similarityMatrix, error) {
	// Allocate for the worst-case scenario where every pair has a score
	// that we need to consider. We might not need that many.
	matrix := make(similarityMatrix, 0, len(srcs)*len(dsts))
	srcSizes := make([]int64, len(srcs))
	dstSizes := make([]int64, len(dsts))
	dstTooLarge := make(map[int]bool)

	// Consider each pair of files, if the score is above the minimum
	// threshold we need to record that scoring in the matrix so we can
	// later find the best matches.
outerLoop:
	for srcIdx, src := range srcs {
		if changeMode(src) != filemode.Regular {
			continue
		}

		// Declare the from file and the similarity index here to be able to
		// reuse it inside the inner loop. The reason to not initialize them
		// here is so we can skip the initialization in case they happen to
		// not be needed later. They will be initialized inside the inner
		// loop if and only if they're needed and reused in subsequent passes.
		var from *File
		var s *similarityIndex
		var err error
		for dstIdx, dst := range dsts {
			if changeMode(dst) != filemode.Regular {
				continue
			}

			if dstTooLarge[dstIdx] {
				continue
			}

			var to *File
			srcSize := srcSizes[srcIdx]
			if srcSize == 0 {
				from, _, err = src.Files()
				if err != nil {
					return nil, err
				}
				srcSize = from.Size + 1
				srcSizes[srcIdx] = srcSize
			}

			dstSize := dstSizes[dstIdx]
			if dstSize == 0 {
				_, to, err = dst.Files()
				if err != nil {
					return nil, err
				}
				dstSize = to.Size + 1
				dstSizes[dstIdx] = dstSize
			}

			min, max := srcSize, dstSize
			if dstSize < srcSize {
				min = dstSize
				max = srcSize
			}

			if int(min*100/max) < renameScore {
				// File sizes are too different to be a match
				continue
			}

			if s == nil {
				s, err = fileSimilarityIndex(from)
				if err != nil {
					if err == errIndexFull {
						continue outerLoop
					}
					return nil, err
				}
			}

			if to == nil {
				_, to, err = dst.Files()
				if err != nil {
					return nil, err
				}
			}

			di, err := fileSimilarityIndex(to)
			if err != nil {
				if err == errIndexFull {
					dstTooLarge[dstIdx] = true
				}

				return nil, err
			}

			contentScore := s.score(di, 10000)
			// The name score returns a value between 0 and 100, so we need to
			// convert it to the same range as the content score.
			nameScore := nameSimilarityScore(src.From.Name, dst.To.Name) * 100
			score := (contentScore*99 + nameScore*1) / 10000

			if score < renameScore {
				continue
			}

			matrix = append(matrix, similarityPair{added: dstIdx, deleted: srcIdx, score: score})
		}
	}

	sort.Stable(matrix)

	return matrix, nil
}

func compactChanges(changes []*Change) []*Change {
	var result []*Change
	for _, c := range changes {
		if c != nil {
			result = append(result, c)
		}
	}
	return result
}

const (
	keyShift      = 32
	maxCountValue = (1 << keyShift) - 1
)

var errIndexFull = errors.New("index is full")

// similarityIndex is an index structure of lines/blocks in one file.
// This structure can be used to compute an approximation of the similarity
// between two files.
// To save space in memory, this index uses a space efficient encoding which
// will not exceed 1MiB per instance. The index starts out at a smaller size
// (closer to 2KiB), but may grow as more distinct blocks within the scanned
// file are discovered.
// see: https://github.com/eclipse/jgit/blob/master/org.eclipse.jgit/src/org/eclipse/jgit/diff/SimilarityIndex.java
type similarityIndex struct {
	hashed uint64
	// number of non-zero entries in hashes
	numHashes int
	growAt    int
	hashes    []keyCountPair
	hashBits  int
}

func fileSimilarityIndex(f *File) (*similarityIndex, error) {
	idx := newSimilarityIndex()
	if err := idx.hash(f); err != nil {
		return nil, err
	}

	sort.Stable(keyCountPairs(idx.hashes))

	return idx, nil
}

func newSimilarityIndex() *similarityIndex {
	return &similarityIndex{
		hashBits: 8,
		hashes:   make([]keyCountPair, 1<<8),
		growAt:   shouldGrowAt(8),
	}
}

func (i *similarityIndex) hash(f *File) error {
	isBin, err := f.IsBinary()
	if err != nil {
		return err
	}

	r, err := f.Reader()
	if err != nil {
		return err
	}

	defer ioutil.CheckClose(r, &err)

	return i.hashContent(r, f.Size, isBin)
}

func (i *similarityIndex) hashContent(r io.Reader, size int64, isBin bool) error {
	var buf = make([]byte, 4096)
	var ptr, cnt int
	remaining := size

	for 0 < remaining {
		hash := 5381
		var blockHashedCnt uint64

		// Hash one line or block, whatever happens first
		n := int64(0)
		for {
			if ptr == cnt {
				ptr = 0
				var err error
				cnt, err = io.ReadFull(r, buf)
				if err != nil && err != io.ErrUnexpectedEOF {
					return err
				}

				if cnt == 0 {
					return io.EOF
				}
			}
			n++
			c := buf[ptr] & 0xff
			ptr++

			// Ignore CR in CRLF sequence if it's text
			if !isBin && c == '\r' && ptr < cnt && buf[ptr] == '\n' {
				continue
			}
			blockHashedCnt++

			if c == '\n' {
				break
			}

			hash = (hash << 5) + hash + int(c)

			if n >= 64 || n >= remaining {
				break
			}
		}
		i.hashed += blockHashedCnt
		if err := i.add(hash, blockHashedCnt); err != nil {
			return err
		}
		remaining -= n
	}

	return nil
}

// score computes the similarity score between this index and another one.
// A region of a file is defined as a line in a text file or a fixed-size
// block in a binary file. To prepare an index, each region in the file is
// hashed; the values and counts of hashes are retained in a sorted table.
// Define the similarity fraction F as the count of matching regions between
// the two files divided between the maximum count of regions in either file.
// The similarity score is F multiplied by the maxScore constant, yielding a
// range [0, maxScore]. It is defined as maxScore for the degenerate case of
// two empty files.
// The similarity score is symmetrical; i.e. a.score(b) == b.score(a).
func (i *similarityIndex) score(other *similarityIndex, maxScore int) int {
	var maxHashed = i.hashed
	if maxHashed < other.hashed {
		maxHashed = other.hashed
	}
	if maxHashed == 0 {
		return maxScore
	}

	return int(i.common(other) * uint64(maxScore) / maxHashed)
}

func (i *similarityIndex) common(dst *similarityIndex) uint64 {
	srcIdx, dstIdx := 0, 0
	if i.numHashes == 0 || dst.numHashes == 0 {
		return 0
	}

	var common uint64
	srcKey, dstKey := i.hashes[srcIdx].key(), dst.hashes[dstIdx].key()

	for {
		if srcKey == dstKey {
			srcCnt, dstCnt := i.hashes[srcIdx].count(), dst.hashes[dstIdx].count()
			if srcCnt < dstCnt {
				common += srcCnt
			} else {
				common += dstCnt
			}

			srcIdx++
			if srcIdx == len(i.hashes) {
				break
			}
			srcKey = i.hashes[srcIdx].key()

			dstIdx++
			if dstIdx == len(dst.hashes) {
				break
			}
			dstKey = dst.hashes[dstIdx].key()
		} else if srcKey < dstKey {
			// Region of src that is not in dst
			srcIdx++
			if srcIdx == len(i.hashes) {
				break
			}
			srcKey = i.hashes[srcIdx].key()
		} else {
			// Region of dst that is not in src
			dstIdx++
			if dstIdx == len(dst.hashes) {
				break
			}
			dstKey = dst.hashes[dstIdx].key()
		}
	}

	return common
}

func (i *similarityIndex) add(key int, cnt uint64) error {
	key = int(uint32(key) * 0x9e370001 >> 1)

	j := i.slot(key)
	for {
		v := i.hashes[j]
		if v == 0 {
			// It's an empty slot, so we can store it here.
			if i.growAt <= i.numHashes {
				if err := i.grow(); err != nil {
					return err
				}
				j = i.slot(key)
				continue
			}

			var err error
			i.hashes[j], err = newKeyCountPair(key, cnt)
			if err != nil {
				return err
			}
			i.numHashes++
			return nil
		} else if v.key() == key {
			// It's the same key, so increment the counter.
			var err error
			i.hashes[j], err = newKeyCountPair(key, v.count()+cnt)
			if err != nil {
				return err
			}
			return nil
		} else if j+1 >= len(i.hashes) {
			j = 0
		} else {
			j++
		}
	}
}

type keyCountPair uint64

func newKeyCountPair(key int, cnt uint64) (keyCountPair, error) {
	if cnt > maxCountValue {
		return 0, errIndexFull
	}

	return keyCountPair((uint64(key) << keyShift) | cnt), nil
}

func (p keyCountPair) key() int {
	return int(p >> keyShift)
}

func (p keyCountPair) count() uint64 {
	return uint64(p) & maxCountValue
}

func (i *similarityIndex) slot(key int) int {
	// We use 31 - hashBits because the upper bit was already forced
	// to be 0 and we want the remaining high bits to be used as the
	// table slot.
	return int(uint32(key) >> uint(31-i.hashBits))
}

func shouldGrowAt(hashBits int) int {
	return (1 << uint(hashBits)) * (hashBits - 3) / hashBits
}

func (i *similarityIndex) grow() error {
	if i.hashBits == 30 {
		return errIndexFull
	}

	old := i.hashes

	i.hashBits++
	i.growAt = shouldGrowAt(i.hashBits)

	// TODO(erizocosmico): find a way to check if it will OOM and return
	// errIndexFull instead.
	i.hashes = make([]keyCountPair, 1<<uint(i.hashBits))

	for _, v := range old {
		if v != 0 {
			j := i.slot(v.key())
			for i.hashes[j] != 0 {
				j++
				if j >= len(i.hashes) {
					j = 0
				}
			}
			i.hashes[j] = v
		}
	}

	return nil
}

type keyCountPairs []keyCountPair

func (p keyCountPairs) Len() int           { return len(p) }
func (p keyCountPairs) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p keyCountPairs) Less(i, j int) bool { return p[i] < p[j] }
