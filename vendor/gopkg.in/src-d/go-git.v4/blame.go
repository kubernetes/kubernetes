package git

import (
	"bytes"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/object"
	"gopkg.in/src-d/go-git.v4/utils/diff"
)

// BlameResult represents the result of a Blame operation.
type BlameResult struct {
	// Path is the path of the File that we're blaming.
	Path string
	// Rev (Revision) is the hash of the specified Commit used to generate this result.
	Rev plumbing.Hash
	// Lines contains every line with its authorship.
	Lines []*Line
}

// Blame returns a BlameResult with the information about the last author of
// each line from file `path` at commit `c`.
func Blame(c *object.Commit, path string) (*BlameResult, error) {
	// The file to blame is identified by the input arguments:
	// commit and path. commit is a Commit object obtained from a Repository. Path
	// represents a path to a specific file contained into the repository.
	//
	// Blaming a file is a two step process:
	//
	// 1. Create a linear history of the commits affecting a file. We use
	// revlist.New for that.
	//
	// 2. Then build a graph with a node for every line in every file in
	// the history of the file.
	//
	// Each node is assigned a commit: Start by the nodes in the first
	// commit. Assign that commit as the creator of all its lines.
	//
	// Then jump to the nodes in the next commit, and calculate the diff
	// between the two files. Newly created lines get
	// assigned the new commit as its origin. Modified lines also get
	// this new commit. Untouched lines retain the old commit.
	//
	// All this work is done in the assignOrigin function which holds all
	// the internal relevant data in a "blame" struct, that is not
	// exported.
	//
	// TODO: ways to improve the efficiency of this function:
	// 1. Improve revlist
	// 2. Improve how to traverse the history (example a backward traversal will
	// be much more efficient)
	//
	// TODO: ways to improve the function in general:
	// 1. Add memoization between revlist and assign.
	// 2. It is using much more memory than needed, see the TODOs below.

	b := new(blame)
	b.fRev = c
	b.path = path

	// get all the file revisions
	if err := b.fillRevs(); err != nil {
		return nil, err
	}

	// calculate the line tracking graph and fill in
	// file contents in data.
	if err := b.fillGraphAndData(); err != nil {
		return nil, err
	}

	file, err := b.fRev.File(b.path)
	if err != nil {
		return nil, err
	}
	finalLines, err := file.Lines()
	if err != nil {
		return nil, err
	}

	// Each node (line) holds the commit where it was introduced or
	// last modified. To achieve that we use the FORWARD algorithm
	// described in Zimmermann, et al. "Mining Version Archives for
	// Co-changed Lines", in proceedings of the Mining Software
	// Repositories workshop, Shanghai, May 22-23, 2006.
	lines, err := newLines(finalLines, b.sliceGraph(len(b.graph)-1))
	if err != nil {
		return nil, err
	}

	return &BlameResult{
		Path:  path,
		Rev:   c.Hash,
		Lines: lines,
	}, nil
}

// Line values represent the contents and author of a line in BlamedResult values.
type Line struct {
	// Author is the email address of the last author that modified the line.
	Author string
	// Text is the original text of the line.
	Text string
}

func newLine(author, text string) *Line {
	return &Line{
		Author: author,
		Text:   text,
	}
}

func newLines(contents []string, commits []*object.Commit) ([]*Line, error) {
	if len(contents) != len(commits) {
		return nil, errors.New("contents and commits have different length")
	}
	result := make([]*Line, 0, len(contents))
	for i := range contents {
		l := newLine(commits[i].Author.Email, contents[i])
		result = append(result, l)
	}
	return result, nil
}

// this struct is internally used by the blame function to hold its
// inputs, outputs and state.
type blame struct {
	// the path of the file to blame
	path string
	// the commit of the final revision of the file to blame
	fRev *object.Commit
	// the chain of revisions affecting the the file to blame
	revs []*object.Commit
	// the contents of the file across all its revisions
	data []string
	// the graph of the lines in the file across all the revisions
	graph [][]*object.Commit
}

// calculate the history of a file "path", starting from commit "from", sorted by commit date.
func (b *blame) fillRevs() error {
	var err error

	b.revs, err = references(b.fRev, b.path)
	if err != nil {
		return err
	}
	return nil
}

// build graph of a file from its revision history
func (b *blame) fillGraphAndData() error {
	//TODO: not all commits are needed, only the current rev and the prev
	b.graph = make([][]*object.Commit, len(b.revs))
	b.data = make([]string, len(b.revs)) // file contents in all the revisions
	// for every revision of the file, starting with the first
	// one...
	for i, rev := range b.revs {
		// get the contents of the file
		file, err := rev.File(b.path)
		if err != nil {
			return nil
		}
		b.data[i], err = file.Contents()
		if err != nil {
			return err
		}
		nLines := countLines(b.data[i])
		// create a node for each line
		b.graph[i] = make([]*object.Commit, nLines)
		// assign a commit to each node
		// if this is the first revision, then the node is assigned to
		// this first commit.
		if i == 0 {
			for j := 0; j < nLines; j++ {
				b.graph[i][j] = (*object.Commit)(b.revs[i])
			}
		} else {
			// if this is not the first commit, then assign to the old
			// commit or to the new one, depending on what the diff
			// says.
			b.assignOrigin(i, i-1)
		}
	}
	return nil
}

// sliceGraph returns a slice of commits (one per line) for a particular
// revision of a file (0=first revision).
func (b *blame) sliceGraph(i int) []*object.Commit {
	fVs := b.graph[i]
	result := make([]*object.Commit, 0, len(fVs))
	for _, v := range fVs {
		c := object.Commit(*v)
		result = append(result, &c)
	}
	return result
}

// Assigns origin to vertexes in current (c) rev from data in its previous (p)
// revision
func (b *blame) assignOrigin(c, p int) {
	// assign origin based on diff info
	hunks := diff.Do(b.data[p], b.data[c])
	sl := -1 // source line
	dl := -1 // destination line
	for h := range hunks {
		hLines := countLines(hunks[h].Text)
		for hl := 0; hl < hLines; hl++ {
			switch {
			case hunks[h].Type == 0:
				sl++
				dl++
				b.graph[c][dl] = b.graph[p][sl]
			case hunks[h].Type == 1:
				dl++
				b.graph[c][dl] = (*object.Commit)(b.revs[c])
			case hunks[h].Type == -1:
				sl++
			default:
				panic("unreachable")
			}
		}
	}
}

// GoString prints the results of a Blame using git-blame's style.
func (b *blame) GoString() string {
	var buf bytes.Buffer

	file, err := b.fRev.File(b.path)
	if err != nil {
		panic("PrettyPrint: internal error in repo.Data")
	}
	contents, err := file.Contents()
	if err != nil {
		panic("PrettyPrint: internal error in repo.Data")
	}

	lines := strings.Split(contents, "\n")
	// max line number length
	mlnl := len(fmt.Sprintf("%s", strconv.Itoa(len(lines))))
	// max author length
	mal := b.maxAuthorLength()
	format := fmt.Sprintf("%%s (%%-%ds %%%dd) %%s\n",
		mal, mlnl)

	fVs := b.graph[len(b.graph)-1]
	for ln, v := range fVs {
		fmt.Fprintf(&buf, format, v.Hash.String()[:8],
			prettyPrintAuthor(fVs[ln]), ln+1, lines[ln])
	}
	return buf.String()
}

// utility function to pretty print the author.
func prettyPrintAuthor(c *object.Commit) string {
	return fmt.Sprintf("%s %s", c.Author.Name, c.Author.When.Format("2006-01-02"))
}

// utility function to calculate the number of runes needed
// to print the longest author name in the blame of a file.
func (b *blame) maxAuthorLength() int {
	memo := make(map[plumbing.Hash]struct{}, len(b.graph)-1)
	fVs := b.graph[len(b.graph)-1]
	m := 0
	for ln := range fVs {
		if _, ok := memo[fVs[ln].Hash]; ok {
			continue
		}
		memo[fVs[ln].Hash] = struct{}{}
		m = max(m, utf8.RuneCountInString(prettyPrintAuthor(fVs[ln])))
	}
	return m
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
