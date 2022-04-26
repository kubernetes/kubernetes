package printer

import (
	"fmt"
	"io"
	"sort"

	"github.com/golangci/dupl/syntax"
)

type text struct {
	cnt int
	w   io.Writer
	ReadFile
}

func NewText(w io.Writer, fread ReadFile) Printer {
	return &text{w: w, ReadFile: fread}
}

func (p *text) PrintHeader() error { return nil }

func (p *text) PrintClones(dups [][]*syntax.Node) error {
	p.cnt++
	fmt.Fprintf(p.w, "found %d clones:\n", len(dups))
	clones, err := prepareClonesInfo(p.ReadFile, dups)
	if err != nil {
		return err
	}
	sort.Sort(byNameAndLine(clones))
	for _, cl := range clones {
		fmt.Fprintf(p.w, "  %s:%d,%d\n", cl.filename, cl.lineStart, cl.lineEnd)
	}
	return nil
}

func (p *text) PrintFooter() error {
	_, err := fmt.Fprintf(p.w, "\nFound total %d clone groups.\n", p.cnt)
	return err
}

func prepareClonesInfo(fread ReadFile, dups [][]*syntax.Node) ([]clone, error) {
	clones := make([]clone, len(dups))
	for i, dup := range dups {
		cnt := len(dup)
		if cnt == 0 {
			panic("zero length dup")
		}
		nstart := dup[0]
		nend := dup[cnt-1]

		file, err := fread(nstart.Filename)
		if err != nil {
			return nil, err
		}

		cl := clone{filename: nstart.Filename}
		cl.lineStart, cl.lineEnd = blockLines(file, nstart.Pos, nend.End)
		clones[i] = cl
	}
	return clones, nil
}

func blockLines(file []byte, from, to int) (int, int) {
	line := 1
	lineStart, lineEnd := 0, 0
	for offset, b := range file {
		if b == '\n' {
			line++
		}
		if offset == from {
			lineStart = line
		}
		if offset == to-1 {
			lineEnd = line
			break
		}
	}
	return lineStart, lineEnd
}

type clone struct {
	filename  string
	lineStart int
	lineEnd   int
	fragment  []byte
}

type byNameAndLine []clone

func (c byNameAndLine) Len() int { return len(c) }

func (c byNameAndLine) Swap(i, j int) { c[i], c[j] = c[j], c[i] }

func (c byNameAndLine) Less(i, j int) bool {
	if c[i].filename == c[j].filename {
		return c[i].lineStart < c[j].lineStart
	}
	return c[i].filename < c[j].filename
}
