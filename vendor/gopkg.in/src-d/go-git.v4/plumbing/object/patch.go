package object

import (
	"bytes"
	"fmt"
	"io"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/filemode"
	fdiff "gopkg.in/src-d/go-git.v4/plumbing/format/diff"
	"gopkg.in/src-d/go-git.v4/utils/diff"

	dmp "github.com/sergi/go-diff/diffmatchpatch"
)

func getPatch(message string, changes ...*Change) (*Patch, error) {
	var filePatches []fdiff.FilePatch
	for _, c := range changes {
		fp, err := filePatch(c)
		if err != nil {
			return nil, err
		}

		filePatches = append(filePatches, fp)
	}

	return &Patch{message, filePatches}, nil
}

func filePatch(c *Change) (fdiff.FilePatch, error) {
	from, to, err := c.Files()
	if err != nil {
		return nil, err
	}
	fromContent, fIsBinary, err := fileContent(from)
	if err != nil {
		return nil, err
	}

	toContent, tIsBinary, err := fileContent(to)
	if err != nil {
		return nil, err
	}

	if fIsBinary || tIsBinary {
		return &textFilePatch{from: c.From, to: c.To}, nil
	}

	diffs := diff.Do(fromContent, toContent)

	var chunks []fdiff.Chunk
	for _, d := range diffs {
		var op fdiff.Operation
		switch d.Type {
		case dmp.DiffEqual:
			op = fdiff.Equal
		case dmp.DiffDelete:
			op = fdiff.Delete
		case dmp.DiffInsert:
			op = fdiff.Add
		}

		chunks = append(chunks, &textChunk{d.Text, op})
	}

	return &textFilePatch{
		chunks: chunks,
		from:   c.From,
		to:     c.To,
	}, nil
}

func fileContent(f *File) (content string, isBinary bool, err error) {
	if f == nil {
		return
	}

	isBinary, err = f.IsBinary()
	if err != nil || isBinary {
		return
	}

	content, err = f.Contents()

	return
}

// textPatch is an implementation of fdiff.Patch interface
type Patch struct {
	message     string
	filePatches []fdiff.FilePatch
}

func (t *Patch) FilePatches() []fdiff.FilePatch {
	return t.filePatches
}

func (t *Patch) Message() string {
	return t.message
}

func (p *Patch) Encode(w io.Writer) error {
	ue := fdiff.NewUnifiedEncoder(w, fdiff.DefaultContextLines)

	return ue.Encode(p)
}

func (p *Patch) String() string {
	buf := bytes.NewBuffer(nil)
	err := p.Encode(buf)
	if err != nil {
		return fmt.Sprintf("malformed patch: %s", err.Error())
	}

	return buf.String()
}

// changeEntryWrapper is an implementation of fdiff.File interface
type changeEntryWrapper struct {
	ce ChangeEntry
}

func (f *changeEntryWrapper) Hash() plumbing.Hash {
	if !f.ce.TreeEntry.Mode.IsFile() {
		return plumbing.ZeroHash
	}

	return f.ce.TreeEntry.Hash
}

func (f *changeEntryWrapper) Mode() filemode.FileMode {
	return f.ce.TreeEntry.Mode
}
func (f *changeEntryWrapper) Path() string {
	if !f.ce.TreeEntry.Mode.IsFile() {
		return ""
	}

	return f.ce.Name
}

func (f *changeEntryWrapper) Empty() bool {
	return !f.ce.TreeEntry.Mode.IsFile()
}

// textFilePatch is an implementation of fdiff.FilePatch interface
type textFilePatch struct {
	chunks   []fdiff.Chunk
	from, to ChangeEntry
}

func (tf *textFilePatch) Files() (from fdiff.File, to fdiff.File) {
	f := &changeEntryWrapper{tf.from}
	t := &changeEntryWrapper{tf.to}

	if !f.Empty() {
		from = f
	}

	if !t.Empty() {
		to = t
	}

	return
}

func (t *textFilePatch) IsBinary() bool {
	return len(t.chunks) == 0
}

func (t *textFilePatch) Chunks() []fdiff.Chunk {
	return t.chunks
}

// textChunk is an implementation of fdiff.Chunk interface
type textChunk struct {
	content string
	op      fdiff.Operation
}

func (t *textChunk) Content() string {
	return t.content
}

func (t *textChunk) Type() fdiff.Operation {
	return t.op
}
