package diff

import (
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"

	"github.com/go-git/go-git/v5/plumbing"
)

// DefaultContextLines is the default number of context lines.
const DefaultContextLines = 3

var (
	splitLinesRegexp = regexp.MustCompile(`[^\n]*(\n|$)`)

	operationChar = map[Operation]byte{
		Add:    '+',
		Delete: '-',
		Equal:  ' ',
	}

	operationColorKey = map[Operation]ColorKey{
		Add:    New,
		Delete: Old,
		Equal:  Context,
	}
)

// UnifiedEncoder encodes an unified diff into the provided Writer. It does not
// support similarity index for renames or sorting hash representations.
type UnifiedEncoder struct {
	io.Writer

	// contextLines is the count of unchanged lines that will appear surrounding
	// a change.
	contextLines int

	// colorConfig is the color configuration. The default is no color.
	color ColorConfig
}

// NewUnifiedEncoder returns a new UnifiedEncoder that writes to w.
func NewUnifiedEncoder(w io.Writer, contextLines int) *UnifiedEncoder {
	return &UnifiedEncoder{
		Writer:       w,
		contextLines: contextLines,
	}
}

// SetColor sets e's color configuration and returns e.
func (e *UnifiedEncoder) SetColor(colorConfig ColorConfig) *UnifiedEncoder {
	e.color = colorConfig
	return e
}

// Encode encodes patch.
func (e *UnifiedEncoder) Encode(patch Patch) error {
	sb := &strings.Builder{}

	if message := patch.Message(); message != "" {
		sb.WriteString(message)
		if !strings.HasSuffix(message, "\n") {
			sb.WriteByte('\n')
		}
	}

	for _, filePatch := range patch.FilePatches() {
		e.writeFilePatchHeader(sb, filePatch)
		g := newHunksGenerator(filePatch.Chunks(), e.contextLines)
		for _, hunk := range g.Generate() {
			hunk.writeTo(sb, e.color)
		}
	}

	_, err := e.Write([]byte(sb.String()))
	return err
}

func (e *UnifiedEncoder) writeFilePatchHeader(sb *strings.Builder, filePatch FilePatch) {
	from, to := filePatch.Files()
	if from == nil && to == nil {
		return
	}
	isBinary := filePatch.IsBinary()

	var lines []string
	switch {
	case from != nil && to != nil:
		hashEquals := from.Hash() == to.Hash()
		lines = append(lines,
			fmt.Sprintf("diff --git a/%s b/%s", from.Path(), to.Path()),
		)
		if from.Mode() != to.Mode() {
			lines = append(lines,
				fmt.Sprintf("old mode %o", from.Mode()),
				fmt.Sprintf("new mode %o", to.Mode()),
			)
		}
		if from.Path() != to.Path() {
			lines = append(lines,
				fmt.Sprintf("rename from %s", from.Path()),
				fmt.Sprintf("rename to %s", to.Path()),
			)
		}
		if from.Mode() != to.Mode() && !hashEquals {
			lines = append(lines,
				fmt.Sprintf("index %s..%s", from.Hash(), to.Hash()),
			)
		} else if !hashEquals {
			lines = append(lines,
				fmt.Sprintf("index %s..%s %o", from.Hash(), to.Hash(), from.Mode()),
			)
		}
		if !hashEquals {
			lines = e.appendPathLines(lines, "a/"+from.Path(), "b/"+to.Path(), isBinary)
		}
	case from == nil:
		lines = append(lines,
			fmt.Sprintf("diff --git a/%s b/%s", to.Path(), to.Path()),
			fmt.Sprintf("new file mode %o", to.Mode()),
			fmt.Sprintf("index %s..%s", plumbing.ZeroHash, to.Hash()),
		)
		lines = e.appendPathLines(lines, "/dev/null", "b/"+to.Path(), isBinary)
	case to == nil:
		lines = append(lines,
			fmt.Sprintf("diff --git a/%s b/%s", from.Path(), from.Path()),
			fmt.Sprintf("deleted file mode %o", from.Mode()),
			fmt.Sprintf("index %s..%s", from.Hash(), plumbing.ZeroHash),
		)
		lines = e.appendPathLines(lines, "a/"+from.Path(), "/dev/null", isBinary)
	}

	sb.WriteString(e.color[Meta])
	sb.WriteString(lines[0])
	for _, line := range lines[1:] {
		sb.WriteByte('\n')
		sb.WriteString(line)
	}
	sb.WriteString(e.color.Reset(Meta))
	sb.WriteByte('\n')
}

func (e *UnifiedEncoder) appendPathLines(lines []string, fromPath, toPath string, isBinary bool) []string {
	if isBinary {
		return append(lines,
			fmt.Sprintf("Binary files %s and %s differ", fromPath, toPath),
		)
	}
	return append(lines,
		fmt.Sprintf("--- %s", fromPath),
		fmt.Sprintf("+++ %s", toPath),
	)
}

type hunksGenerator struct {
	fromLine, toLine            int
	ctxLines                    int
	chunks                      []Chunk
	current                     *hunk
	hunks                       []*hunk
	beforeContext, afterContext []string
}

func newHunksGenerator(chunks []Chunk, ctxLines int) *hunksGenerator {
	return &hunksGenerator{
		chunks:   chunks,
		ctxLines: ctxLines,
	}
}

func (g *hunksGenerator) Generate() []*hunk {
	for i, chunk := range g.chunks {
		lines := splitLines(chunk.Content())
		nLines := len(lines)

		switch chunk.Type() {
		case Equal:
			g.fromLine += nLines
			g.toLine += nLines
			g.processEqualsLines(lines, i)
		case Delete:
			if nLines != 0 {
				g.fromLine++
			}

			g.processHunk(i, chunk.Type())
			g.fromLine += nLines - 1
			g.current.AddOp(chunk.Type(), lines...)
		case Add:
			if nLines != 0 {
				g.toLine++
			}
			g.processHunk(i, chunk.Type())
			g.toLine += nLines - 1
			g.current.AddOp(chunk.Type(), lines...)
		}

		if i == len(g.chunks)-1 && g.current != nil {
			g.hunks = append(g.hunks, g.current)
		}
	}

	return g.hunks
}

func (g *hunksGenerator) processHunk(i int, op Operation) {
	if g.current != nil {
		return
	}

	var ctxPrefix string
	linesBefore := len(g.beforeContext)
	if linesBefore > g.ctxLines {
		ctxPrefix = g.beforeContext[linesBefore-g.ctxLines-1]
		g.beforeContext = g.beforeContext[linesBefore-g.ctxLines:]
		linesBefore = g.ctxLines
	}

	g.current = &hunk{ctxPrefix: strings.TrimSuffix(ctxPrefix, "\n")}
	g.current.AddOp(Equal, g.beforeContext...)

	switch op {
	case Delete:
		g.current.fromLine, g.current.toLine =
			g.addLineNumbers(g.fromLine, g.toLine, linesBefore, i, Add)
	case Add:
		g.current.toLine, g.current.fromLine =
			g.addLineNumbers(g.toLine, g.fromLine, linesBefore, i, Delete)
	}

	g.beforeContext = nil
}

// addLineNumbers obtains the line numbers in a new chunk.
func (g *hunksGenerator) addLineNumbers(la, lb int, linesBefore int, i int, op Operation) (cla, clb int) {
	cla = la - linesBefore
	// we need to search for a reference for the next diff
	switch {
	case linesBefore != 0 && g.ctxLines != 0:
		if lb > g.ctxLines {
			clb = lb - g.ctxLines + 1
		} else {
			clb = 1
		}
	case g.ctxLines == 0:
		clb = lb
	case i != len(g.chunks)-1:
		next := g.chunks[i+1]
		if next.Type() == op || next.Type() == Equal {
			// this diff will be into this chunk
			clb = lb + 1
		}
	}

	return
}

func (g *hunksGenerator) processEqualsLines(ls []string, i int) {
	if g.current == nil {
		g.beforeContext = append(g.beforeContext, ls...)
		return
	}

	g.afterContext = append(g.afterContext, ls...)
	if len(g.afterContext) <= g.ctxLines*2 && i != len(g.chunks)-1 {
		g.current.AddOp(Equal, g.afterContext...)
		g.afterContext = nil
	} else {
		ctxLines := g.ctxLines
		if ctxLines > len(g.afterContext) {
			ctxLines = len(g.afterContext)
		}
		g.current.AddOp(Equal, g.afterContext[:ctxLines]...)
		g.hunks = append(g.hunks, g.current)

		g.current = nil
		g.beforeContext = g.afterContext[ctxLines:]
		g.afterContext = nil
	}
}

func splitLines(s string) []string {
	out := splitLinesRegexp.FindAllString(s, -1)
	if out[len(out)-1] == "" {
		out = out[:len(out)-1]
	}
	return out
}

type hunk struct {
	fromLine int
	toLine   int

	fromCount int
	toCount   int

	ctxPrefix string
	ops       []*op
}

func (h *hunk) writeTo(sb *strings.Builder, color ColorConfig) {
	sb.WriteString(color[Frag])
	sb.WriteString("@@ -")

	if h.fromCount == 1 {
		sb.WriteString(strconv.Itoa(h.fromLine))
	} else {
		sb.WriteString(strconv.Itoa(h.fromLine))
		sb.WriteByte(',')
		sb.WriteString(strconv.Itoa(h.fromCount))
	}

	sb.WriteString(" +")

	if h.toCount == 1 {
		sb.WriteString(strconv.Itoa(h.toLine))
	} else {
		sb.WriteString(strconv.Itoa(h.toLine))
		sb.WriteByte(',')
		sb.WriteString(strconv.Itoa(h.toCount))
	}

	sb.WriteString(" @@")
	sb.WriteString(color.Reset(Frag))

	if h.ctxPrefix != "" {
		sb.WriteByte(' ')
		sb.WriteString(color[Func])
		sb.WriteString(h.ctxPrefix)
		sb.WriteString(color.Reset(Func))
	}

	sb.WriteByte('\n')

	for _, op := range h.ops {
		op.writeTo(sb, color)
	}
}

func (h *hunk) AddOp(t Operation, ss ...string) {
	n := len(ss)
	switch t {
	case Add:
		h.toCount += n
	case Delete:
		h.fromCount += n
	case Equal:
		h.toCount += n
		h.fromCount += n
	}

	for _, s := range ss {
		h.ops = append(h.ops, &op{s, t})
	}
}

type op struct {
	text string
	t    Operation
}

func (o *op) writeTo(sb *strings.Builder, color ColorConfig) {
	colorKey := operationColorKey[o.t]
	sb.WriteString(color[colorKey])
	sb.WriteByte(operationChar[o.t])
	if strings.HasSuffix(o.text, "\n") {
		sb.WriteString(strings.TrimSuffix(o.text, "\n"))
	} else {
		sb.WriteString(o.text + "\n\\ No newline at end of file")
	}
	sb.WriteString(color.Reset(colorKey))
	sb.WriteByte('\n')
}
