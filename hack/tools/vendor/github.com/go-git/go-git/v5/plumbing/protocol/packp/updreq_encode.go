package packp

import (
	"fmt"
	"io"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/format/pktline"
	"github.com/go-git/go-git/v5/plumbing/protocol/packp/capability"
)

var (
	zeroHashString = plumbing.ZeroHash.String()
)

// Encode writes the ReferenceUpdateRequest encoding to the stream.
func (req *ReferenceUpdateRequest) Encode(w io.Writer) error {
	if err := req.validate(); err != nil {
		return err
	}

	e := pktline.NewEncoder(w)

	if err := req.encodeShallow(e, req.Shallow); err != nil {
		return err
	}

	if err := req.encodeCommands(e, req.Commands, req.Capabilities); err != nil {
		return err
	}

	if req.Packfile != nil {
		if _, err := io.Copy(w, req.Packfile); err != nil {
			return err
		}

		return req.Packfile.Close()
	}

	return nil
}

func (req *ReferenceUpdateRequest) encodeShallow(e *pktline.Encoder,
	h *plumbing.Hash) error {

	if h == nil {
		return nil
	}

	objId := []byte(h.String())
	return e.Encodef("%s%s", shallow, objId)
}

func (req *ReferenceUpdateRequest) encodeCommands(e *pktline.Encoder,
	cmds []*Command, cap *capability.List) error {

	if err := e.Encodef("%s\x00%s",
		formatCommand(cmds[0]), cap.String()); err != nil {
		return err
	}

	for _, cmd := range cmds[1:] {
		if err := e.Encodef(formatCommand(cmd)); err != nil {
			return err
		}
	}

	return e.Flush()
}

func formatCommand(cmd *Command) string {
	o := cmd.Old.String()
	n := cmd.New.String()
	return fmt.Sprintf("%s %s %s", o, n, cmd.Name)
}
