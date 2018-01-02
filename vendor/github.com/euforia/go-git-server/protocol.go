package gitserver

import (
	"fmt"
	"io"
	"log"
	"strings"

	"github.com/bargez/pktline"
	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"

	"github.com/euforia/go-git-server/packfile"
	"github.com/euforia/go-git-server/repository"
)

// GitServiceType can be either receive or upload pack
type GitServiceType string

const (
	// GitServiceRecvPack constant for receive-pack
	GitServiceRecvPack = GitServiceType("git-receive-pack")
	// GitServiceUploadPack constant for upload-pack
	GitServiceUploadPack = GitServiceType("git-upload-pack")
)

// TxRef is a transaction to update a repo reference
type TxRef struct {
	oldHash plumbing.Hash
	newHash plumbing.Hash
	ref     string
}

// Protocol implements the git pack protocol
type Protocol struct {
	w io.Writer
	r io.Reader
}

// NewProtocol instantiates a new protocol with the given reader and writer
func NewProtocol(w io.Writer, r io.Reader) *Protocol {
	return &Protocol{w: w, r: r}
}

// ListReferences writes the references in the pack protocol given the repository
// and service type
func (proto *Protocol) ListReferences(service GitServiceType, refs *repository.RepositoryReferences) {

	// Start sending info
	enc := pktline.NewEncoder(proto.w)
	enc.Encode([]byte(fmt.Sprintf("# service=%s\n", service)))
	enc.Encode(nil)

	// Repo empty so send zeros
	if (refs.Heads == nil || len(refs.Heads) == 0) && (refs.Tags == nil || len(refs.Tags) == 0) {
		b0 := append([]byte("0000000000000000000000000000000000000000"), 32)
		b0 = append(b0, nullCapabilities()...)

		enc.Encode(append(b0, 10))
		enc.Encode(nil)
		return
	}

	// Send HEAD info
	head := refs.Head

	lh := append([]byte(fmt.Sprintf("%s HEAD", head.Hash.String())), '\x00')
	lh = append(lh, capabilities()...)

	if service == GitServiceUploadPack {
		lh = append(lh, []byte(" symref=HEAD:refs/"+head.Ref)...)
	}

	enc.Encode(append(lh, 10))

	// Send refs - heads
	for href, h := range refs.Heads {
		enc.Encode([]byte(fmt.Sprintf("%s refs/heads/%s\n", h.String(), href)))
	}

	// Send refs - tags
	for tref, h := range refs.Tags {
		enc.Encode([]byte(fmt.Sprintf("%s refs/tags/%s\n", h.String(), tref)))
	}

	enc.Encode(nil)

}

// UploadPack implements the git upload pack protocol
func (proto *Protocol) UploadPack(store storer.EncodedObjectStorer) ([]byte, error) {
	wants, haves, err := parseUploadPackWantsAndHaves(proto.r)
	if err != nil {
		return nil, err
	}

	log.Printf("DBG [upload-pack] wants=%d haves=%d", len(wants), len(haves))

	enc := pktline.NewEncoder(proto.w)
	enc.Encode([]byte("NAK\n"))

	packenc := packfile.NewEncoder(proto.w, store)
	return packenc.Encode(wants...)
}

// ReceivePack implements the git receive pack protocol
func (proto *Protocol) ReceivePack(repo *repository.Repository, repostore repository.RepositoryStore, objstore storer.EncodedObjectStorer) error {

	enc := pktline.NewEncoder(proto.w)

	//var txs []TxRef
	txs, err := parseReceivePackClientRefLines(proto.r)
	if err != nil {
		enc.Encode([]byte(fmt.Sprintf("unpack %v", err)))
		return err
	}

	// Decode packfile
	packdec := packfile.NewDecoder(proto.r, objstore)
	if err = packdec.Decode(); err != nil {
		enc.Encode([]byte(fmt.Sprintf("unpack %v", err)))
		return err
	}
	enc.Encode([]byte("unpack ok\n"))

	// Update repo refs
	for _, tx := range txs {
		if er := repo.Refs.UpdateRef(tx.ref, tx.oldHash, tx.newHash); er != nil {
			log.Println("ERR [receive-pack]", er)
			continue
		}
		enc.Encode([]byte(fmt.Sprintf("ok %s\n", tx.ref)))
	}

	// Store update repo
	if err = repostore.UpdateRepo(repo); err != nil {
		log.Println("ERR [receive-pack] Failed to update repo:", err)
	}

	enc.Encode(nil)
	return err
}

func parseReceivePackClientRefLines(r io.Reader) ([]TxRef, error) {

	dec := pktline.NewDecoder(r)

	var lines [][]byte

	// Read refs from client
	if err := dec.DecodeUntilFlush(&lines); err != nil {
		//log.Printf("[receive-pack] ERR %v", e)
		return nil, err
	}

	txs := make([]TxRef, len(lines))
	for i, l := range lines {
		log.Printf("DBG [receive-pack] %s", l)

		rt, err := parseReceiveRefUpdateLine(l)
		if err != nil {
			return nil, err
		}
		txs[i] = rt

	}

	return txs, nil
}

// Parses old hash, new hash, and ref from a line
func parseReceiveRefUpdateLine(line []byte) (rt TxRef, err error) {
	s := string(line)
	arr := strings.Split(s, " ")
	if len(arr) < 3 {
		err = fmt.Errorf("invalid line: %s", line)
		return
	}

	rt = TxRef{
		oldHash: plumbing.NewHash(arr[0]),
		newHash: plumbing.NewHash(arr[1]),
		ref:     strings.TrimSuffix(arr[2], string([]byte{0})),
	}

	return
}

func parseUploadPackWantsAndHaves(r io.Reader) (wants, haves []plumbing.Hash, err error) {

	dec := pktline.NewDecoder(r)

	for {
		var line []byte
		if err = dec.Decode(&line); err != nil {
			break
		} else if len(line) == 0 {
			continue
		} else {
			line = line[:len(line)-1]
		}

		if string(line) == "done" {
			break
		}

		log.Printf("DBG [upload-pack] %s", line)

		op := strings.Split(string(line), " ")
		switch op[0] {
		case "want":
			wants = append(wants, plumbing.NewHash(op[1]))

		case "have":
			haves = append(haves, plumbing.NewHash(op[1]))

		}
	}

	return
}

func capabilities() []byte {
	//return []byte("report-status delete-refs ofs-delta multi_ack_detailed")
	return []byte("report-status delete-refs ofs-delta")
}

func nullCapabilities() []byte {
	return append(append([]byte("capabilities^{}"), '\x00'), capabilities()...)
}
