package proxy

import "bytes"
import "crypto/rand"
import "encoding/binary"
import "errors"
import "net"
import "os"
import "github.com/davecgh/go-xdr/xdr2"

const (
	// Message Types
	CALL  = 0
	REPLY = 1

	// Reply Status values
	MSG_ACCEPTED = 0
	MSG_DENIED   = 1

	// Accept Status values
	SUCCESS       = 0
	PROG_UNAVAIL  = 1
	PROG_MISMATCH = 2
	PROC_UNAVAIL  = 3
	GARBAGE_ARGS  = 4
	SYSTEM_ERR    = 5

	// Reject Status values
	RPC_MISMATCH = 0
	AUTH_ERROR   = 1

	// Auth Why values
	AUTH_OK           = 0
	AUTH_BADCRED      = 1
	AUTH_REJECTEDCRED = 2
	AUTH_BADVERF      = 3
	AUTH_REJECTEDVERF = 4
	AUTH_TOOWEAK      = 5
	AUTH_INVALIDRESP  = 6
	AUTH_FAILED       = 7

	// Auth flavors
	AUTH_NONE = 0 /* no authentication */
	AUTH_NULL = AUTH_NONE
	AUTH_SYS  = 1 /* old-school unix style */
	AUTH_UNIX = AUTH_SYS
)

type rpcOpaqueAuth struct {
	Flavor uint32
	Body   []byte
}

type rpcCallMsg struct {
	Xid, MsgType, RpcVers, Prog, Vers, Proc uint32
	Cred, Verf                              rpcOpaqueAuth
}

type rpcReplyHeader struct {
	Xid, MsgType, ReplyStat uint32
}

type rpcReplyAcceptedMiddle struct {
	Verf       rpcOpaqueAuth
	AcceptStat uint32
}

type rpcReplyRejectedMiddle struct {
	RejectStat uint32
}

func uint32array(array []int) (values []uint32) {
	values = make([]uint32, len(array))
	for i, val := range array {
		values[i] = uint32(val)
	}
	return
}

func makeAuth(authFlavor uint32) (Cred, Verf rpcOpaqueAuth) {
	var sys struct {
		Stamp       int32
		Machinename string
		Uid, Gid    uint32
		Gids        []uint32
	}

	switch authFlavor {
	case AUTH_UNIX:
		var buf bytes.Buffer
		sys.Stamp = 0
		hostname, err := os.Hostname()
		if err != nil {
			sys.Machinename = "localhost"
		} else {
			sys.Machinename = hostname
		}
		sys.Uid = uint32(os.Getuid())
		sys.Gid = uint32(os.Getgid())
		gids, _ := os.Getgroups()
		if gids != nil {
			sys.Gids = uint32array(gids)
		} else {
			sys.Gids = make([]uint32, 0)
		}
		n, err := xdr.Marshal(&buf, sys)
		if err != nil {
			return
		}
		if n > 0 {
			Cred.Flavor = authFlavor
			Cred.Body = buf.Bytes()
			Verf.Flavor = AUTH_NONE
			Verf.Body = make([]byte, 0)
		} else {
			Cred.Flavor = AUTH_NONE
			Cred.Body = make([]byte, 0)
			Verf.Flavor = AUTH_NONE
			Verf.Body = make([]byte, 0)
		}
		return
	default:
		Cred.Flavor = AUTH_NONE
		Cred.Body = make([]byte, 0)
		Verf.Flavor = AUTH_NONE
		Verf.Body = make([]byte, 0)
		return
	}
}

/* CallRpc invokes a minimal stream-based ONC RPC call over the provided connection.  While it can supply AUTH_UNIX, it doesn't verify any credentials in the response from the server. */
func CallRpc(conn *net.Conn, prog, vers, proc, authFlavor uint32, body []byte, reply *bytes.Buffer) (err error) {
	var cheader rpcCallMsg
	var rheader rpcReplyHeader
	var amiddle rpcReplyAcceptedMiddle
	var rmiddle rpcReplyRejectedMiddle
	var cbuf, rbuf bytes.Buffer
	var flen uint32
	var nb int

	/* Fill out the RPC call header. */
	xid := make([]byte, 4)
	rand.Read(xid)
	cheader.Xid = uint32(xid[0]<<24 | xid[1]<<16 | xid[2]<<8 | xid[3])
	cheader.MsgType = CALL
	cheader.RpcVers = 2
	cheader.Prog = prog
	cheader.Vers = vers
	cheader.Proc = proc
	cheader.Cred, cheader.Verf = makeAuth(authFlavor)
	nh, err := xdr.Marshal(&cbuf, &cheader)
	if err != nil {
		return
	}

	/* Format the RPC call body. */
	if body != nil {
		nb = len(body)
		cbuf.Write(body)
	}

	/* Calculate the lone fragment's length. */
	if int64(nh)+int64(nb) < 0 || int64(nh)+int64(nb) >= 0x80000000 {
		err = errors.New("RPC call message would have an invalid length")
		return
	}
	flen = uint32(nh + nb)
	flen |= 0x80000000

	/* Send the fragment length and the request. */
	binary.Write(*conn, binary.BigEndian, flen)
	binary.Write(*conn, binary.BigEndian, cbuf.Bytes())

	/* Read the first fragment's length. */
	err = binary.Read(*conn, binary.BigEndian, &flen)
	if err != nil {
		return
	}

	/* So long as we're still getting fragments,... */
	for flen != 0 {
		/* Read the current fragment.... */
		for flen&0x7fffffff != 0 {
			tmp := make([]byte, flen&0x7fffffff)
			err = binary.Read(*conn, binary.BigEndian, tmp)
			if err != nil {
				return
			}
			rbuf.Write(tmp)
			flen -= uint32(len(tmp))
		}
		/* And if it's not the last fragment,... */
		if flen&0x80000000 != 0 {
			break
		}
		/* Read the length of the next fragment. */
		err = binary.Read(*conn, binary.BigEndian, &flen)
		if err != nil {
			return
		}
	}

	/* Check the Xid and message type. */
	_, err = xdr.Unmarshal(&rbuf, &rheader)
	if err != nil {
		return
	}
	if rheader.MsgType != REPLY {
		err = errors.New("RPC message was not marked as a reply")
		return
	}
	if rheader.Xid != cheader.Xid {
		err = errors.New("RPC reply was for a different RPC call")
		return
	}
	if rheader.ReplyStat == MSG_ACCEPTED {
		/* Check the execution status. */
		_, err = xdr.Unmarshal(&rbuf, &amiddle)
		if err != nil {
			return
		}
		/* Check for an execution error. */
		switch amiddle.AcceptStat {
		case PROG_UNAVAIL:
			err = errors.New("RPC program unavailable")
		case PROG_MISMATCH:
			err = errors.New("RPC program mismatch")
		case PROC_UNAVAIL:
			err = errors.New("RPC procedure unavailable")
		case GARBAGE_ARGS:
			err = errors.New("RPC procedure arguments could not be parsed")
		case SYSTEM_ERR:
			err = errors.New("RPC system-level error")
		}
		if err != nil {
			return
		}
		/* Return the rest of the data. */
		*reply = rbuf
	} else {
		/* Check what sort of rejection/denial this was. */
		_, err = xdr.Unmarshal(&rbuf, &rmiddle)
		if err != nil {
			return
		}
		switch rmiddle.RejectStat {
		case RPC_MISMATCH:
			err = errors.New("RPC mismatch")
		case AUTH_ERROR:
			err = errors.New("RPC authentication error")
		default:
			err = errors.New("unknown error")
		}
		return
	}
	return
}
