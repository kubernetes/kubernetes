// Package sftp implements the SSH File Transfer Protocol as described in
// https://filezilla-project.org/specs/draft-ietf-secsh-filexfer-02.txt
package sftp

import (
	"fmt"

	"github.com/pkg/errors"
)

const (
	ssh_FXP_INIT           = 1
	ssh_FXP_VERSION        = 2
	ssh_FXP_OPEN           = 3
	ssh_FXP_CLOSE          = 4
	ssh_FXP_READ           = 5
	ssh_FXP_WRITE          = 6
	ssh_FXP_LSTAT          = 7
	ssh_FXP_FSTAT          = 8
	ssh_FXP_SETSTAT        = 9
	ssh_FXP_FSETSTAT       = 10
	ssh_FXP_OPENDIR        = 11
	ssh_FXP_READDIR        = 12
	ssh_FXP_REMOVE         = 13
	ssh_FXP_MKDIR          = 14
	ssh_FXP_RMDIR          = 15
	ssh_FXP_REALPATH       = 16
	ssh_FXP_STAT           = 17
	ssh_FXP_RENAME         = 18
	ssh_FXP_READLINK       = 19
	ssh_FXP_SYMLINK        = 20
	ssh_FXP_STATUS         = 101
	ssh_FXP_HANDLE         = 102
	ssh_FXP_DATA           = 103
	ssh_FXP_NAME           = 104
	ssh_FXP_ATTRS          = 105
	ssh_FXP_EXTENDED       = 200
	ssh_FXP_EXTENDED_REPLY = 201
)

const (
	ssh_FX_OK                = 0
	ssh_FX_EOF               = 1
	ssh_FX_NO_SUCH_FILE      = 2
	ssh_FX_PERMISSION_DENIED = 3
	ssh_FX_FAILURE           = 4
	ssh_FX_BAD_MESSAGE       = 5
	ssh_FX_NO_CONNECTION     = 6
	ssh_FX_CONNECTION_LOST   = 7
	ssh_FX_OP_UNSUPPORTED    = 8

	// see draft-ietf-secsh-filexfer-13
	// https://tools.ietf.org/html/draft-ietf-secsh-filexfer-13#section-9.1
	ssh_FX_INVALID_HANDLE              = 9
	ssh_FX_NO_SUCH_PATH                = 10
	ssh_FX_FILE_ALREADY_EXISTS         = 11
	ssh_FX_WRITE_PROTECT               = 12
	ssh_FX_NO_MEDIA                    = 13
	ssh_FX_NO_SPACE_ON_FILESYSTEM      = 14
	ssh_FX_QUOTA_EXCEEDED              = 15
	ssh_FX_UNKNOWN_PRINCIPAL           = 16
	ssh_FX_LOCK_CONFLICT               = 17
	ssh_FX_DIR_NOT_EMPTY               = 18
	ssh_FX_NOT_A_DIRECTORY             = 19
	ssh_FX_INVALID_FILENAME            = 20
	ssh_FX_LINK_LOOP                   = 21
	ssh_FX_CANNOT_DELETE               = 22
	ssh_FX_INVALID_PARAMETER           = 23
	ssh_FX_FILE_IS_A_DIRECTORY         = 24
	ssh_FX_BYTE_RANGE_LOCK_CONFLICT    = 25
	ssh_FX_BYTE_RANGE_LOCK_REFUSED     = 26
	ssh_FX_DELETE_PENDING              = 27
	ssh_FX_FILE_CORRUPT                = 28
	ssh_FX_OWNER_INVALID               = 29
	ssh_FX_GROUP_INVALID               = 30
	ssh_FX_NO_MATCHING_BYTE_RANGE_LOCK = 31
)

const (
	ssh_FXF_READ   = 0x00000001
	ssh_FXF_WRITE  = 0x00000002
	ssh_FXF_APPEND = 0x00000004
	ssh_FXF_CREAT  = 0x00000008
	ssh_FXF_TRUNC  = 0x00000010
	ssh_FXF_EXCL   = 0x00000020
)

type fxp uint8

func (f fxp) String() string {
	switch f {
	case ssh_FXP_INIT:
		return "SSH_FXP_INIT"
	case ssh_FXP_VERSION:
		return "SSH_FXP_VERSION"
	case ssh_FXP_OPEN:
		return "SSH_FXP_OPEN"
	case ssh_FXP_CLOSE:
		return "SSH_FXP_CLOSE"
	case ssh_FXP_READ:
		return "SSH_FXP_READ"
	case ssh_FXP_WRITE:
		return "SSH_FXP_WRITE"
	case ssh_FXP_LSTAT:
		return "SSH_FXP_LSTAT"
	case ssh_FXP_FSTAT:
		return "SSH_FXP_FSTAT"
	case ssh_FXP_SETSTAT:
		return "SSH_FXP_SETSTAT"
	case ssh_FXP_FSETSTAT:
		return "SSH_FXP_FSETSTAT"
	case ssh_FXP_OPENDIR:
		return "SSH_FXP_OPENDIR"
	case ssh_FXP_READDIR:
		return "SSH_FXP_READDIR"
	case ssh_FXP_REMOVE:
		return "SSH_FXP_REMOVE"
	case ssh_FXP_MKDIR:
		return "SSH_FXP_MKDIR"
	case ssh_FXP_RMDIR:
		return "SSH_FXP_RMDIR"
	case ssh_FXP_REALPATH:
		return "SSH_FXP_REALPATH"
	case ssh_FXP_STAT:
		return "SSH_FXP_STAT"
	case ssh_FXP_RENAME:
		return "SSH_FXP_RENAME"
	case ssh_FXP_READLINK:
		return "SSH_FXP_READLINK"
	case ssh_FXP_SYMLINK:
		return "SSH_FXP_SYMLINK"
	case ssh_FXP_STATUS:
		return "SSH_FXP_STATUS"
	case ssh_FXP_HANDLE:
		return "SSH_FXP_HANDLE"
	case ssh_FXP_DATA:
		return "SSH_FXP_DATA"
	case ssh_FXP_NAME:
		return "SSH_FXP_NAME"
	case ssh_FXP_ATTRS:
		return "SSH_FXP_ATTRS"
	case ssh_FXP_EXTENDED:
		return "SSH_FXP_EXTENDED"
	case ssh_FXP_EXTENDED_REPLY:
		return "SSH_FXP_EXTENDED_REPLY"
	default:
		return "unknown"
	}
}

type fx uint8

func (f fx) String() string {
	switch f {
	case ssh_FX_OK:
		return "SSH_FX_OK"
	case ssh_FX_EOF:
		return "SSH_FX_EOF"
	case ssh_FX_NO_SUCH_FILE:
		return "SSH_FX_NO_SUCH_FILE"
	case ssh_FX_PERMISSION_DENIED:
		return "SSH_FX_PERMISSION_DENIED"
	case ssh_FX_FAILURE:
		return "SSH_FX_FAILURE"
	case ssh_FX_BAD_MESSAGE:
		return "SSH_FX_BAD_MESSAGE"
	case ssh_FX_NO_CONNECTION:
		return "SSH_FX_NO_CONNECTION"
	case ssh_FX_CONNECTION_LOST:
		return "SSH_FX_CONNECTION_LOST"
	case ssh_FX_OP_UNSUPPORTED:
		return "SSH_FX_OP_UNSUPPORTED"
	default:
		return "unknown"
	}
}

type unexpectedPacketErr struct {
	want, got uint8
}

func (u *unexpectedPacketErr) Error() string {
	return fmt.Sprintf("sftp: unexpected packet: want %v, got %v", fxp(u.want), fxp(u.got))
}

func unimplementedPacketErr(u uint8) error {
	return errors.Errorf("sftp: unimplemented packet type: got %v", fxp(u))
}

type unexpectedIDErr struct{ want, got uint32 }

func (u *unexpectedIDErr) Error() string {
	return fmt.Sprintf("sftp: unexpected id: want %v, got %v", u.want, u.got)
}

func unimplementedSeekWhence(whence int) error {
	return errors.Errorf("sftp: unimplemented seek whence %v", whence)
}

func unexpectedCount(want, got uint32) error {
	return errors.Errorf("sftp: unexpected count: want %v, got %v", want, got)
}

type unexpectedVersionErr struct{ want, got uint32 }

func (u *unexpectedVersionErr) Error() string {
	return fmt.Sprintf("sftp: unexpected server version: want %v, got %v", u.want, u.got)
}

// A StatusError is returned when an SFTP operation fails, and provides
// additional information about the failure.
type StatusError struct {
	Code      uint32
	msg, lang string
}

func (s *StatusError) Error() string { return fmt.Sprintf("sftp: %q (%v)", s.msg, fx(s.Code)) }
