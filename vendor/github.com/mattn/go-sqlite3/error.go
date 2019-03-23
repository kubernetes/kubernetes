// Copyright (C) 2014 Yasuhiro Matsumoto <mattn.jp@gmail.com>.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package sqlite3

import "C"

// ErrNo inherit errno.
type ErrNo int

// ErrNoMask is mask code.
const ErrNoMask C.int = 0xff

// ErrNoExtended is extended errno.
type ErrNoExtended int

// Error implement sqlite error code.
type Error struct {
	Code         ErrNo         /* The error code returned by SQLite */
	ExtendedCode ErrNoExtended /* The extended error code returned by SQLite */
	err          string        /* The error string returned by sqlite3_errmsg(),
	this usually contains more specific details. */
}

// result codes from http://www.sqlite.org/c3ref/c_abort.html
var (
	ErrError      = ErrNo(1)  /* SQL error or missing database */
	ErrInternal   = ErrNo(2)  /* Internal logic error in SQLite */
	ErrPerm       = ErrNo(3)  /* Access permission denied */
	ErrAbort      = ErrNo(4)  /* Callback routine requested an abort */
	ErrBusy       = ErrNo(5)  /* The database file is locked */
	ErrLocked     = ErrNo(6)  /* A table in the database is locked */
	ErrNomem      = ErrNo(7)  /* A malloc() failed */
	ErrReadonly   = ErrNo(8)  /* Attempt to write a readonly database */
	ErrInterrupt  = ErrNo(9)  /* Operation terminated by sqlite3_interrupt() */
	ErrIoErr      = ErrNo(10) /* Some kind of disk I/O error occurred */
	ErrCorrupt    = ErrNo(11) /* The database disk image is malformed */
	ErrNotFound   = ErrNo(12) /* Unknown opcode in sqlite3_file_control() */
	ErrFull       = ErrNo(13) /* Insertion failed because database is full */
	ErrCantOpen   = ErrNo(14) /* Unable to open the database file */
	ErrProtocol   = ErrNo(15) /* Database lock protocol error */
	ErrEmpty      = ErrNo(16) /* Database is empty */
	ErrSchema     = ErrNo(17) /* The database schema changed */
	ErrTooBig     = ErrNo(18) /* String or BLOB exceeds size limit */
	ErrConstraint = ErrNo(19) /* Abort due to constraint violation */
	ErrMismatch   = ErrNo(20) /* Data type mismatch */
	ErrMisuse     = ErrNo(21) /* Library used incorrectly */
	ErrNoLFS      = ErrNo(22) /* Uses OS features not supported on host */
	ErrAuth       = ErrNo(23) /* Authorization denied */
	ErrFormat     = ErrNo(24) /* Auxiliary database format error */
	ErrRange      = ErrNo(25) /* 2nd parameter to sqlite3_bind out of range */
	ErrNotADB     = ErrNo(26) /* File opened that is not a database file */
	ErrNotice     = ErrNo(27) /* Notifications from sqlite3_log() */
	ErrWarning    = ErrNo(28) /* Warnings from sqlite3_log() */
)

// Error return error message from errno.
func (err ErrNo) Error() string {
	return Error{Code: err}.Error()
}

// Extend return extended errno.
func (err ErrNo) Extend(by int) ErrNoExtended {
	return ErrNoExtended(int(err) | (by << 8))
}

// Error return error message that is extended code.
func (err ErrNoExtended) Error() string {
	return Error{Code: ErrNo(C.int(err) & ErrNoMask), ExtendedCode: err}.Error()
}

func (err Error) Error() string {
	if err.err != "" {
		return err.err
	}
	return errorString(err)
}

// result codes from http://www.sqlite.org/c3ref/c_abort_rollback.html
var (
	ErrIoErrRead              = ErrIoErr.Extend(1)
	ErrIoErrShortRead         = ErrIoErr.Extend(2)
	ErrIoErrWrite             = ErrIoErr.Extend(3)
	ErrIoErrFsync             = ErrIoErr.Extend(4)
	ErrIoErrDirFsync          = ErrIoErr.Extend(5)
	ErrIoErrTruncate          = ErrIoErr.Extend(6)
	ErrIoErrFstat             = ErrIoErr.Extend(7)
	ErrIoErrUnlock            = ErrIoErr.Extend(8)
	ErrIoErrRDlock            = ErrIoErr.Extend(9)
	ErrIoErrDelete            = ErrIoErr.Extend(10)
	ErrIoErrBlocked           = ErrIoErr.Extend(11)
	ErrIoErrNoMem             = ErrIoErr.Extend(12)
	ErrIoErrAccess            = ErrIoErr.Extend(13)
	ErrIoErrCheckReservedLock = ErrIoErr.Extend(14)
	ErrIoErrLock              = ErrIoErr.Extend(15)
	ErrIoErrClose             = ErrIoErr.Extend(16)
	ErrIoErrDirClose          = ErrIoErr.Extend(17)
	ErrIoErrSHMOpen           = ErrIoErr.Extend(18)
	ErrIoErrSHMSize           = ErrIoErr.Extend(19)
	ErrIoErrSHMLock           = ErrIoErr.Extend(20)
	ErrIoErrSHMMap            = ErrIoErr.Extend(21)
	ErrIoErrSeek              = ErrIoErr.Extend(22)
	ErrIoErrDeleteNoent       = ErrIoErr.Extend(23)
	ErrIoErrMMap              = ErrIoErr.Extend(24)
	ErrIoErrGetTempPath       = ErrIoErr.Extend(25)
	ErrIoErrConvPath          = ErrIoErr.Extend(26)
	ErrLockedSharedCache      = ErrLocked.Extend(1)
	ErrBusyRecovery           = ErrBusy.Extend(1)
	ErrBusySnapshot           = ErrBusy.Extend(2)
	ErrCantOpenNoTempDir      = ErrCantOpen.Extend(1)
	ErrCantOpenIsDir          = ErrCantOpen.Extend(2)
	ErrCantOpenFullPath       = ErrCantOpen.Extend(3)
	ErrCantOpenConvPath       = ErrCantOpen.Extend(4)
	ErrCorruptVTab            = ErrCorrupt.Extend(1)
	ErrReadonlyRecovery       = ErrReadonly.Extend(1)
	ErrReadonlyCantLock       = ErrReadonly.Extend(2)
	ErrReadonlyRollback       = ErrReadonly.Extend(3)
	ErrReadonlyDbMoved        = ErrReadonly.Extend(4)
	ErrAbortRollback          = ErrAbort.Extend(2)
	ErrConstraintCheck        = ErrConstraint.Extend(1)
	ErrConstraintCommitHook   = ErrConstraint.Extend(2)
	ErrConstraintForeignKey   = ErrConstraint.Extend(3)
	ErrConstraintFunction     = ErrConstraint.Extend(4)
	ErrConstraintNotNull      = ErrConstraint.Extend(5)
	ErrConstraintPrimaryKey   = ErrConstraint.Extend(6)
	ErrConstraintTrigger      = ErrConstraint.Extend(7)
	ErrConstraintUnique       = ErrConstraint.Extend(8)
	ErrConstraintVTab         = ErrConstraint.Extend(9)
	ErrConstraintRowID        = ErrConstraint.Extend(10)
	ErrNoticeRecoverWAL       = ErrNotice.Extend(1)
	ErrNoticeRecoverRollback  = ErrNotice.Extend(2)
	ErrWarningAutoIndex       = ErrWarning.Extend(1)
)
