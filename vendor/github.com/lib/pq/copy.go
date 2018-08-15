package pq

import (
	"database/sql/driver"
	"encoding/binary"
	"errors"
	"fmt"
	"sync"
)

var (
	errCopyInClosed               = errors.New("pq: copyin statement has already been closed")
	errBinaryCopyNotSupported     = errors.New("pq: only text format supported for COPY")
	errCopyToNotSupported         = errors.New("pq: COPY TO is not supported")
	errCopyNotSupportedOutsideTxn = errors.New("pq: COPY is only allowed inside a transaction")
	errCopyInProgress             = errors.New("pq: COPY in progress")
)

// CopyIn creates a COPY FROM statement which can be prepared with
// Tx.Prepare().  The target table should be visible in search_path.
func CopyIn(table string, columns ...string) string {
	stmt := "COPY " + QuoteIdentifier(table) + " ("
	for i, col := range columns {
		if i != 0 {
			stmt += ", "
		}
		stmt += QuoteIdentifier(col)
	}
	stmt += ") FROM STDIN"
	return stmt
}

// CopyInSchema creates a COPY FROM statement which can be prepared with
// Tx.Prepare().
func CopyInSchema(schema, table string, columns ...string) string {
	stmt := "COPY " + QuoteIdentifier(schema) + "." + QuoteIdentifier(table) + " ("
	for i, col := range columns {
		if i != 0 {
			stmt += ", "
		}
		stmt += QuoteIdentifier(col)
	}
	stmt += ") FROM STDIN"
	return stmt
}

type copyin struct {
	cn      *conn
	buffer  []byte
	rowData chan []byte
	done    chan bool

	closed bool

	sync.Mutex // guards err
	err        error
}

const ciBufferSize = 64 * 1024

// flush buffer before the buffer is filled up and needs reallocation
const ciBufferFlushSize = 63 * 1024

func (cn *conn) prepareCopyIn(q string) (_ driver.Stmt, err error) {
	if !cn.isInTransaction() {
		return nil, errCopyNotSupportedOutsideTxn
	}

	ci := &copyin{
		cn:      cn,
		buffer:  make([]byte, 0, ciBufferSize),
		rowData: make(chan []byte),
		done:    make(chan bool, 1),
	}
	// add CopyData identifier + 4 bytes for message length
	ci.buffer = append(ci.buffer, 'd', 0, 0, 0, 0)

	b := cn.writeBuf('Q')
	b.string(q)
	cn.send(b)

awaitCopyInResponse:
	for {
		t, r := cn.recv1()
		switch t {
		case 'G':
			if r.byte() != 0 {
				err = errBinaryCopyNotSupported
				break awaitCopyInResponse
			}
			go ci.resploop()
			return ci, nil
		case 'H':
			err = errCopyToNotSupported
			break awaitCopyInResponse
		case 'E':
			err = parseError(r)
		case 'Z':
			if err == nil {
				ci.setBad()
				errorf("unexpected ReadyForQuery in response to COPY")
			}
			cn.processReadyForQuery(r)
			return nil, err
		default:
			ci.setBad()
			errorf("unknown response for copy query: %q", t)
		}
	}

	// something went wrong, abort COPY before we return
	b = cn.writeBuf('f')
	b.string(err.Error())
	cn.send(b)

	for {
		t, r := cn.recv1()
		switch t {
		case 'c', 'C', 'E':
		case 'Z':
			// correctly aborted, we're done
			cn.processReadyForQuery(r)
			return nil, err
		default:
			ci.setBad()
			errorf("unknown response for CopyFail: %q", t)
		}
	}
}

func (ci *copyin) flush(buf []byte) {
	// set message length (without message identifier)
	binary.BigEndian.PutUint32(buf[1:], uint32(len(buf)-1))

	_, err := ci.cn.c.Write(buf)
	if err != nil {
		panic(err)
	}
}

func (ci *copyin) resploop() {
	for {
		var r readBuf
		t, err := ci.cn.recvMessage(&r)
		if err != nil {
			ci.setBad()
			ci.setError(err)
			ci.done <- true
			return
		}
		switch t {
		case 'C':
			// complete
		case 'N':
			// NoticeResponse
		case 'Z':
			ci.cn.processReadyForQuery(&r)
			ci.done <- true
			return
		case 'E':
			err := parseError(&r)
			ci.setError(err)
		default:
			ci.setBad()
			ci.setError(fmt.Errorf("unknown response during CopyIn: %q", t))
			ci.done <- true
			return
		}
	}
}

func (ci *copyin) setBad() {
	ci.Lock()
	ci.cn.bad = true
	ci.Unlock()
}

func (ci *copyin) isBad() bool {
	ci.Lock()
	b := ci.cn.bad
	ci.Unlock()
	return b
}

func (ci *copyin) isErrorSet() bool {
	ci.Lock()
	isSet := (ci.err != nil)
	ci.Unlock()
	return isSet
}

// setError() sets ci.err if one has not been set already.  Caller must not be
// holding ci.Mutex.
func (ci *copyin) setError(err error) {
	ci.Lock()
	if ci.err == nil {
		ci.err = err
	}
	ci.Unlock()
}

func (ci *copyin) NumInput() int {
	return -1
}

func (ci *copyin) Query(v []driver.Value) (r driver.Rows, err error) {
	return nil, ErrNotSupported
}

// Exec inserts values into the COPY stream. The insert is asynchronous
// and Exec can return errors from previous Exec calls to the same
// COPY stmt.
//
// You need to call Exec(nil) to sync the COPY stream and to get any
// errors from pending data, since Stmt.Close() doesn't return errors
// to the user.
func (ci *copyin) Exec(v []driver.Value) (r driver.Result, err error) {
	if ci.closed {
		return nil, errCopyInClosed
	}

	if ci.isBad() {
		return nil, driver.ErrBadConn
	}
	defer ci.cn.errRecover(&err)

	if ci.isErrorSet() {
		return nil, ci.err
	}

	if len(v) == 0 {
		return nil, ci.Close()
	}

	numValues := len(v)
	for i, value := range v {
		ci.buffer = appendEncodedText(&ci.cn.parameterStatus, ci.buffer, value)
		if i < numValues-1 {
			ci.buffer = append(ci.buffer, '\t')
		}
	}

	ci.buffer = append(ci.buffer, '\n')

	if len(ci.buffer) > ciBufferFlushSize {
		ci.flush(ci.buffer)
		// reset buffer, keep bytes for message identifier and length
		ci.buffer = ci.buffer[:5]
	}

	return driver.RowsAffected(0), nil
}

func (ci *copyin) Close() (err error) {
	if ci.closed { // Don't do anything, we're already closed
		return nil
	}
	ci.closed = true

	if ci.isBad() {
		return driver.ErrBadConn
	}
	defer ci.cn.errRecover(&err)

	if len(ci.buffer) > 0 {
		ci.flush(ci.buffer)
	}
	// Avoid touching the scratch buffer as resploop could be using it.
	err = ci.cn.sendSimpleMessage('c')
	if err != nil {
		return err
	}

	<-ci.done
	ci.cn.inCopy = false

	if ci.isErrorSet() {
		err = ci.err
		return err
	}
	return nil
}
