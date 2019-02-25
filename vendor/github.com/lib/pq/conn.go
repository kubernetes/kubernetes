package pq

import (
	"bufio"
	"crypto/md5"
	"database/sql"
	"database/sql/driver"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/user"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/lib/pq/oid"
)

// Common error types
var (
	ErrNotSupported              = errors.New("pq: Unsupported command")
	ErrInFailedTransaction       = errors.New("pq: Could not complete operation in a failed transaction")
	ErrSSLNotSupported           = errors.New("pq: SSL is not enabled on the server")
	ErrSSLKeyHasWorldPermissions = errors.New("pq: Private key file has group or world access. Permissions should be u=rw (0600) or less")
	ErrCouldNotDetectUsername    = errors.New("pq: Could not detect default username. Please provide one explicitly")

	errUnexpectedReady = errors.New("unexpected ReadyForQuery")
	errNoRowsAffected  = errors.New("no RowsAffected available after the empty statement")
	errNoLastInsertID  = errors.New("no LastInsertId available after the empty statement")
)

// Driver is the Postgres database driver.
type Driver struct{}

// Open opens a new connection to the database. name is a connection string.
// Most users should only use it through database/sql package from the standard
// library.
func (d *Driver) Open(name string) (driver.Conn, error) {
	return Open(name)
}

func init() {
	sql.Register("postgres", &Driver{})
}

type parameterStatus struct {
	// server version in the same format as server_version_num, or 0 if
	// unavailable
	serverVersion int

	// the current location based on the TimeZone value of the session, if
	// available
	currentLocation *time.Location
}

type transactionStatus byte

const (
	txnStatusIdle                transactionStatus = 'I'
	txnStatusIdleInTransaction   transactionStatus = 'T'
	txnStatusInFailedTransaction transactionStatus = 'E'
)

func (s transactionStatus) String() string {
	switch s {
	case txnStatusIdle:
		return "idle"
	case txnStatusIdleInTransaction:
		return "idle in transaction"
	case txnStatusInFailedTransaction:
		return "in a failed transaction"
	default:
		errorf("unknown transactionStatus %d", s)
	}

	panic("not reached")
}

// Dialer is the dialer interface. It can be used to obtain more control over
// how pq creates network connections.
type Dialer interface {
	Dial(network, address string) (net.Conn, error)
	DialTimeout(network, address string, timeout time.Duration) (net.Conn, error)
}

type defaultDialer struct{}

func (d defaultDialer) Dial(ntw, addr string) (net.Conn, error) {
	return net.Dial(ntw, addr)
}
func (d defaultDialer) DialTimeout(ntw, addr string, timeout time.Duration) (net.Conn, error) {
	return net.DialTimeout(ntw, addr, timeout)
}

type conn struct {
	c         net.Conn
	buf       *bufio.Reader
	namei     int
	scratch   [512]byte
	txnStatus transactionStatus
	txnFinish func()

	// Save connection arguments to use during CancelRequest.
	dialer Dialer
	opts   values

	// Cancellation key data for use with CancelRequest messages.
	processID int
	secretKey int

	parameterStatus parameterStatus

	saveMessageType   byte
	saveMessageBuffer []byte

	// If true, this connection is bad and all public-facing functions should
	// return ErrBadConn.
	bad bool

	// If set, this connection should never use the binary format when
	// receiving query results from prepared statements.  Only provided for
	// debugging.
	disablePreparedBinaryResult bool

	// Whether to always send []byte parameters over as binary.  Enables single
	// round-trip mode for non-prepared Query calls.
	binaryParameters bool

	// If true this connection is in the middle of a COPY
	inCopy bool
}

// Handle driver-side settings in parsed connection string.
func (cn *conn) handleDriverSettings(o values) (err error) {
	boolSetting := func(key string, val *bool) error {
		if value, ok := o[key]; ok {
			if value == "yes" {
				*val = true
			} else if value == "no" {
				*val = false
			} else {
				return fmt.Errorf("unrecognized value %q for %s", value, key)
			}
		}
		return nil
	}

	err = boolSetting("disable_prepared_binary_result", &cn.disablePreparedBinaryResult)
	if err != nil {
		return err
	}
	return boolSetting("binary_parameters", &cn.binaryParameters)
}

func (cn *conn) handlePgpass(o values) {
	// if a password was supplied, do not process .pgpass
	if _, ok := o["password"]; ok {
		return
	}
	filename := os.Getenv("PGPASSFILE")
	if filename == "" {
		// XXX this code doesn't work on Windows where the default filename is
		// XXX %APPDATA%\postgresql\pgpass.conf
		// Prefer $HOME over user.Current due to glibc bug: golang.org/issue/13470
		userHome := os.Getenv("HOME")
		if userHome == "" {
			user, err := user.Current()
			if err != nil {
				return
			}
			userHome = user.HomeDir
		}
		filename = filepath.Join(userHome, ".pgpass")
	}
	fileinfo, err := os.Stat(filename)
	if err != nil {
		return
	}
	mode := fileinfo.Mode()
	if mode&(0x77) != 0 {
		// XXX should warn about incorrect .pgpass permissions as psql does
		return
	}
	file, err := os.Open(filename)
	if err != nil {
		return
	}
	defer file.Close()
	scanner := bufio.NewScanner(io.Reader(file))
	hostname := o["host"]
	ntw, _ := network(o)
	port := o["port"]
	db := o["dbname"]
	username := o["user"]
	// From: https://github.com/tg/pgpass/blob/master/reader.go
	getFields := func(s string) []string {
		fs := make([]string, 0, 5)
		f := make([]rune, 0, len(s))

		var esc bool
		for _, c := range s {
			switch {
			case esc:
				f = append(f, c)
				esc = false
			case c == '\\':
				esc = true
			case c == ':':
				fs = append(fs, string(f))
				f = f[:0]
			default:
				f = append(f, c)
			}
		}
		return append(fs, string(f))
	}
	for scanner.Scan() {
		line := scanner.Text()
		if len(line) == 0 || line[0] == '#' {
			continue
		}
		split := getFields(line)
		if len(split) != 5 {
			continue
		}
		if (split[0] == "*" || split[0] == hostname || (split[0] == "localhost" && (hostname == "" || ntw == "unix"))) && (split[1] == "*" || split[1] == port) && (split[2] == "*" || split[2] == db) && (split[3] == "*" || split[3] == username) {
			o["password"] = split[4]
			return
		}
	}
}

func (cn *conn) writeBuf(b byte) *writeBuf {
	cn.scratch[0] = b
	return &writeBuf{
		buf: cn.scratch[:5],
		pos: 1,
	}
}

// Open opens a new connection to the database. name is a connection string.
// Most users should only use it through database/sql package from the standard
// library.
func Open(name string) (_ driver.Conn, err error) {
	return DialOpen(defaultDialer{}, name)
}

// DialOpen opens a new connection to the database using a dialer.
func DialOpen(d Dialer, name string) (_ driver.Conn, err error) {
	// Handle any panics during connection initialization.  Note that we
	// specifically do *not* want to use errRecover(), as that would turn any
	// connection errors into ErrBadConns, hiding the real error message from
	// the user.
	defer errRecoverNoErrBadConn(&err)

	o := make(values)

	// A number of defaults are applied here, in this order:
	//
	// * Very low precedence defaults applied in every situation
	// * Environment variables
	// * Explicitly passed connection information
	o["host"] = "localhost"
	o["port"] = "5432"
	// N.B.: Extra float digits should be set to 3, but that breaks
	// Postgres 8.4 and older, where the max is 2.
	o["extra_float_digits"] = "2"
	for k, v := range parseEnviron(os.Environ()) {
		o[k] = v
	}

	if strings.HasPrefix(name, "postgres://") || strings.HasPrefix(name, "postgresql://") {
		name, err = ParseURL(name)
		if err != nil {
			return nil, err
		}
	}

	if err := parseOpts(name, o); err != nil {
		return nil, err
	}

	// Use the "fallback" application name if necessary
	if fallback, ok := o["fallback_application_name"]; ok {
		if _, ok := o["application_name"]; !ok {
			o["application_name"] = fallback
		}
	}

	// We can't work with any client_encoding other than UTF-8 currently.
	// However, we have historically allowed the user to set it to UTF-8
	// explicitly, and there's no reason to break such programs, so allow that.
	// Note that the "options" setting could also set client_encoding, but
	// parsing its value is not worth it.  Instead, we always explicitly send
	// client_encoding as a separate run-time parameter, which should override
	// anything set in options.
	if enc, ok := o["client_encoding"]; ok && !isUTF8(enc) {
		return nil, errors.New("client_encoding must be absent or 'UTF8'")
	}
	o["client_encoding"] = "UTF8"
	// DateStyle needs a similar treatment.
	if datestyle, ok := o["datestyle"]; ok {
		if datestyle != "ISO, MDY" {
			panic(fmt.Sprintf("setting datestyle must be absent or %v; got %v",
				"ISO, MDY", datestyle))
		}
	} else {
		o["datestyle"] = "ISO, MDY"
	}

	// If a user is not provided by any other means, the last
	// resort is to use the current operating system provided user
	// name.
	if _, ok := o["user"]; !ok {
		u, err := userCurrent()
		if err != nil {
			return nil, err
		}
		o["user"] = u
	}

	cn := &conn{
		opts:   o,
		dialer: d,
	}
	err = cn.handleDriverSettings(o)
	if err != nil {
		return nil, err
	}
	cn.handlePgpass(o)

	cn.c, err = dial(d, o)
	if err != nil {
		return nil, err
	}

	// cn.ssl and cn.startup panic on error. Make sure we don't leak cn.c.
	panicking := true
	defer func() {
		if panicking {
			cn.c.Close()
		}
	}()

	cn.ssl(o)
	cn.buf = bufio.NewReader(cn.c)
	cn.startup(o)

	// reset the deadline, in case one was set (see dial)
	if timeout, ok := o["connect_timeout"]; ok && timeout != "0" {
		err = cn.c.SetDeadline(time.Time{})
	}
	panicking = false
	return cn, err
}

func dial(d Dialer, o values) (net.Conn, error) {
	ntw, addr := network(o)
	// SSL is not necessary or supported over UNIX domain sockets
	if ntw == "unix" {
		o["sslmode"] = "disable"
	}

	// Zero or not specified means wait indefinitely.
	if timeout, ok := o["connect_timeout"]; ok && timeout != "0" {
		seconds, err := strconv.ParseInt(timeout, 10, 0)
		if err != nil {
			return nil, fmt.Errorf("invalid value for parameter connect_timeout: %s", err)
		}
		duration := time.Duration(seconds) * time.Second
		// connect_timeout should apply to the entire connection establishment
		// procedure, so we both use a timeout for the TCP connection
		// establishment and set a deadline for doing the initial handshake.
		// The deadline is then reset after startup() is done.
		deadline := time.Now().Add(duration)
		conn, err := d.DialTimeout(ntw, addr, duration)
		if err != nil {
			return nil, err
		}
		err = conn.SetDeadline(deadline)
		return conn, err
	}
	return d.Dial(ntw, addr)
}

func network(o values) (string, string) {
	host := o["host"]

	if strings.HasPrefix(host, "/") {
		sockPath := path.Join(host, ".s.PGSQL."+o["port"])
		return "unix", sockPath
	}

	return "tcp", net.JoinHostPort(host, o["port"])
}

type values map[string]string

// scanner implements a tokenizer for libpq-style option strings.
type scanner struct {
	s []rune
	i int
}

// newScanner returns a new scanner initialized with the option string s.
func newScanner(s string) *scanner {
	return &scanner{[]rune(s), 0}
}

// Next returns the next rune.
// It returns 0, false if the end of the text has been reached.
func (s *scanner) Next() (rune, bool) {
	if s.i >= len(s.s) {
		return 0, false
	}
	r := s.s[s.i]
	s.i++
	return r, true
}

// SkipSpaces returns the next non-whitespace rune.
// It returns 0, false if the end of the text has been reached.
func (s *scanner) SkipSpaces() (rune, bool) {
	r, ok := s.Next()
	for unicode.IsSpace(r) && ok {
		r, ok = s.Next()
	}
	return r, ok
}

// parseOpts parses the options from name and adds them to the values.
//
// The parsing code is based on conninfo_parse from libpq's fe-connect.c
func parseOpts(name string, o values) error {
	s := newScanner(name)

	for {
		var (
			keyRunes, valRunes []rune
			r                  rune
			ok                 bool
		)

		if r, ok = s.SkipSpaces(); !ok {
			break
		}

		// Scan the key
		for !unicode.IsSpace(r) && r != '=' {
			keyRunes = append(keyRunes, r)
			if r, ok = s.Next(); !ok {
				break
			}
		}

		// Skip any whitespace if we're not at the = yet
		if r != '=' {
			r, ok = s.SkipSpaces()
		}

		// The current character should be =
		if r != '=' || !ok {
			return fmt.Errorf(`missing "=" after %q in connection info string"`, string(keyRunes))
		}

		// Skip any whitespace after the =
		if r, ok = s.SkipSpaces(); !ok {
			// If we reach the end here, the last value is just an empty string as per libpq.
			o[string(keyRunes)] = ""
			break
		}

		if r != '\'' {
			for !unicode.IsSpace(r) {
				if r == '\\' {
					if r, ok = s.Next(); !ok {
						return fmt.Errorf(`missing character after backslash`)
					}
				}
				valRunes = append(valRunes, r)

				if r, ok = s.Next(); !ok {
					break
				}
			}
		} else {
		quote:
			for {
				if r, ok = s.Next(); !ok {
					return fmt.Errorf(`unterminated quoted string literal in connection string`)
				}
				switch r {
				case '\'':
					break quote
				case '\\':
					r, _ = s.Next()
					fallthrough
				default:
					valRunes = append(valRunes, r)
				}
			}
		}

		o[string(keyRunes)] = string(valRunes)
	}

	return nil
}

func (cn *conn) isInTransaction() bool {
	return cn.txnStatus == txnStatusIdleInTransaction ||
		cn.txnStatus == txnStatusInFailedTransaction
}

func (cn *conn) checkIsInTransaction(intxn bool) {
	if cn.isInTransaction() != intxn {
		cn.bad = true
		errorf("unexpected transaction status %v", cn.txnStatus)
	}
}

func (cn *conn) Begin() (_ driver.Tx, err error) {
	return cn.begin("")
}

func (cn *conn) begin(mode string) (_ driver.Tx, err error) {
	if cn.bad {
		return nil, driver.ErrBadConn
	}
	defer cn.errRecover(&err)

	cn.checkIsInTransaction(false)
	_, commandTag, err := cn.simpleExec("BEGIN" + mode)
	if err != nil {
		return nil, err
	}
	if commandTag != "BEGIN" {
		cn.bad = true
		return nil, fmt.Errorf("unexpected command tag %s", commandTag)
	}
	if cn.txnStatus != txnStatusIdleInTransaction {
		cn.bad = true
		return nil, fmt.Errorf("unexpected transaction status %v", cn.txnStatus)
	}
	return cn, nil
}

func (cn *conn) closeTxn() {
	if finish := cn.txnFinish; finish != nil {
		finish()
	}
}

func (cn *conn) Commit() (err error) {
	defer cn.closeTxn()
	if cn.bad {
		return driver.ErrBadConn
	}
	defer cn.errRecover(&err)

	cn.checkIsInTransaction(true)
	// We don't want the client to think that everything is okay if it tries
	// to commit a failed transaction.  However, no matter what we return,
	// database/sql will release this connection back into the free connection
	// pool so we have to abort the current transaction here.  Note that you
	// would get the same behaviour if you issued a COMMIT in a failed
	// transaction, so it's also the least surprising thing to do here.
	if cn.txnStatus == txnStatusInFailedTransaction {
		if err := cn.Rollback(); err != nil {
			return err
		}
		return ErrInFailedTransaction
	}

	_, commandTag, err := cn.simpleExec("COMMIT")
	if err != nil {
		if cn.isInTransaction() {
			cn.bad = true
		}
		return err
	}
	if commandTag != "COMMIT" {
		cn.bad = true
		return fmt.Errorf("unexpected command tag %s", commandTag)
	}
	cn.checkIsInTransaction(false)
	return nil
}

func (cn *conn) Rollback() (err error) {
	defer cn.closeTxn()
	if cn.bad {
		return driver.ErrBadConn
	}
	defer cn.errRecover(&err)

	cn.checkIsInTransaction(true)
	_, commandTag, err := cn.simpleExec("ROLLBACK")
	if err != nil {
		if cn.isInTransaction() {
			cn.bad = true
		}
		return err
	}
	if commandTag != "ROLLBACK" {
		return fmt.Errorf("unexpected command tag %s", commandTag)
	}
	cn.checkIsInTransaction(false)
	return nil
}

func (cn *conn) gname() string {
	cn.namei++
	return strconv.FormatInt(int64(cn.namei), 10)
}

func (cn *conn) simpleExec(q string) (res driver.Result, commandTag string, err error) {
	b := cn.writeBuf('Q')
	b.string(q)
	cn.send(b)

	for {
		t, r := cn.recv1()
		switch t {
		case 'C':
			res, commandTag = cn.parseComplete(r.string())
		case 'Z':
			cn.processReadyForQuery(r)
			if res == nil && err == nil {
				err = errUnexpectedReady
			}
			// done
			return
		case 'E':
			err = parseError(r)
		case 'I':
			res = emptyRows
		case 'T', 'D':
			// ignore any results
		default:
			cn.bad = true
			errorf("unknown response for simple query: %q", t)
		}
	}
}

func (cn *conn) simpleQuery(q string) (res *rows, err error) {
	defer cn.errRecover(&err)

	b := cn.writeBuf('Q')
	b.string(q)
	cn.send(b)

	for {
		t, r := cn.recv1()
		switch t {
		case 'C', 'I':
			// We allow queries which don't return any results through Query as
			// well as Exec.  We still have to give database/sql a rows object
			// the user can close, though, to avoid connections from being
			// leaked.  A "rows" with done=true works fine for that purpose.
			if err != nil {
				cn.bad = true
				errorf("unexpected message %q in simple query execution", t)
			}
			if res == nil {
				res = &rows{
					cn: cn,
				}
			}
			// Set the result and tag to the last command complete if there wasn't a
			// query already run. Although queries usually return from here and cede
			// control to Next, a query with zero results does not.
			if t == 'C' && res.colNames == nil {
				res.result, res.tag = cn.parseComplete(r.string())
			}
			res.done = true
		case 'Z':
			cn.processReadyForQuery(r)
			// done
			return
		case 'E':
			res = nil
			err = parseError(r)
		case 'D':
			if res == nil {
				cn.bad = true
				errorf("unexpected DataRow in simple query execution")
			}
			// the query didn't fail; kick off to Next
			cn.saveMessage(t, r)
			return
		case 'T':
			// res might be non-nil here if we received a previous
			// CommandComplete, but that's fine; just overwrite it
			res = &rows{cn: cn}
			res.colNames, res.colFmts, res.colTyps = parsePortalRowDescribe(r)

			// To work around a bug in QueryRow in Go 1.2 and earlier, wait
			// until the first DataRow has been received.
		default:
			cn.bad = true
			errorf("unknown response for simple query: %q", t)
		}
	}
}

type noRows struct{}

var emptyRows noRows

var _ driver.Result = noRows{}

func (noRows) LastInsertId() (int64, error) {
	return 0, errNoLastInsertID
}

func (noRows) RowsAffected() (int64, error) {
	return 0, errNoRowsAffected
}

// Decides which column formats to use for a prepared statement.  The input is
// an array of type oids, one element per result column.
func decideColumnFormats(colTyps []fieldDesc, forceText bool) (colFmts []format, colFmtData []byte) {
	if len(colTyps) == 0 {
		return nil, colFmtDataAllText
	}

	colFmts = make([]format, len(colTyps))
	if forceText {
		return colFmts, colFmtDataAllText
	}

	allBinary := true
	allText := true
	for i, t := range colTyps {
		switch t.OID {
		// This is the list of types to use binary mode for when receiving them
		// through a prepared statement.  If a type appears in this list, it
		// must also be implemented in binaryDecode in encode.go.
		case oid.T_bytea:
			fallthrough
		case oid.T_int8:
			fallthrough
		case oid.T_int4:
			fallthrough
		case oid.T_int2:
			fallthrough
		case oid.T_uuid:
			colFmts[i] = formatBinary
			allText = false

		default:
			allBinary = false
		}
	}

	if allBinary {
		return colFmts, colFmtDataAllBinary
	} else if allText {
		return colFmts, colFmtDataAllText
	} else {
		colFmtData = make([]byte, 2+len(colFmts)*2)
		binary.BigEndian.PutUint16(colFmtData, uint16(len(colFmts)))
		for i, v := range colFmts {
			binary.BigEndian.PutUint16(colFmtData[2+i*2:], uint16(v))
		}
		return colFmts, colFmtData
	}
}

func (cn *conn) prepareTo(q, stmtName string) *stmt {
	st := &stmt{cn: cn, name: stmtName}

	b := cn.writeBuf('P')
	b.string(st.name)
	b.string(q)
	b.int16(0)

	b.next('D')
	b.byte('S')
	b.string(st.name)

	b.next('S')
	cn.send(b)

	cn.readParseResponse()
	st.paramTyps, st.colNames, st.colTyps = cn.readStatementDescribeResponse()
	st.colFmts, st.colFmtData = decideColumnFormats(st.colTyps, cn.disablePreparedBinaryResult)
	cn.readReadyForQuery()
	return st
}

func (cn *conn) Prepare(q string) (_ driver.Stmt, err error) {
	if cn.bad {
		return nil, driver.ErrBadConn
	}
	defer cn.errRecover(&err)

	if len(q) >= 4 && strings.EqualFold(q[:4], "COPY") {
		s, err := cn.prepareCopyIn(q)
		if err == nil {
			cn.inCopy = true
		}
		return s, err
	}
	return cn.prepareTo(q, cn.gname()), nil
}

func (cn *conn) Close() (err error) {
	// Skip cn.bad return here because we always want to close a connection.
	defer cn.errRecover(&err)

	// Ensure that cn.c.Close is always run. Since error handling is done with
	// panics and cn.errRecover, the Close must be in a defer.
	defer func() {
		cerr := cn.c.Close()
		if err == nil {
			err = cerr
		}
	}()

	// Don't go through send(); ListenerConn relies on us not scribbling on the
	// scratch buffer of this connection.
	return cn.sendSimpleMessage('X')
}

// Implement the "Queryer" interface
func (cn *conn) Query(query string, args []driver.Value) (driver.Rows, error) {
	return cn.query(query, args)
}

func (cn *conn) query(query string, args []driver.Value) (_ *rows, err error) {
	if cn.bad {
		return nil, driver.ErrBadConn
	}
	if cn.inCopy {
		return nil, errCopyInProgress
	}
	defer cn.errRecover(&err)

	// Check to see if we can use the "simpleQuery" interface, which is
	// *much* faster than going through prepare/exec
	if len(args) == 0 {
		return cn.simpleQuery(query)
	}

	if cn.binaryParameters {
		cn.sendBinaryModeQuery(query, args)

		cn.readParseResponse()
		cn.readBindResponse()
		rows := &rows{cn: cn}
		rows.colNames, rows.colFmts, rows.colTyps = cn.readPortalDescribeResponse()
		cn.postExecuteWorkaround()
		return rows, nil
	}
	st := cn.prepareTo(query, "")
	st.exec(args)
	return &rows{
		cn:       cn,
		colNames: st.colNames,
		colTyps:  st.colTyps,
		colFmts:  st.colFmts,
	}, nil
}

// Implement the optional "Execer" interface for one-shot queries
func (cn *conn) Exec(query string, args []driver.Value) (res driver.Result, err error) {
	if cn.bad {
		return nil, driver.ErrBadConn
	}
	defer cn.errRecover(&err)

	// Check to see if we can use the "simpleExec" interface, which is
	// *much* faster than going through prepare/exec
	if len(args) == 0 {
		// ignore commandTag, our caller doesn't care
		r, _, err := cn.simpleExec(query)
		return r, err
	}

	if cn.binaryParameters {
		cn.sendBinaryModeQuery(query, args)

		cn.readParseResponse()
		cn.readBindResponse()
		cn.readPortalDescribeResponse()
		cn.postExecuteWorkaround()
		res, _, err = cn.readExecuteResponse("Execute")
		return res, err
	}
	// Use the unnamed statement to defer planning until bind
	// time, or else value-based selectivity estimates cannot be
	// used.
	st := cn.prepareTo(query, "")
	r, err := st.Exec(args)
	if err != nil {
		panic(err)
	}
	return r, err
}

func (cn *conn) send(m *writeBuf) {
	_, err := cn.c.Write(m.wrap())
	if err != nil {
		panic(err)
	}
}

func (cn *conn) sendStartupPacket(m *writeBuf) error {
	_, err := cn.c.Write((m.wrap())[1:])
	return err
}

// Send a message of type typ to the server on the other end of cn.  The
// message should have no payload.  This method does not use the scratch
// buffer.
func (cn *conn) sendSimpleMessage(typ byte) (err error) {
	_, err = cn.c.Write([]byte{typ, '\x00', '\x00', '\x00', '\x04'})
	return err
}

// saveMessage memorizes a message and its buffer in the conn struct.
// recvMessage will then return these values on the next call to it.  This
// method is useful in cases where you have to see what the next message is
// going to be (e.g. to see whether it's an error or not) but you can't handle
// the message yourself.
func (cn *conn) saveMessage(typ byte, buf *readBuf) {
	if cn.saveMessageType != 0 {
		cn.bad = true
		errorf("unexpected saveMessageType %d", cn.saveMessageType)
	}
	cn.saveMessageType = typ
	cn.saveMessageBuffer = *buf
}

// recvMessage receives any message from the backend, or returns an error if
// a problem occurred while reading the message.
func (cn *conn) recvMessage(r *readBuf) (byte, error) {
	// workaround for a QueryRow bug, see exec
	if cn.saveMessageType != 0 {
		t := cn.saveMessageType
		*r = cn.saveMessageBuffer
		cn.saveMessageType = 0
		cn.saveMessageBuffer = nil
		return t, nil
	}

	x := cn.scratch[:5]
	_, err := io.ReadFull(cn.buf, x)
	if err != nil {
		return 0, err
	}

	// read the type and length of the message that follows
	t := x[0]
	n := int(binary.BigEndian.Uint32(x[1:])) - 4
	var y []byte
	if n <= len(cn.scratch) {
		y = cn.scratch[:n]
	} else {
		y = make([]byte, n)
	}
	_, err = io.ReadFull(cn.buf, y)
	if err != nil {
		return 0, err
	}
	*r = y
	return t, nil
}

// recv receives a message from the backend, but if an error happened while
// reading the message or the received message was an ErrorResponse, it panics.
// NoticeResponses are ignored.  This function should generally be used only
// during the startup sequence.
func (cn *conn) recv() (t byte, r *readBuf) {
	for {
		var err error
		r = &readBuf{}
		t, err = cn.recvMessage(r)
		if err != nil {
			panic(err)
		}

		switch t {
		case 'E':
			panic(parseError(r))
		case 'N':
			// ignore
		default:
			return
		}
	}
}

// recv1Buf is exactly equivalent to recv1, except it uses a buffer supplied by
// the caller to avoid an allocation.
func (cn *conn) recv1Buf(r *readBuf) byte {
	for {
		t, err := cn.recvMessage(r)
		if err != nil {
			panic(err)
		}

		switch t {
		case 'A', 'N':
			// ignore
		case 'S':
			cn.processParameterStatus(r)
		default:
			return t
		}
	}
}

// recv1 receives a message from the backend, panicking if an error occurs
// while attempting to read it.  All asynchronous messages are ignored, with
// the exception of ErrorResponse.
func (cn *conn) recv1() (t byte, r *readBuf) {
	r = &readBuf{}
	t = cn.recv1Buf(r)
	return t, r
}

func (cn *conn) ssl(o values) {
	upgrade := ssl(o)
	if upgrade == nil {
		// Nothing to do
		return
	}

	w := cn.writeBuf(0)
	w.int32(80877103)
	if err := cn.sendStartupPacket(w); err != nil {
		panic(err)
	}

	b := cn.scratch[:1]
	_, err := io.ReadFull(cn.c, b)
	if err != nil {
		panic(err)
	}

	if b[0] != 'S' {
		panic(ErrSSLNotSupported)
	}

	cn.c = upgrade(cn.c)
}

// isDriverSetting returns true iff a setting is purely for configuring the
// driver's options and should not be sent to the server in the connection
// startup packet.
func isDriverSetting(key string) bool {
	switch key {
	case "host", "port":
		return true
	case "password":
		return true
	case "sslmode", "sslcert", "sslkey", "sslrootcert":
		return true
	case "fallback_application_name":
		return true
	case "connect_timeout":
		return true
	case "disable_prepared_binary_result":
		return true
	case "binary_parameters":
		return true

	default:
		return false
	}
}

func (cn *conn) startup(o values) {
	w := cn.writeBuf(0)
	w.int32(196608)
	// Send the backend the name of the database we want to connect to, and the
	// user we want to connect as.  Additionally, we send over any run-time
	// parameters potentially included in the connection string.  If the server
	// doesn't recognize any of them, it will reply with an error.
	for k, v := range o {
		if isDriverSetting(k) {
			// skip options which can't be run-time parameters
			continue
		}
		// The protocol requires us to supply the database name as "database"
		// instead of "dbname".
		if k == "dbname" {
			k = "database"
		}
		w.string(k)
		w.string(v)
	}
	w.string("")
	if err := cn.sendStartupPacket(w); err != nil {
		panic(err)
	}

	for {
		t, r := cn.recv()
		switch t {
		case 'K':
			cn.processBackendKeyData(r)
		case 'S':
			cn.processParameterStatus(r)
		case 'R':
			cn.auth(r, o)
		case 'Z':
			cn.processReadyForQuery(r)
			return
		default:
			errorf("unknown response for startup: %q", t)
		}
	}
}

func (cn *conn) auth(r *readBuf, o values) {
	switch code := r.int32(); code {
	case 0:
		// OK
	case 3:
		w := cn.writeBuf('p')
		w.string(o["password"])
		cn.send(w)

		t, r := cn.recv()
		if t != 'R' {
			errorf("unexpected password response: %q", t)
		}

		if r.int32() != 0 {
			errorf("unexpected authentication response: %q", t)
		}
	case 5:
		s := string(r.next(4))
		w := cn.writeBuf('p')
		w.string("md5" + md5s(md5s(o["password"]+o["user"])+s))
		cn.send(w)

		t, r := cn.recv()
		if t != 'R' {
			errorf("unexpected password response: %q", t)
		}

		if r.int32() != 0 {
			errorf("unexpected authentication response: %q", t)
		}
	default:
		errorf("unknown authentication response: %d", code)
	}
}

type format int

const formatText format = 0
const formatBinary format = 1

// One result-column format code with the value 1 (i.e. all binary).
var colFmtDataAllBinary = []byte{0, 1, 0, 1}

// No result-column format codes (i.e. all text).
var colFmtDataAllText = []byte{0, 0}

type stmt struct {
	cn         *conn
	name       string
	colNames   []string
	colFmts    []format
	colFmtData []byte
	colTyps    []fieldDesc
	paramTyps  []oid.Oid
	closed     bool
}

func (st *stmt) Close() (err error) {
	if st.closed {
		return nil
	}
	if st.cn.bad {
		return driver.ErrBadConn
	}
	defer st.cn.errRecover(&err)

	w := st.cn.writeBuf('C')
	w.byte('S')
	w.string(st.name)
	st.cn.send(w)

	st.cn.send(st.cn.writeBuf('S'))

	t, _ := st.cn.recv1()
	if t != '3' {
		st.cn.bad = true
		errorf("unexpected close response: %q", t)
	}
	st.closed = true

	t, r := st.cn.recv1()
	if t != 'Z' {
		st.cn.bad = true
		errorf("expected ready for query, but got: %q", t)
	}
	st.cn.processReadyForQuery(r)

	return nil
}

func (st *stmt) Query(v []driver.Value) (r driver.Rows, err error) {
	if st.cn.bad {
		return nil, driver.ErrBadConn
	}
	defer st.cn.errRecover(&err)

	st.exec(v)
	return &rows{
		cn:       st.cn,
		colNames: st.colNames,
		colTyps:  st.colTyps,
		colFmts:  st.colFmts,
	}, nil
}

func (st *stmt) Exec(v []driver.Value) (res driver.Result, err error) {
	if st.cn.bad {
		return nil, driver.ErrBadConn
	}
	defer st.cn.errRecover(&err)

	st.exec(v)
	res, _, err = st.cn.readExecuteResponse("simple query")
	return res, err
}

func (st *stmt) exec(v []driver.Value) {
	if len(v) >= 65536 {
		errorf("got %d parameters but PostgreSQL only supports 65535 parameters", len(v))
	}
	if len(v) != len(st.paramTyps) {
		errorf("got %d parameters but the statement requires %d", len(v), len(st.paramTyps))
	}

	cn := st.cn
	w := cn.writeBuf('B')
	w.byte(0) // unnamed portal
	w.string(st.name)

	if cn.binaryParameters {
		cn.sendBinaryParameters(w, v)
	} else {
		w.int16(0)
		w.int16(len(v))
		for i, x := range v {
			if x == nil {
				w.int32(-1)
			} else {
				b := encode(&cn.parameterStatus, x, st.paramTyps[i])
				w.int32(len(b))
				w.bytes(b)
			}
		}
	}
	w.bytes(st.colFmtData)

	w.next('E')
	w.byte(0)
	w.int32(0)

	w.next('S')
	cn.send(w)

	cn.readBindResponse()
	cn.postExecuteWorkaround()

}

func (st *stmt) NumInput() int {
	return len(st.paramTyps)
}

// parseComplete parses the "command tag" from a CommandComplete message, and
// returns the number of rows affected (if applicable) and a string
// identifying only the command that was executed, e.g. "ALTER TABLE".  If the
// command tag could not be parsed, parseComplete panics.
func (cn *conn) parseComplete(commandTag string) (driver.Result, string) {
	commandsWithAffectedRows := []string{
		"SELECT ",
		// INSERT is handled below
		"UPDATE ",
		"DELETE ",
		"FETCH ",
		"MOVE ",
		"COPY ",
	}

	var affectedRows *string
	for _, tag := range commandsWithAffectedRows {
		if strings.HasPrefix(commandTag, tag) {
			t := commandTag[len(tag):]
			affectedRows = &t
			commandTag = tag[:len(tag)-1]
			break
		}
	}
	// INSERT also includes the oid of the inserted row in its command tag.
	// Oids in user tables are deprecated, and the oid is only returned when
	// exactly one row is inserted, so it's unlikely to be of value to any
	// real-world application and we can ignore it.
	if affectedRows == nil && strings.HasPrefix(commandTag, "INSERT ") {
		parts := strings.Split(commandTag, " ")
		if len(parts) != 3 {
			cn.bad = true
			errorf("unexpected INSERT command tag %s", commandTag)
		}
		affectedRows = &parts[len(parts)-1]
		commandTag = "INSERT"
	}
	// There should be no affected rows attached to the tag, just return it
	if affectedRows == nil {
		return driver.RowsAffected(0), commandTag
	}
	n, err := strconv.ParseInt(*affectedRows, 10, 64)
	if err != nil {
		cn.bad = true
		errorf("could not parse commandTag: %s", err)
	}
	return driver.RowsAffected(n), commandTag
}

type rows struct {
	cn       *conn
	finish   func()
	colNames []string
	colTyps  []fieldDesc
	colFmts  []format
	done     bool
	rb       readBuf
	result   driver.Result
	tag      string
}

func (rs *rows) Close() error {
	if finish := rs.finish; finish != nil {
		defer finish()
	}
	// no need to look at cn.bad as Next() will
	for {
		err := rs.Next(nil)
		switch err {
		case nil:
		case io.EOF:
			// rs.Next can return io.EOF on both 'Z' (ready for query) and 'T' (row
			// description, used with HasNextResultSet). We need to fetch messages until
			// we hit a 'Z', which is done by waiting for done to be set.
			if rs.done {
				return nil
			}
		default:
			return err
		}
	}
}

func (rs *rows) Columns() []string {
	return rs.colNames
}

func (rs *rows) Result() driver.Result {
	if rs.result == nil {
		return emptyRows
	}
	return rs.result
}

func (rs *rows) Tag() string {
	return rs.tag
}

func (rs *rows) Next(dest []driver.Value) (err error) {
	if rs.done {
		return io.EOF
	}

	conn := rs.cn
	if conn.bad {
		return driver.ErrBadConn
	}
	defer conn.errRecover(&err)

	for {
		t := conn.recv1Buf(&rs.rb)
		switch t {
		case 'E':
			err = parseError(&rs.rb)
		case 'C', 'I':
			if t == 'C' {
				rs.result, rs.tag = conn.parseComplete(rs.rb.string())
			}
			continue
		case 'Z':
			conn.processReadyForQuery(&rs.rb)
			rs.done = true
			if err != nil {
				return err
			}
			return io.EOF
		case 'D':
			n := rs.rb.int16()
			if err != nil {
				conn.bad = true
				errorf("unexpected DataRow after error %s", err)
			}
			if n < len(dest) {
				dest = dest[:n]
			}
			for i := range dest {
				l := rs.rb.int32()
				if l == -1 {
					dest[i] = nil
					continue
				}
				dest[i] = decode(&conn.parameterStatus, rs.rb.next(l), rs.colTyps[i].OID, rs.colFmts[i])
			}
			return
		case 'T':
			rs.colNames, rs.colFmts, rs.colTyps = parsePortalRowDescribe(&rs.rb)
			return io.EOF
		default:
			errorf("unexpected message after execute: %q", t)
		}
	}
}

func (rs *rows) HasNextResultSet() bool {
	return !rs.done
}

func (rs *rows) NextResultSet() error {
	return nil
}

// QuoteIdentifier quotes an "identifier" (e.g. a table or a column name) to be
// used as part of an SQL statement.  For example:
//
//    tblname := "my_table"
//    data := "my_data"
//    quoted := pq.QuoteIdentifier(tblname)
//    err := db.Exec(fmt.Sprintf("INSERT INTO %s VALUES ($1)", quoted), data)
//
// Any double quotes in name will be escaped.  The quoted identifier will be
// case sensitive when used in a query.  If the input string contains a zero
// byte, the result will be truncated immediately before it.
func QuoteIdentifier(name string) string {
	end := strings.IndexRune(name, 0)
	if end > -1 {
		name = name[:end]
	}
	return `"` + strings.Replace(name, `"`, `""`, -1) + `"`
}

func md5s(s string) string {
	h := md5.New()
	h.Write([]byte(s))
	return fmt.Sprintf("%x", h.Sum(nil))
}

func (cn *conn) sendBinaryParameters(b *writeBuf, args []driver.Value) {
	// Do one pass over the parameters to see if we're going to send any of
	// them over in binary.  If we are, create a paramFormats array at the
	// same time.
	var paramFormats []int
	for i, x := range args {
		_, ok := x.([]byte)
		if ok {
			if paramFormats == nil {
				paramFormats = make([]int, len(args))
			}
			paramFormats[i] = 1
		}
	}
	if paramFormats == nil {
		b.int16(0)
	} else {
		b.int16(len(paramFormats))
		for _, x := range paramFormats {
			b.int16(x)
		}
	}

	b.int16(len(args))
	for _, x := range args {
		if x == nil {
			b.int32(-1)
		} else {
			datum := binaryEncode(&cn.parameterStatus, x)
			b.int32(len(datum))
			b.bytes(datum)
		}
	}
}

func (cn *conn) sendBinaryModeQuery(query string, args []driver.Value) {
	if len(args) >= 65536 {
		errorf("got %d parameters but PostgreSQL only supports 65535 parameters", len(args))
	}

	b := cn.writeBuf('P')
	b.byte(0) // unnamed statement
	b.string(query)
	b.int16(0)

	b.next('B')
	b.int16(0) // unnamed portal and statement
	cn.sendBinaryParameters(b, args)
	b.bytes(colFmtDataAllText)

	b.next('D')
	b.byte('P')
	b.byte(0) // unnamed portal

	b.next('E')
	b.byte(0)
	b.int32(0)

	b.next('S')
	cn.send(b)
}

func (cn *conn) processParameterStatus(r *readBuf) {
	var err error

	param := r.string()
	switch param {
	case "server_version":
		var major1 int
		var major2 int
		var minor int
		_, err = fmt.Sscanf(r.string(), "%d.%d.%d", &major1, &major2, &minor)
		if err == nil {
			cn.parameterStatus.serverVersion = major1*10000 + major2*100 + minor
		}

	case "TimeZone":
		cn.parameterStatus.currentLocation, err = time.LoadLocation(r.string())
		if err != nil {
			cn.parameterStatus.currentLocation = nil
		}

	default:
		// ignore
	}
}

func (cn *conn) processReadyForQuery(r *readBuf) {
	cn.txnStatus = transactionStatus(r.byte())
}

func (cn *conn) readReadyForQuery() {
	t, r := cn.recv1()
	switch t {
	case 'Z':
		cn.processReadyForQuery(r)
		return
	default:
		cn.bad = true
		errorf("unexpected message %q; expected ReadyForQuery", t)
	}
}

func (cn *conn) processBackendKeyData(r *readBuf) {
	cn.processID = r.int32()
	cn.secretKey = r.int32()
}

func (cn *conn) readParseResponse() {
	t, r := cn.recv1()
	switch t {
	case '1':
		return
	case 'E':
		err := parseError(r)
		cn.readReadyForQuery()
		panic(err)
	default:
		cn.bad = true
		errorf("unexpected Parse response %q", t)
	}
}

func (cn *conn) readStatementDescribeResponse() (paramTyps []oid.Oid, colNames []string, colTyps []fieldDesc) {
	for {
		t, r := cn.recv1()
		switch t {
		case 't':
			nparams := r.int16()
			paramTyps = make([]oid.Oid, nparams)
			for i := range paramTyps {
				paramTyps[i] = r.oid()
			}
		case 'n':
			return paramTyps, nil, nil
		case 'T':
			colNames, colTyps = parseStatementRowDescribe(r)
			return paramTyps, colNames, colTyps
		case 'E':
			err := parseError(r)
			cn.readReadyForQuery()
			panic(err)
		default:
			cn.bad = true
			errorf("unexpected Describe statement response %q", t)
		}
	}
}

func (cn *conn) readPortalDescribeResponse() (colNames []string, colFmts []format, colTyps []fieldDesc) {
	t, r := cn.recv1()
	switch t {
	case 'T':
		return parsePortalRowDescribe(r)
	case 'n':
		return nil, nil, nil
	case 'E':
		err := parseError(r)
		cn.readReadyForQuery()
		panic(err)
	default:
		cn.bad = true
		errorf("unexpected Describe response %q", t)
	}
	panic("not reached")
}

func (cn *conn) readBindResponse() {
	t, r := cn.recv1()
	switch t {
	case '2':
		return
	case 'E':
		err := parseError(r)
		cn.readReadyForQuery()
		panic(err)
	default:
		cn.bad = true
		errorf("unexpected Bind response %q", t)
	}
}

func (cn *conn) postExecuteWorkaround() {
	// Work around a bug in sql.DB.QueryRow: in Go 1.2 and earlier it ignores
	// any errors from rows.Next, which masks errors that happened during the
	// execution of the query.  To avoid the problem in common cases, we wait
	// here for one more message from the database.  If it's not an error the
	// query will likely succeed (or perhaps has already, if it's a
	// CommandComplete), so we push the message into the conn struct; recv1
	// will return it as the next message for rows.Next or rows.Close.
	// However, if it's an error, we wait until ReadyForQuery and then return
	// the error to our caller.
	for {
		t, r := cn.recv1()
		switch t {
		case 'E':
			err := parseError(r)
			cn.readReadyForQuery()
			panic(err)
		case 'C', 'D', 'I':
			// the query didn't fail, but we can't process this message
			cn.saveMessage(t, r)
			return
		default:
			cn.bad = true
			errorf("unexpected message during extended query execution: %q", t)
		}
	}
}

// Only for Exec(), since we ignore the returned data
func (cn *conn) readExecuteResponse(protocolState string) (res driver.Result, commandTag string, err error) {
	for {
		t, r := cn.recv1()
		switch t {
		case 'C':
			if err != nil {
				cn.bad = true
				errorf("unexpected CommandComplete after error %s", err)
			}
			res, commandTag = cn.parseComplete(r.string())
		case 'Z':
			cn.processReadyForQuery(r)
			if res == nil && err == nil {
				err = errUnexpectedReady
			}
			return res, commandTag, err
		case 'E':
			err = parseError(r)
		case 'T', 'D', 'I':
			if err != nil {
				cn.bad = true
				errorf("unexpected %q after error %s", t, err)
			}
			if t == 'I' {
				res = emptyRows
			}
			// ignore any results
		default:
			cn.bad = true
			errorf("unknown %s response: %q", protocolState, t)
		}
	}
}

func parseStatementRowDescribe(r *readBuf) (colNames []string, colTyps []fieldDesc) {
	n := r.int16()
	colNames = make([]string, n)
	colTyps = make([]fieldDesc, n)
	for i := range colNames {
		colNames[i] = r.string()
		r.next(6)
		colTyps[i].OID = r.oid()
		colTyps[i].Len = r.int16()
		colTyps[i].Mod = r.int32()
		// format code not known when describing a statement; always 0
		r.next(2)
	}
	return
}

func parsePortalRowDescribe(r *readBuf) (colNames []string, colFmts []format, colTyps []fieldDesc) {
	n := r.int16()
	colNames = make([]string, n)
	colFmts = make([]format, n)
	colTyps = make([]fieldDesc, n)
	for i := range colNames {
		colNames[i] = r.string()
		r.next(6)
		colTyps[i].OID = r.oid()
		colTyps[i].Len = r.int16()
		colTyps[i].Mod = r.int32()
		colFmts[i] = format(r.int16())
	}
	return
}

// parseEnviron tries to mimic some of libpq's environment handling
//
// To ease testing, it does not directly reference os.Environ, but is
// designed to accept its output.
//
// Environment-set connection information is intended to have a higher
// precedence than a library default but lower than any explicitly
// passed information (such as in the URL or connection string).
func parseEnviron(env []string) (out map[string]string) {
	out = make(map[string]string)

	for _, v := range env {
		parts := strings.SplitN(v, "=", 2)

		accrue := func(keyname string) {
			out[keyname] = parts[1]
		}
		unsupported := func() {
			panic(fmt.Sprintf("setting %v not supported", parts[0]))
		}

		// The order of these is the same as is seen in the
		// PostgreSQL 9.1 manual. Unsupported but well-defined
		// keys cause a panic; these should be unset prior to
		// execution. Options which pq expects to be set to a
		// certain value are allowed, but must be set to that
		// value if present (they can, of course, be absent).
		switch parts[0] {
		case "PGHOST":
			accrue("host")
		case "PGHOSTADDR":
			unsupported()
		case "PGPORT":
			accrue("port")
		case "PGDATABASE":
			accrue("dbname")
		case "PGUSER":
			accrue("user")
		case "PGPASSWORD":
			accrue("password")
		case "PGSERVICE", "PGSERVICEFILE", "PGREALM":
			unsupported()
		case "PGOPTIONS":
			accrue("options")
		case "PGAPPNAME":
			accrue("application_name")
		case "PGSSLMODE":
			accrue("sslmode")
		case "PGSSLCERT":
			accrue("sslcert")
		case "PGSSLKEY":
			accrue("sslkey")
		case "PGSSLROOTCERT":
			accrue("sslrootcert")
		case "PGREQUIRESSL", "PGSSLCRL":
			unsupported()
		case "PGREQUIREPEER":
			unsupported()
		case "PGKRBSRVNAME", "PGGSSLIB":
			unsupported()
		case "PGCONNECT_TIMEOUT":
			accrue("connect_timeout")
		case "PGCLIENTENCODING":
			accrue("client_encoding")
		case "PGDATESTYLE":
			accrue("datestyle")
		case "PGTZ":
			accrue("timezone")
		case "PGGEQO":
			accrue("geqo")
		case "PGSYSCONFDIR", "PGLOCALEDIR":
			unsupported()
		}
	}

	return out
}

// isUTF8 returns whether name is a fuzzy variation of the string "UTF-8".
func isUTF8(name string) bool {
	// Recognize all sorts of silly things as "UTF-8", like Postgres does
	s := strings.Map(alnumLowerASCII, name)
	return s == "utf8" || s == "unicode"
}

func alnumLowerASCII(ch rune) rune {
	if 'A' <= ch && ch <= 'Z' {
		return ch + ('a' - 'A')
	}
	if 'a' <= ch && ch <= 'z' || '0' <= ch && ch <= '9' {
		return ch
	}
	return -1 // discard
}
