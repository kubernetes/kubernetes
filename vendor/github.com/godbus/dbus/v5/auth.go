package dbus

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"os"
	"strconv"
)

// AuthStatus represents the Status of an authentication mechanism.
type AuthStatus byte

const (
	// AuthOk signals that authentication is finished; the next command
	// from the server should be an OK.
	AuthOk AuthStatus = iota

	// AuthContinue signals that additional data is needed; the next command
	// from the server should be a DATA.
	AuthContinue

	// AuthError signals an error; the server sent invalid data or some
	// other unexpected thing happened and the current authentication
	// process should be aborted.
	AuthError
)

type authState byte

const (
	waitingForData authState = iota
	waitingForOk
	waitingForReject
)

// Auth defines the behaviour of an authentication mechanism.
type Auth interface {
	// Return the name of the mechanism, the argument to the first AUTH command
	// and the next status.
	FirstData() (name, resp []byte, status AuthStatus)

	// Process the given DATA command, and return the argument to the DATA
	// command and the next status. If len(resp) == 0, no DATA command is sent.
	HandleData(data []byte) (resp []byte, status AuthStatus)
}

// Auth authenticates the connection, trying the given list of authentication
// mechanisms (in that order). If nil is passed, the EXTERNAL and
// DBUS_COOKIE_SHA1 mechanisms are tried for the current user. For private
// connections, this method must be called before sending any messages to the
// bus. Auth must not be called on shared connections.
func (conn *Conn) Auth(methods []Auth) error {
	if methods == nil {
		uid := strconv.Itoa(os.Getuid())
		methods = []Auth{AuthExternal(uid), AuthCookieSha1(uid, getHomeDir())}
	}
	in := bufio.NewReader(conn.transport)
	err := conn.transport.SendNullByte()
	if err != nil {
		return err
	}
	err = authWriteLine(conn.transport, []byte("AUTH"))
	if err != nil {
		return err
	}
	s, err := authReadLine(in)
	if err != nil {
		return err
	}
	if len(s) < 2 || !bytes.Equal(s[0], []byte("REJECTED")) {
		return errors.New("dbus: authentication protocol error")
	}
	s = s[1:]
	for _, v := range s {
		for _, m := range methods {
			if name, data, status := m.FirstData(); bytes.Equal(v, name) {
				var ok bool
				err = authWriteLine(conn.transport, []byte("AUTH"), v, data)
				if err != nil {
					return err
				}
				switch status {
				case AuthOk:
					err, ok = conn.tryAuth(m, waitingForOk, in)
				case AuthContinue:
					err, ok = conn.tryAuth(m, waitingForData, in)
				default:
					panic("dbus: invalid authentication status")
				}
				if err != nil {
					return err
				}
				if ok {
					if conn.transport.SupportsUnixFDs() {
						err = authWriteLine(conn, []byte("NEGOTIATE_UNIX_FD"))
						if err != nil {
							return err
						}
						line, err := authReadLine(in)
						if err != nil {
							return err
						}
						switch {
						case bytes.Equal(line[0], []byte("AGREE_UNIX_FD")):
							conn.EnableUnixFDs()
							conn.unixFD = true
						case bytes.Equal(line[0], []byte("ERROR")):
						default:
							return errors.New("dbus: authentication protocol error")
						}
					}
					err = authWriteLine(conn.transport, []byte("BEGIN"))
					if err != nil {
						return err
					}
					go conn.inWorker()
					return nil
				}
			}
		}
	}
	return errors.New("dbus: authentication failed")
}

// tryAuth tries to authenticate with m as the mechanism, using state as the
// initial authState and in for reading input. It returns (nil, true) on
// success, (nil, false) on a REJECTED and (someErr, false) if some other
// error occurred.
func (conn *Conn) tryAuth(m Auth, state authState, in *bufio.Reader) (error, bool) {
	for {
		s, err := authReadLine(in)
		if err != nil {
			return err, false
		}
		switch {
		case state == waitingForData && string(s[0]) == "DATA":
			if len(s) != 2 {
				err = authWriteLine(conn.transport, []byte("ERROR"))
				if err != nil {
					return err, false
				}
				continue
			}
			data, status := m.HandleData(s[1])
			switch status {
			case AuthOk, AuthContinue:
				if len(data) != 0 {
					err = authWriteLine(conn.transport, []byte("DATA"), data)
					if err != nil {
						return err, false
					}
				}
				if status == AuthOk {
					state = waitingForOk
				}
			case AuthError:
				err = authWriteLine(conn.transport, []byte("ERROR"))
				if err != nil {
					return err, false
				}
			}
		case state == waitingForData && string(s[0]) == "REJECTED":
			return nil, false
		case state == waitingForData && string(s[0]) == "ERROR":
			err = authWriteLine(conn.transport, []byte("CANCEL"))
			if err != nil {
				return err, false
			}
			state = waitingForReject
		case state == waitingForData && string(s[0]) == "OK":
			if len(s) != 2 {
				err = authWriteLine(conn.transport, []byte("CANCEL"))
				if err != nil {
					return err, false
				}
				state = waitingForReject
			}
			conn.uuid = string(s[1])
			return nil, true
		case state == waitingForData:
			err = authWriteLine(conn.transport, []byte("ERROR"))
			if err != nil {
				return err, false
			}
		case state == waitingForOk && string(s[0]) == "OK":
			if len(s) != 2 {
				err = authWriteLine(conn.transport, []byte("CANCEL"))
				if err != nil {
					return err, false
				}
				state = waitingForReject
			}
			conn.uuid = string(s[1])
			return nil, true
		case state == waitingForOk && string(s[0]) == "REJECTED":
			return nil, false
		case state == waitingForOk && (string(s[0]) == "DATA" ||
			string(s[0]) == "ERROR"):

			err = authWriteLine(conn.transport, []byte("CANCEL"))
			if err != nil {
				return err, false
			}
			state = waitingForReject
		case state == waitingForOk:
			err = authWriteLine(conn.transport, []byte("ERROR"))
			if err != nil {
				return err, false
			}
		case state == waitingForReject && string(s[0]) == "REJECTED":
			return nil, false
		case state == waitingForReject:
			return errors.New("dbus: authentication protocol error"), false
		default:
			panic("dbus: invalid auth state")
		}
	}
}

// authReadLine reads a line and separates it into its fields.
func authReadLine(in *bufio.Reader) ([][]byte, error) {
	data, err := in.ReadBytes('\n')
	if err != nil {
		return nil, err
	}
	data = bytes.TrimSuffix(data, []byte("\r\n"))
	return bytes.Split(data, []byte{' '}), nil
}

// authWriteLine writes the given line in the authentication protocol format
// (elements of data separated by a " " and terminated by "\r\n").
func authWriteLine(out io.Writer, data ...[]byte) error {
	buf := make([]byte, 0)
	for i, v := range data {
		buf = append(buf, v...)
		if i != len(data)-1 {
			buf = append(buf, ' ')
		}
	}
	buf = append(buf, '\r')
	buf = append(buf, '\n')
	n, err := out.Write(buf)
	if err != nil {
		return err
	}
	if n != len(buf) {
		return io.ErrUnexpectedEOF
	}
	return nil
}
