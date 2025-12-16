//go:build !js

package websocket

import (
	"bytes"
	"context"
	"crypto/sha1"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/textproto"
	"net/url"
	"path"
	"strings"

	"github.com/coder/websocket/internal/errd"
)

// AcceptOptions represents Accept's options.
type AcceptOptions struct {
	// Subprotocols lists the WebSocket subprotocols that Accept will negotiate with the client.
	// The empty subprotocol will always be negotiated as per RFC 6455. If you would like to
	// reject it, close the connection when c.Subprotocol() == "".
	Subprotocols []string

	// InsecureSkipVerify is used to disable Accept's origin verification behaviour.
	//
	// You probably want to use OriginPatterns instead.
	InsecureSkipVerify bool

	// OriginPatterns lists the host patterns for authorized origins.
	// The request host is always authorized.
	// Use this to enable cross origin WebSockets.
	//
	// i.e javascript running on example.com wants to access a WebSocket server at chat.example.com.
	// In such a case, example.com is the origin and chat.example.com is the request host.
	// One would set this field to []string{"example.com"} to authorize example.com to connect.
	//
	// Each pattern is matched case insensitively with path.Match (see
	// https://golang.org/pkg/path/#Match). By default, it is matched
	// against the request origin host. If the pattern contains a URI
	// scheme ("://"), it will be matched against "scheme://host".
	//
	// Please ensure you understand the ramifications of enabling this.
	// If used incorrectly your WebSocket server will be open to CSRF attacks.
	//
	// Do not use * as a pattern to allow any origin, prefer to use InsecureSkipVerify instead
	// to bring attention to the danger of such a setting.
	OriginPatterns []string

	// CompressionMode controls the compression mode.
	// Defaults to CompressionDisabled.
	//
	// See docs on CompressionMode for details.
	CompressionMode CompressionMode

	// CompressionThreshold controls the minimum size of a message before compression is applied.
	//
	// Defaults to 512 bytes for CompressionNoContextTakeover and 128 bytes
	// for CompressionContextTakeover.
	CompressionThreshold int

	// OnPingReceived is an optional callback invoked synchronously when a ping frame is received.
	//
	// The payload contains the application data of the ping frame.
	// If the callback returns false, the subsequent pong frame will not be sent.
	// To avoid blocking, any expensive processing should be performed asynchronously using a goroutine.
	OnPingReceived func(ctx context.Context, payload []byte) bool

	// OnPongReceived is an optional callback invoked synchronously when a pong frame is received.
	//
	// The payload contains the application data of the pong frame.
	// To avoid blocking, any expensive processing should be performed asynchronously using a goroutine.
	//
	// Unlike OnPingReceived, this callback does not return a value because a pong frame
	// is a response to a ping and does not trigger any further frame transmission.
	OnPongReceived func(ctx context.Context, payload []byte)
}

func (opts *AcceptOptions) cloneWithDefaults() *AcceptOptions {
	var o AcceptOptions
	if opts != nil {
		o = *opts
	}
	return &o
}

// Accept accepts a WebSocket handshake from a client and upgrades the
// the connection to a WebSocket.
//
// Accept will not allow cross origin requests by default.
// See the InsecureSkipVerify and OriginPatterns options to allow cross origin requests.
//
// Accept will write a response to w on all errors.
//
// Note that using the http.Request Context after Accept returns may lead to
// unexpected behavior (see http.Hijacker).
func Accept(w http.ResponseWriter, r *http.Request, opts *AcceptOptions) (*Conn, error) {
	return accept(w, r, opts)
}

func accept(w http.ResponseWriter, r *http.Request, opts *AcceptOptions) (_ *Conn, err error) {
	defer errd.Wrap(&err, "failed to accept WebSocket connection")

	errCode, err := verifyClientRequest(w, r)
	if err != nil {
		http.Error(w, err.Error(), errCode)
		return nil, err
	}

	opts = opts.cloneWithDefaults()
	if !opts.InsecureSkipVerify {
		err = authenticateOrigin(r, opts.OriginPatterns)
		if err != nil {
			if errors.Is(err, path.ErrBadPattern) {
				log.Printf("websocket: %v", err)
				err = errors.New(http.StatusText(http.StatusForbidden))
			}
			http.Error(w, err.Error(), http.StatusForbidden)
			return nil, err
		}
	}

	hj, ok := hijacker(w)
	if !ok {
		err = errors.New("http.ResponseWriter does not implement http.Hijacker")
		http.Error(w, http.StatusText(http.StatusNotImplemented), http.StatusNotImplemented)
		return nil, err
	}

	w.Header().Set("Upgrade", "websocket")
	w.Header().Set("Connection", "Upgrade")

	key := r.Header.Get("Sec-WebSocket-Key")
	w.Header().Set("Sec-WebSocket-Accept", secWebSocketAccept(key))

	subproto := selectSubprotocol(r, opts.Subprotocols)
	if subproto != "" {
		w.Header().Set("Sec-WebSocket-Protocol", subproto)
	}

	copts, ok := selectDeflate(websocketExtensions(r.Header), opts.CompressionMode)
	if ok {
		w.Header().Set("Sec-WebSocket-Extensions", copts.String())
	}

	w.WriteHeader(http.StatusSwitchingProtocols)
	// See https://github.com/nhooyr/websocket/issues/166
	if ginWriter, ok := w.(interface {
		WriteHeaderNow()
	}); ok {
		ginWriter.WriteHeaderNow()
	}

	netConn, brw, err := hj.Hijack()
	if err != nil {
		err = fmt.Errorf("failed to hijack connection: %w", err)
		http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
		return nil, err
	}

	// https://github.com/golang/go/issues/32314
	b, _ := brw.Reader.Peek(brw.Reader.Buffered())
	brw.Reader.Reset(io.MultiReader(bytes.NewReader(b), netConn))

	return newConn(connConfig{
		subprotocol:    w.Header().Get("Sec-WebSocket-Protocol"),
		rwc:            netConn,
		client:         false,
		copts:          copts,
		flateThreshold: opts.CompressionThreshold,
		onPingReceived: opts.OnPingReceived,
		onPongReceived: opts.OnPongReceived,

		br: brw.Reader,
		bw: brw.Writer,
	}), nil
}

func verifyClientRequest(w http.ResponseWriter, r *http.Request) (errCode int, _ error) {
	if !r.ProtoAtLeast(1, 1) {
		return http.StatusUpgradeRequired, fmt.Errorf("WebSocket protocol violation: handshake request must be at least HTTP/1.1: %q", r.Proto)
	}

	if !headerContainsTokenIgnoreCase(r.Header, "Connection", "Upgrade") {
		w.Header().Set("Connection", "Upgrade")
		w.Header().Set("Upgrade", "websocket")
		return http.StatusUpgradeRequired, fmt.Errorf("WebSocket protocol violation: Connection header %q does not contain Upgrade", r.Header.Get("Connection"))
	}

	if !headerContainsTokenIgnoreCase(r.Header, "Upgrade", "websocket") {
		w.Header().Set("Connection", "Upgrade")
		w.Header().Set("Upgrade", "websocket")
		return http.StatusUpgradeRequired, fmt.Errorf("WebSocket protocol violation: Upgrade header %q does not contain websocket", r.Header.Get("Upgrade"))
	}

	if r.Method != "GET" {
		return http.StatusMethodNotAllowed, fmt.Errorf("WebSocket protocol violation: handshake request method is not GET but %q", r.Method)
	}

	if r.Header.Get("Sec-WebSocket-Version") != "13" {
		w.Header().Set("Sec-WebSocket-Version", "13")
		return http.StatusBadRequest, fmt.Errorf("unsupported WebSocket protocol version (only 13 is supported): %q", r.Header.Get("Sec-WebSocket-Version"))
	}

	websocketSecKeys := r.Header.Values("Sec-WebSocket-Key")
	if len(websocketSecKeys) == 0 {
		return http.StatusBadRequest, errors.New("WebSocket protocol violation: missing Sec-WebSocket-Key")
	}

	if len(websocketSecKeys) > 1 {
		return http.StatusBadRequest, errors.New("WebSocket protocol violation: multiple Sec-WebSocket-Key headers")
	}

	// The RFC states to remove any leading or trailing whitespace.
	websocketSecKey := strings.TrimSpace(websocketSecKeys[0])
	if v, err := base64.StdEncoding.DecodeString(websocketSecKey); err != nil || len(v) != 16 {
		return http.StatusBadRequest, fmt.Errorf("WebSocket protocol violation: invalid Sec-WebSocket-Key %q, must be a 16 byte base64 encoded string", websocketSecKey)
	}

	return 0, nil
}

func authenticateOrigin(r *http.Request, originHosts []string) error {
	origin := r.Header.Get("Origin")
	if origin == "" {
		return nil
	}

	u, err := url.Parse(origin)
	if err != nil {
		return fmt.Errorf("failed to parse Origin header %q: %w", origin, err)
	}

	if strings.EqualFold(r.Host, u.Host) {
		return nil
	}

	for _, hostPattern := range originHosts {
		target := u.Host
		if strings.Contains(hostPattern, "://") {
			target = u.Scheme + "://" + u.Host
		}
		matched, err := match(hostPattern, target)
		if err != nil {
			return fmt.Errorf("failed to parse path pattern %q: %w", hostPattern, err)
		}
		if matched {
			return nil
		}
	}
	if u.Host == "" {
		return fmt.Errorf("request Origin %q is not a valid URL with a host", origin)
	}
	return fmt.Errorf("request Origin %q is not authorized for Host %q", u.Host, r.Host)
}

func match(pattern, s string) (bool, error) {
	return path.Match(strings.ToLower(pattern), strings.ToLower(s))
}

func selectSubprotocol(r *http.Request, subprotocols []string) string {
	cps := headerTokens(r.Header, "Sec-WebSocket-Protocol")
	for _, sp := range subprotocols {
		for _, cp := range cps {
			if strings.EqualFold(sp, cp) {
				return cp
			}
		}
	}
	return ""
}

func selectDeflate(extensions []websocketExtension, mode CompressionMode) (*compressionOptions, bool) {
	if mode == CompressionDisabled {
		return nil, false
	}
	for _, ext := range extensions {
		switch ext.name {
		// We used to implement x-webkit-deflate-frame too for Safari but Safari has bugs...
		// See https://github.com/nhooyr/websocket/issues/218
		case "permessage-deflate":
			copts, ok := acceptDeflate(ext, mode)
			if ok {
				return copts, true
			}
		}
	}
	return nil, false
}

func acceptDeflate(ext websocketExtension, mode CompressionMode) (*compressionOptions, bool) {
	copts := mode.opts()
	for _, p := range ext.params {
		switch p {
		case "client_no_context_takeover":
			copts.clientNoContextTakeover = true
			continue
		case "server_no_context_takeover":
			copts.serverNoContextTakeover = true
			continue
		case "client_max_window_bits",
			"server_max_window_bits=15":
			continue
		}

		if strings.HasPrefix(p, "client_max_window_bits=") {
			// We can't adjust the deflate window, but decoding with a larger window is acceptable.
			continue
		}
		return nil, false
	}
	return copts, true
}

func headerContainsTokenIgnoreCase(h http.Header, key, token string) bool {
	for _, t := range headerTokens(h, key) {
		if strings.EqualFold(t, token) {
			return true
		}
	}
	return false
}

type websocketExtension struct {
	name   string
	params []string
}

func websocketExtensions(h http.Header) []websocketExtension {
	var exts []websocketExtension
	extStrs := headerTokens(h, "Sec-WebSocket-Extensions")
	for _, extStr := range extStrs {
		if extStr == "" {
			continue
		}

		vals := strings.Split(extStr, ";")
		for i := range vals {
			vals[i] = strings.TrimSpace(vals[i])
		}

		e := websocketExtension{
			name:   vals[0],
			params: vals[1:],
		}

		exts = append(exts, e)
	}
	return exts
}

func headerTokens(h http.Header, key string) []string {
	key = textproto.CanonicalMIMEHeaderKey(key)
	var tokens []string
	for _, v := range h[key] {
		v = strings.TrimSpace(v)
		for _, t := range strings.Split(v, ",") {
			t = strings.TrimSpace(t)
			tokens = append(tokens, t)
		}
	}
	return tokens
}

var keyGUID = []byte("258EAFA5-E914-47DA-95CA-C5AB0DC85B11")

func secWebSocketAccept(secWebSocketKey string) string {
	h := sha1.New()
	h.Write([]byte(secWebSocketKey))
	h.Write(keyGUID)

	return base64.StdEncoding.EncodeToString(h.Sum(nil))
}
