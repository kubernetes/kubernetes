//go:build js
// +build js

// Package wsjs implements typed access to the browser javascript WebSocket API.
//
// https://developer.mozilla.org/en-US/docs/Web/API/WebSocket
package wsjs

import (
	"syscall/js"
)

func handleJSError(err *error, onErr func()) {
	r := recover()

	if jsErr, ok := r.(js.Error); ok {
		*err = jsErr

		if onErr != nil {
			onErr()
		}
		return
	}

	if r != nil {
		panic(r)
	}
}

// New is a wrapper around the javascript WebSocket constructor.
func New(url string, protocols []string) (c WebSocket, err error) {
	defer handleJSError(&err, func() {
		c = WebSocket{}
	})

	jsProtocols := make([]any, len(protocols))
	for i, p := range protocols {
		jsProtocols[i] = p
	}

	c = WebSocket{
		v: js.Global().Get("WebSocket").New(url, jsProtocols),
	}

	c.setBinaryType("arraybuffer")

	return c, nil
}

// WebSocket is a wrapper around a javascript WebSocket object.
type WebSocket struct {
	v js.Value
}

func (c WebSocket) setBinaryType(typ string) {
	c.v.Set("binaryType", string(typ))
}

func (c WebSocket) addEventListener(eventType string, fn func(e js.Value)) func() {
	f := js.FuncOf(func(this js.Value, args []js.Value) any {
		fn(args[0])
		return nil
	})
	c.v.Call("addEventListener", eventType, f)

	return func() {
		c.v.Call("removeEventListener", eventType, f)
		f.Release()
	}
}

// CloseEvent is the type passed to a WebSocket close handler.
type CloseEvent struct {
	Code     uint16
	Reason   string
	WasClean bool
}

// OnClose registers a function to be called when the WebSocket is closed.
func (c WebSocket) OnClose(fn func(CloseEvent)) (remove func()) {
	return c.addEventListener("close", func(e js.Value) {
		ce := CloseEvent{
			Code:     uint16(e.Get("code").Int()),
			Reason:   e.Get("reason").String(),
			WasClean: e.Get("wasClean").Bool(),
		}
		fn(ce)
	})
}

// OnError registers a function to be called when there is an error
// with the WebSocket.
func (c WebSocket) OnError(fn func(e js.Value)) (remove func()) {
	return c.addEventListener("error", fn)
}

// MessageEvent is the type passed to a message handler.
type MessageEvent struct {
	// string or []byte.
	Data any

	// There are more fields to the interface but we don't use them.
	// See https://developer.mozilla.org/en-US/docs/Web/API/MessageEvent
}

// OnMessage registers a function to be called when the WebSocket receives a message.
func (c WebSocket) OnMessage(fn func(m MessageEvent)) (remove func()) {
	return c.addEventListener("message", func(e js.Value) {
		var data any

		arrayBuffer := e.Get("data")
		if arrayBuffer.Type() == js.TypeString {
			data = arrayBuffer.String()
		} else {
			data = extractArrayBuffer(arrayBuffer)
		}

		me := MessageEvent{
			Data: data,
		}
		fn(me)
	})
}

// Subprotocol returns the WebSocket subprotocol in use.
func (c WebSocket) Subprotocol() string {
	return c.v.Get("protocol").String()
}

// OnOpen registers a function to be called when the WebSocket is opened.
func (c WebSocket) OnOpen(fn func(e js.Value)) (remove func()) {
	return c.addEventListener("open", fn)
}

// Close closes the WebSocket with the given code and reason.
func (c WebSocket) Close(code int, reason string) (err error) {
	defer handleJSError(&err, nil)
	c.v.Call("close", code, reason)
	return err
}

// SendText sends the given string as a text message
// on the WebSocket.
func (c WebSocket) SendText(v string) (err error) {
	defer handleJSError(&err, nil)
	c.v.Call("send", v)
	return err
}

// SendBytes sends the given message as a binary message
// on the WebSocket.
func (c WebSocket) SendBytes(v []byte) (err error) {
	defer handleJSError(&err, nil)
	c.v.Call("send", uint8Array(v))
	return err
}

func extractArrayBuffer(arrayBuffer js.Value) []byte {
	uint8Array := js.Global().Get("Uint8Array").New(arrayBuffer)
	dst := make([]byte, uint8Array.Length())
	js.CopyBytesToGo(dst, uint8Array)
	return dst
}

func uint8Array(src []byte) js.Value {
	uint8Array := js.Global().Get("Uint8Array").New(len(src))
	js.CopyBytesToJS(uint8Array, src)
	return uint8Array
}
