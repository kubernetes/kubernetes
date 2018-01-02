// Copyright 2013 The Gorilla WebSocket Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command server is a test server for the Autobahn WebSockets Test Suite.
package main

import (
	"errors"
	"flag"
	"github.com/gorilla/websocket"
	"io"
	"log"
	"net/http"
	"time"
	"unicode/utf8"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  4096,
	WriteBufferSize: 4096,
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

// echoCopy echoes messages from the client using io.Copy.
func echoCopy(w http.ResponseWriter, r *http.Request, writerOnly bool) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("Upgrade:", err)
		return
	}
	defer conn.Close()
	for {
		mt, r, err := conn.NextReader()
		if err != nil {
			if err != io.EOF {
				log.Println("NextReader:", err)
			}
			return
		}
		if mt == websocket.TextMessage {
			r = &validator{r: r}
		}
		w, err := conn.NextWriter(mt)
		if err != nil {
			log.Println("NextWriter:", err)
			return
		}
		if mt == websocket.TextMessage {
			r = &validator{r: r}
		}
		if writerOnly {
			_, err = io.Copy(struct{ io.Writer }{w}, r)
		} else {
			_, err = io.Copy(w, r)
		}
		if err != nil {
			if err == errInvalidUTF8 {
				conn.WriteControl(websocket.CloseMessage,
					websocket.FormatCloseMessage(websocket.CloseInvalidFramePayloadData, ""),
					time.Time{})
			}
			log.Println("Copy:", err)
			return
		}
		err = w.Close()
		if err != nil {
			log.Println("Close:", err)
			return
		}
	}
}

func echoCopyWriterOnly(w http.ResponseWriter, r *http.Request) {
	echoCopy(w, r, true)
}

func echoCopyFull(w http.ResponseWriter, r *http.Request) {
	echoCopy(w, r, false)
}

// echoReadAll echoes messages from the client by reading the entire message
// with ioutil.ReadAll.
func echoReadAll(w http.ResponseWriter, r *http.Request, writeMessage bool) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("Upgrade:", err)
		return
	}
	defer conn.Close()
	for {
		mt, b, err := conn.ReadMessage()
		if err != nil {
			if err != io.EOF {
				log.Println("NextReader:", err)
			}
			return
		}
		if mt == websocket.TextMessage {
			if !utf8.Valid(b) {
				conn.WriteControl(websocket.CloseMessage,
					websocket.FormatCloseMessage(websocket.CloseInvalidFramePayloadData, ""),
					time.Time{})
				log.Println("ReadAll: invalid utf8")
			}
		}
		if writeMessage {
			err = conn.WriteMessage(mt, b)
			if err != nil {
				log.Println("WriteMessage:", err)
			}
		} else {
			w, err := conn.NextWriter(mt)
			if err != nil {
				log.Println("NextWriter:", err)
				return
			}
			if _, err := w.Write(b); err != nil {
				log.Println("Writer:", err)
				return
			}
			if err := w.Close(); err != nil {
				log.Println("Close:", err)
				return
			}
		}
	}
}

func echoReadAllWriter(w http.ResponseWriter, r *http.Request) {
	echoReadAll(w, r, false)
}

func echoReadAllWriteMessage(w http.ResponseWriter, r *http.Request) {
	echoReadAll(w, r, true)
}

func serveHome(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.Error(w, "Not found.", 404)
		return
	}
	if r.Method != "GET" {
		http.Error(w, "Method not allowed", 405)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	io.WriteString(w, "<html><body>Echo Server</body></html>")
}

var addr = flag.String("addr", ":9000", "http service address")

func main() {
	flag.Parse()
	http.HandleFunc("/", serveHome)
	http.HandleFunc("/c", echoCopyWriterOnly)
	http.HandleFunc("/f", echoCopyFull)
	http.HandleFunc("/r", echoReadAllWriter)
	http.HandleFunc("/m", echoReadAllWriteMessage)
	err := http.ListenAndServe(*addr, nil)
	if err != nil {
		log.Fatal("ListenAndServe: ", err)
	}
}

type validator struct {
	state int
	x     rune
	r     io.Reader
}

var errInvalidUTF8 = errors.New("invalid utf8")

func (r *validator) Read(p []byte) (int, error) {
	n, err := r.r.Read(p)
	state := r.state
	x := r.x
	for _, b := range p[:n] {
		state, x = decode(state, x, b)
		if state == utf8Reject {
			break
		}
	}
	r.state = state
	r.x = x
	if state == utf8Reject || (err == io.EOF && state != utf8Accept) {
		return n, errInvalidUTF8
	}
	return n, err
}

// UTF-8 decoder from http://bjoern.hoehrmann.de/utf-8/decoder/dfa/
//
// Copyright (c) 2008-2009 Bjoern Hoehrmann <bjoern@hoehrmann.de>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
var utf8d = [...]byte{
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 00..1f
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 20..3f
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 40..5f
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 60..7f
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 80..9f
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, // a0..bf
	8, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // c0..df
	0xa, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x4, 0x3, 0x3, // e0..ef
	0xb, 0x6, 0x6, 0x6, 0x5, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, // f0..ff
	0x0, 0x1, 0x2, 0x3, 0x5, 0x8, 0x7, 0x1, 0x1, 0x1, 0x4, 0x6, 0x1, 0x1, 0x1, 0x1, // s0..s0
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, // s1..s2
	1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, // s3..s4
	1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, // s5..s6
	1, 3, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // s7..s8
}

const (
	utf8Accept = 0
	utf8Reject = 1
)

func decode(state int, x rune, b byte) (int, rune) {
	t := utf8d[b]
	if state != utf8Accept {
		x = rune(b&0x3f) | (x << 6)
	} else {
		x = rune((0xff >> t) & b)
	}
	state = int(utf8d[256+state*16+int(t)])
	return state, x
}
