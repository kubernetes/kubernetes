package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"

	"github.com/cloudflare/cfssl/transport"
	"github.com/cloudflare/cfssl/transport/core"
	"github.com/cloudflare/cfssl/transport/example/exlib"
)

// maclient is a mutual-authentication client, meant to demonstrate
// using the client-side mutual authentication side of the transport
// package.

var messages = []string{"hello world", "hello", "world"}

func main() {
	var addr, conf string
	flag.StringVar(&addr, "a", "127.0.0.1:9876", "`address` of server")
	flag.StringVar(&conf, "f", "client.json", "config `file` to use")
	flag.Parse()

	var id = new(core.Identity)
	data, err := ioutil.ReadFile(conf)
	if err != nil {
		exlib.Err(1, err, "reading config file")
	}

	err = json.Unmarshal(data, id)
	if err != nil {
		exlib.Err(1, err, "parsing config file")
	}

	tr, err := transport.New(exlib.Before, id)
	if err != nil {
		exlib.Err(1, err, "creating transport")
	}

	conn, err := transport.Dial(addr, tr)
	if err != nil {
		exlib.Err(1, err, "dialing %s", addr)
	}
	defer conn.Close()

	for _, msg := range messages {
		if err = exlib.Pack(conn, []byte(msg)); err != nil {
			exlib.Err(1, err, "sending message")
		}

		var resp []byte
		resp, err = exlib.Unpack(conn)
		if err != nil {
			exlib.Err(1, err, "receiving message")
		}

		if !bytes.Equal(resp, []byte("OK")) {
			exlib.Errx(1, "server didn't send an OK message; received '%s'", resp)
		}
	}

	err = exlib.Pack(conn, []byte{})
	if err != nil {
		exlib.Err(1, err, "sending shutdown message failed")
	}

	fmt.Println("OK")
}
