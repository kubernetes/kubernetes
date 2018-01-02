package main

import (
	"encoding/json"
	"flag"
	"io/ioutil"
	"net"

	"github.com/cloudflare/cfssl/log"
	"github.com/cloudflare/cfssl/transport"
	"github.com/cloudflare/cfssl/transport/core"
	"github.com/cloudflare/cfssl/transport/example/exlib"
)

// maclient is a mutual-authentication server, meant to demonstrate
// using the client-side mutual authentication side of the transport
// package.

func main() {
	var addr, conf string
	flag.StringVar(&addr, "a", "127.0.0.1:9876", "`address` of server")
	flag.StringVar(&conf, "f", "server.json", "config `file` to use")
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

	l, err := transport.Listen(addr, tr)
	if err != nil {
		exlib.Err(1, err, "setting up listener")
	}

	var errChan = make(chan error, 0)
	go func(ec <-chan error) {
		for {
			err, ok := <-ec
			if !ok {
				log.Warning("error channel closed, future errors will not be reported")
				break
			}
			log.Errorf("auto update error: %v", err)
		}
	}(errChan)

	log.Info("setting up auto-update")
	go l.AutoUpdate(nil, errChan)

	log.Info("listening on ", addr)
	exlib.Warn(serve(l), "serving listener")
}

func connHandler(conn net.Conn) {
	defer conn.Close()

	for {
		buf, err := exlib.Unpack(conn)
		if err != nil {
			exlib.Warn(err, "unpack message")
			return
		}

		if len(buf) == 0 {
			log.Info(conn.RemoteAddr(), " sent empty record, closing connection")
			return
		}

		log.Infof("received %d-byte message: %s", len(buf), buf)

		err = exlib.Pack(conn, []byte("OK"))
		if err != nil {
			exlib.Warn(err, "pack message")
			return
		}
	}
}

func serve(l net.Listener) error {
	defer l.Close()
	for {
		conn, err := l.Accept()
		if err != nil {
			exlib.Warn(err, "client connection failed")
			continue
		}

		log.Info("connection from ", conn.RemoteAddr())
		go connHandler(conn)
	}
}
