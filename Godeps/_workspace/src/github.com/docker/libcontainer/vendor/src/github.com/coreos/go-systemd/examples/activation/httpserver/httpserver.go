package main

import (
	"io"
	"net/http"

	"github.com/coreos/go-systemd/activation"
)

func HelloServer(w http.ResponseWriter, req *http.Request) {
	io.WriteString(w, "hello socket activated world!\n")
}

func main() {
	listeners, err := activation.Listeners(true)
	if err != nil {
		panic(err)
	}

	if len(listeners) != 1 {
		panic("Unexpected number of socket activation fds")
	}

	http.HandleFunc("/", HelloServer)
	http.Serve(listeners[0], nil)
}
