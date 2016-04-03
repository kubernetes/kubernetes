package listenbuffer

import (
	"io/ioutil"
	"net"
	"testing"
)

func TestListenBufferAllowsAcceptingWhenActivated(t *testing.T) {
	lock := make(chan struct{})
	buffer, err := NewListenBuffer("tcp", "", lock)
	if err != nil {
		t.Fatal("Unable to create listen buffer: ", err)
	}

	go func() {
		conn, err := net.Dial("tcp", buffer.Addr().String())
		if err != nil {
			t.Fatal("Client failed to establish connection to server: ", err)
		}

		conn.Write([]byte("ping"))
		conn.Close()
	}()

	close(lock)

	client, err := buffer.Accept()
	if err != nil {
		t.Fatal("Failed to accept client: ", err)
	}

	response, err := ioutil.ReadAll(client)
	if err != nil {
		t.Fatal("Failed to read from client: ", err)
	}

	if string(response) != "ping" {
		t.Fatal("Expected to receive ping from client, received: ", string(response))
	}
}
