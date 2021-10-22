package csm

import (
	"encoding/json"
	"fmt"
	"net"
	"testing"
)

func startUDPServer(done chan struct{}, fn func([]byte)) (string, error) {
	addr, err := net.ResolveUDPAddr("udp", "127.0.0.1:0")
	if err != nil {
		return "", err
	}

	conn, err := net.ListenUDP("udp", addr)
	if err != nil {
		return "", err
	}

	buf := make([]byte, 1024)
	go func() {
		defer conn.Close()

		for {
			select {
			case <-done:
				return
			default:
			}

			n, _, err := conn.ReadFromUDP(buf)
			fn(buf[:n])

			if err != nil {
				panic(err)
			}
		}
	}()

	return conn.LocalAddr().String(), nil
}

func TestDifferentParams(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic with different parameters")
		}
	}()
	Start("clientID2", ":0")
}

var MetricsCh = make(chan map[string]interface{}, 1)
var Done = make(chan struct{})

func init() {
	url, err := startUDPServer(Done, func(b []byte) {
		m := map[string]interface{}{}
		if err := json.Unmarshal(b, &m); err != nil {
			panic(fmt.Sprintf("expected no error, but received %v", err))
		}

		MetricsCh <- m
	})

	if err != nil {
		panic(err)
	}

	_, err = Start("clientID", url)
	if err != nil {
		panic(err)
	}
}
