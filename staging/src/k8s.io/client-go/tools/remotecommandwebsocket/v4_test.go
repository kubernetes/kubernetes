package remotecommandwebsocket

import (
	"bufio"
	"encoding/base64"
	"io"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func TestV4Binary(t *testing.T) {

	m := http.NewServeMux()
	s := &testServer{
		httpServer:   &http.Server{Addr: "127.0.0.1:8765", Handler: m},
		t:            t,
		wg:           &sync.WaitGroup{},
		stderrPassed: false,
		stdinPassed:  false,
		stdoutPassed: false,
	}
	m.HandleFunc("/wsbinary", s.wsBinaryv4)

	go s.httpServer.ListenAndServe()

	time.Sleep(2 * time.Second)
	s.isBinary = true
	runTestCasev4(t, true, true, true, s)
	runTestCasev4(t, true, true, false, s)
	runTestCasev4(t, true, false, true, s)
	runTestCasev4(t, false, true, true, s)
	runTestCasev4(t, false, false, true, s)
	runTestCasev4(t, true, false, false, s)
	runTestCasev4(t, false, true, false, s)
	runTestCasev4(t, false, false, false, s)

	s.isBinary = false
	runTestCasev4(t, true, true, true, s)
	runTestCasev4(t, true, true, false, s)
	runTestCasev4(t, true, false, true, s)
	runTestCasev4(t, false, true, true, s)
	runTestCasev4(t, false, false, true, s)
	runTestCasev4(t, true, false, false, s)
	runTestCasev4(t, false, true, false, s)
	runTestCasev4(t, false, false, false, s)

}

func runTestCasev4(t *testing.T, runStdIn bool, runStdOut bool, runStdErr bool, s *testServer) {
	t.Logf("Test Case - stdin : %t / stdout: %t / stederr : %t", runStdIn, runStdOut, runStdErr)

	s.stdinPassed = false
	s.stdoutPassed = false
	s.stderrPassed = false

	var stdinin, stdoutin, stderrin io.Reader
	var stdoutout, stderrout, stdinout io.Writer

	if runStdIn {
		stdinin, stdinout = io.Pipe()
	}

	if runStdOut {
		stdoutin, stdoutout = io.Pipe()
	}

	if runStdErr {
		stderrin, stderrout = io.Pipe()
	}

	s.streamOptions = &StreamOptions{
		Stdin:             stdinin,
		Stdout:            stdoutout,
		Stderr:            stderrout,
		Tty:               false,
		TerminalSizeQueue: nil,
	}

	conn, _, err := websocket.DefaultDialer.Dial("ws://127.0.0.1:8765/wsbinary", nil)
	s.wsClient = conn
	if err != nil {
		panic(err)
	}

	var streamer streamProtocolHandler
	if s.isBinary {
		streamer = newBinaryV4(*s.streamOptions)
	} else {
		streamer = newBase64V4(*s.streamOptions)
	}
	go streamer.stream(conn)

	if s.streamOptions.Stdin != nil {
		go func() {
			// write to standard in
			stdinbuf := bufio.NewWriter(stdinout)
			_, err = stdinbuf.Write([]byte(stdinTestData))

			if err != nil {
				panic(err)
			}
			err = stdinbuf.Flush()
			if err != nil {
				panic(err)
			}
		}()
	}

	if s.streamOptions.Stdout != nil {
		s.wg.Add(1)
		go s.readFromStdOut(stdoutin)
	}

	if s.streamOptions.Stderr != nil {
		s.wg.Add(1)
		go s.readFromStdErr(stderrin)
	}

	s.wg.Wait()

	if s.streamOptions.Stdin != nil && !s.stdinPassed {
		t.Log("Stdin not passed")
		t.Fail()
	}

	if s.streamOptions.Stdout != nil && !s.stdoutPassed {
		t.Log("Stdout not passed")
		t.Fail()
	}

	if s.streamOptions.Stderr != nil && !s.stderrPassed {
		t.Log("Stderr not passed")
		t.Fail()
	}

}

func (s *testServer) wsBinaryv4(w http.ResponseWriter, r *http.Request) {

	ws, err := upgrader.Upgrade(w, r, nil)
	s.wsServer = ws
	if err != nil {
		panic(err)
	}

	if s.streamOptions.Stdout != nil {
		var data []byte
		data = append(data, StreamStdOut)
		data = append(data, []byte(stdOutTestData)...)

		if !s.isBinary {
			enc := base64.StdEncoding.EncodeToString(data)
			data = append([]byte{'1'}, []byte(enc)...)
		}

		ws.WriteMessage(websocket.BinaryMessage, data)
	}

	if s.streamOptions.Stderr != nil {
		var data []byte
		data = append(data, StreamStdErr)
		data = append(data, []byte(stdErrTestData)...)

		if !s.isBinary {
			enc := base64.StdEncoding.EncodeToString(data)
			data = append([]byte{'2'}, []byte(enc)...)
		}

		ws.WriteMessage(websocket.BinaryMessage, data)
	}

	if s.streamOptions.Stdin != nil {
		s.readPumpv4(ws)
	} else {
		defer ws.Close()
	}

}

func (s *testServer) readPumpv4(conn *websocket.Conn) {
	defer func() {
		s.wg.Done()
		conn.Close()
	}()
	s.wg.Add(1)
	conn.SetReadLimit(maxMessageSize)
	conn.SetReadDeadline(time.Now().Add(pongWait))
	conn.SetPongHandler(func(string) error { conn.SetReadDeadline(time.Now().Add(pongWait)); return nil })
	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				panic(err)
			}
			break
		}

		if s.isBinary {
			// make sure the message starts with 0 (stdin)
			if message[0] != StreamStdIn {
				s.t.Log("Invalid channel byte")
				s.t.FailNow()
			}
		} else {
			if message[0] != Base64StreamStdIn {
				s.t.Log("Invalid channel byte")
				s.t.FailNow()
			}
		}

		messageAfterStream := message[1:]
		messageAsString := string(messageAfterStream)

		if !s.isBinary {
			decodedBytes, err := base64.StdEncoding.DecodeString(messageAsString)
			if err != nil {
				panic(err)
			}
			messageAsString = string(decodedBytes)
		}

		// check the message didn't change
		if messageAsString != stdinTestData {
			s.t.FailNow()
		}

		s.t.Log("Std in passed")
		s.stdinPassed = true
		break

	}
}
