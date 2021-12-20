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

type testServer struct {
	httpServer    *http.Server
	t             *testing.T
	wg            *sync.WaitGroup
	streamOptions *StreamOptions

	wsClient *websocket.Conn
	wsServer *websocket.Conn

	stdinPassed  bool
	stdoutPassed bool
	stderrPassed bool

	isBinary bool
}

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
}

const (
	stdinTestData  = "this is a \ntest\n"
	stdOutTestData = "this\nis\n from \r\n stdanard out \n"
	stdErrTestData = "this\nis\n \t\tfrom \r\n stdanard err \n"
)

func TestPreV4Binary(t *testing.T) {

	m := http.NewServeMux()
	s := &testServer{
		httpServer:   &http.Server{Addr: "127.0.0.1:8765", Handler: m},
		t:            t,
		wg:           &sync.WaitGroup{},
		stderrPassed: false,
		stdinPassed:  false,
		stdoutPassed: false,
	}
	m.HandleFunc("/wsbinary", s.wsBinary)

	go s.httpServer.ListenAndServe()

	time.Sleep(2 * time.Second)
	s.isBinary = true
	runTestCase(t, true, true, true, s)
	runTestCase(t, true, true, false, s)
	runTestCase(t, true, false, true, s)
	runTestCase(t, false, true, true, s)
	runTestCase(t, false, false, true, s)
	runTestCase(t, true, false, false, s)
	runTestCase(t, false, true, false, s)
	runTestCase(t, false, false, false, s)

	s.isBinary = false
	runTestCase(t, true, true, true, s)
	runTestCase(t, true, true, false, s)
	runTestCase(t, true, false, true, s)
	runTestCase(t, false, true, true, s)
	runTestCase(t, false, false, true, s)
	runTestCase(t, true, false, false, s)
	runTestCase(t, false, true, false, s)
	runTestCase(t, false, false, false, s)

}

func runTestCase(t *testing.T, runStdIn bool, runStdOut bool, runStdErr bool, s *testServer) {
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
		streamer = newPreV4BinaryProtocol(*s.streamOptions)
	} else {
		streamer = newPreV4Base64Protocol(*s.streamOptions)
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

func (s *testServer) wsBinary(w http.ResponseWriter, r *http.Request) {

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
		s.readPump(ws)
	}

}

func (s *testServer) readFromStdOut(reader io.Reader) {

	buffer := make([]byte, 1024)
	numBytes, err := reader.Read(buffer)
	if err != nil {
		panic(err)

	}

	var messageAsString string

	messageAsString = string(buffer[0:numBytes])

	if messageAsString == stdOutTestData {
		s.t.Log("stdout success")
		s.stdoutPassed = true
	} else {
		s.t.Log("stdout failed")
	}

	s.wg.Done()

}

func (s *testServer) readFromStdErr(reader io.Reader) {

	defer s.wg.Done()

	buffer := make([]byte, 1024)
	numBytes, err := reader.Read(buffer)
	if err != nil {
		panic(err)

	}

	messageAsString := string(buffer[0:numBytes])

	if messageAsString == stdErrTestData {
		s.t.Log("stderr success")
		s.stderrPassed = true
	} else {
		s.t.Log("stderr failed")
	}
}

func (s *testServer) readPump(conn *websocket.Conn) {
	defer func() {
		s.wg.Done()

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
