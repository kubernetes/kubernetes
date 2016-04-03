package zk

import (
	"net"
	"testing"
	"time"
)

var (
	zkSrvrOut = `Zookeeper version: 3.4.6-1569965, built on 02/20/2014 09:09 GMT
Latency min/avg/max: 0/1/10
Received: 4207
Sent: 4220
Connections: 81
Outstanding: 1
Zxid: 0x110a7a8f37
Mode: leader
Node count: 306
`
	zkConsOut = ` /10.42.45.231:45361[1](queued=0,recved=9435,sent=9457,sid=0x94c2989e04716b5,lop=PING,est=1427238717217,to=20001,lcxid=0x55120915,lzxid=0xffffffffffffffff,lresp=1427259255908,llat=0,minlat=0,avglat=1,maxlat=17)
 /10.55.33.98:34342[1](queued=0,recved=9338,sent=9350,sid=0x94c2989e0471731,lop=PING,est=1427238849319,to=20001,lcxid=0x55120944,lzxid=0xffffffffffffffff,lresp=1427259252294,llat=0,minlat=0,avglat=1,maxlat=18)
 /10.44.145.114:46556[1](queued=0,recved=109253,sent=109617,sid=0x94c2989e0471709,lop=DELE,est=1427238791305,to=20001,lcxid=0x55139618,lzxid=0x110a7b187d,lresp=1427259257423,llat=2,minlat=0,avglat=1,maxlat=23)

`
)

func TestFLWRuok(t *testing.T) {
	l, err := net.Listen("tcp", "127.0.0.1:2181")

	if err != nil {
		t.Fatalf(err.Error())
	}

	go tcpServer(l, "")

	var oks []bool
	var ok bool

	oks = FLWRuok([]string{"127.0.0.1"}, time.Second*10)

	// close the connection, and pause shortly
	// to cheat around a race condition
	l.Close()
	time.Sleep(time.Millisecond * 1)

	if len(oks) == 0 {
		t.Errorf("no values returned")
	}

	ok = oks[0]

	if !ok {
		t.Errorf("instance should be marked as OK")
	}

	//
	// Confirm that it also returns false for dead instances
	//
	l, err = net.Listen("tcp", "127.0.0.1:2181")

	if err != nil {
		t.Fatalf(err.Error())
	}

	defer l.Close()

	go tcpServer(l, "dead")

	oks = FLWRuok([]string{"127.0.0.1"}, time.Second*10)

	if len(oks) == 0 {
		t.Errorf("no values returned")
	}

	ok = oks[0]

	if ok {
		t.Errorf("instance should be marked as not OK")
	}
}

func TestFLWSrvr(t *testing.T) {
	l, err := net.Listen("tcp", "127.0.0.1:2181")

	if err != nil {
		t.Fatalf(err.Error())
	}

	defer l.Close()

	go tcpServer(l, "")

	var statsSlice []*ServerStats
	var stats *ServerStats
	var ok bool

	statsSlice, ok = FLWSrvr([]string{"127.0.0.1:2181"}, time.Second*10)

	if !ok {
		t.Errorf("failure indicated on 'srvr' parsing")
	}

	if len(statsSlice) == 0 {
		t.Errorf("no *ServerStats instances returned")
	}

	stats = statsSlice[0]

	if stats.Error != nil {
		t.Fatalf("error seen in stats: %v", err.Error())
	}

	if stats.Sent != 4220 {
		t.Errorf("Sent != 4220")
	}

	if stats.Received != 4207 {
		t.Errorf("Received != 4207")
	}

	if stats.NodeCount != 306 {
		t.Errorf("NodeCount != 306")
	}

	if stats.MinLatency != 0 {
		t.Errorf("MinLatency != 0")
	}

	if stats.AvgLatency != 1 {
		t.Errorf("AvgLatency != 1")
	}

	if stats.MaxLatency != 10 {
		t.Errorf("MaxLatency != 10")
	}

	if stats.Connections != 81 {
		t.Errorf("Connection != 81")
	}

	if stats.Outstanding != 1 {
		t.Errorf("Outstanding != 1")
	}

	if stats.Epoch != 17 {
		t.Errorf("Epoch != 17")
	}

	if stats.Counter != 175804215 {
		t.Errorf("Counter != 175804215")
	}

	if stats.Mode != ModeLeader {
		t.Errorf("Mode != ModeLeader")
	}

	if stats.Version != "3.4.6-1569965" {
		t.Errorf("Version expected: 3.4.6-1569965")
	}

	buildTime, err := time.Parse("01/02/2006 15:04 MST", "02/20/2014 09:09 GMT")

	if !stats.BuildTime.Equal(buildTime) {

	}
}

func TestFLWCons(t *testing.T) {
	l, err := net.Listen("tcp", "127.0.0.1:2181")

	if err != nil {
		t.Fatalf(err.Error())
	}

	defer l.Close()

	go tcpServer(l, "")

	var clients []*ServerClients
	var ok bool

	clients, ok = FLWCons([]string{"127.0.0.1"}, time.Second*10)

	if !ok {
		t.Errorf("failure indicated on 'cons' parsing")
	}

	if len(clients) == 0 {
		t.Errorf("no *ServerClients instances returned")
	}

	results := []*ServerClient{
		&ServerClient{
			Queued:        0,
			Received:      9435,
			Sent:          9457,
			SessionID:     669956116721374901,
			LastOperation: "PING",
			Established:   time.Unix(1427238717217, 0),
			Timeout:       20001,
			Lcxid:         1427245333,
			Lzxid:         -1,
			LastResponse:  time.Unix(1427259255908, 0),
			LastLatency:   0,
			MinLatency:    0,
			AvgLatency:    1,
			MaxLatency:    17,
			Addr:          "10.42.45.231:45361",
		},
		&ServerClient{
			Queued:        0,
			Received:      9338,
			Sent:          9350,
			SessionID:     669956116721375025,
			LastOperation: "PING",
			Established:   time.Unix(1427238849319, 0),
			Timeout:       20001,
			Lcxid:         1427245380,
			Lzxid:         -1,
			LastResponse:  time.Unix(1427259252294, 0),
			LastLatency:   0,
			MinLatency:    0,
			AvgLatency:    1,
			MaxLatency:    18,
			Addr:          "10.55.33.98:34342",
		},
		&ServerClient{
			Queued:        0,
			Received:      109253,
			Sent:          109617,
			SessionID:     669956116721374985,
			LastOperation: "DELE",
			Established:   time.Unix(1427238791305, 0),
			Timeout:       20001,
			Lcxid:         1427346968,
			Lzxid:         73190283389,
			LastResponse:  time.Unix(1427259257423, 0),
			LastLatency:   2,
			MinLatency:    0,
			AvgLatency:    1,
			MaxLatency:    23,
			Addr:          "10.44.145.114:46556",
		},
	}

	for _, z := range clients {
		if z.Error != nil {
			t.Errorf("error seen: %v", err.Error())
		}

		for i, v := range z.Clients {
			c := results[i]

			if v.Error != nil {
				t.Errorf("client error seen: %v", err.Error())
			}

			if v.Queued != c.Queued {
				t.Errorf("Queued value mismatch (%d/%d)", v.Queued, c.Queued)
			}

			if v.Received != c.Received {
				t.Errorf("Received value mismatch (%d/%d)", v.Received, c.Received)
			}

			if v.Sent != c.Sent {
				t.Errorf("Sent value mismatch (%d/%d)", v.Sent, c.Sent)
			}

			if v.SessionID != c.SessionID {
				t.Errorf("SessionID value mismatch (%d/%d)", v.SessionID, c.SessionID)
			}

			if v.LastOperation != c.LastOperation {
				t.Errorf("LastOperation value mismatch ('%v'/'%v')", v.LastOperation, c.LastOperation)
			}

			if v.Timeout != c.Timeout {
				t.Errorf("Timeout value mismatch (%d/%d)", v.Timeout, c.Timeout)
			}

			if v.Lcxid != c.Lcxid {
				t.Errorf("Lcxid value mismatch (%d/%d)", v.Lcxid, c.Lcxid)
			}

			if v.Lzxid != c.Lzxid {
				t.Errorf("Lzxid value mismatch (%d/%d)", v.Lzxid, c.Lzxid)
			}

			if v.LastLatency != c.LastLatency {
				t.Errorf("LastLatency value mismatch (%d/%d)", v.LastLatency, c.LastLatency)
			}

			if v.MinLatency != c.MinLatency {
				t.Errorf("MinLatency value mismatch (%d/%d)", v.MinLatency, c.MinLatency)
			}

			if v.AvgLatency != c.AvgLatency {
				t.Errorf("AvgLatency value mismatch (%d/%d)", v.AvgLatency, c.AvgLatency)
			}

			if v.MaxLatency != c.MaxLatency {
				t.Errorf("MaxLatency value mismatch (%d/%d)", v.MaxLatency, c.MaxLatency)
			}

			if v.Addr != c.Addr {
				t.Errorf("Addr value mismatch ('%v'/'%v')", v.Addr, c.Addr)
			}

			if !c.Established.Equal(v.Established) {
				t.Errorf("Established value mismatch (%v/%v)", c.Established, v.Established)
			}

			if !c.LastResponse.Equal(v.LastResponse) {
				t.Errorf("Established value mismatch (%v/%v)", c.LastResponse, v.LastResponse)
			}
		}
	}
}

func tcpServer(listener net.Listener, thing string) {
	for {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		go connHandler(conn, thing)
	}
}

func connHandler(conn net.Conn, thing string) {
	defer conn.Close()

	data := make([]byte, 4)

	_, err := conn.Read(data)

	if err != nil {
		return
	}

	switch string(data) {
	case "ruok":
		switch thing {
		case "dead":
			return
		default:
			conn.Write([]byte("imok"))
		}
	case "srvr":
		switch thing {
		case "dead":
			return
		default:
			conn.Write([]byte(zkSrvrOut))
		}
	case "cons":
		switch thing {
		case "dead":
			return
		default:
			conn.Write([]byte(zkConsOut))
		}
	default:
		conn.Write([]byte("This ZooKeeper instance is not currently serving requests."))
	}
}
