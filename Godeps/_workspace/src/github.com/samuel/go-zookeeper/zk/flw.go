package zk

import (
	"bufio"
	"bytes"
	"fmt"
	"io/ioutil"
	"math/big"
	"net"
	"regexp"
	"strconv"
	"time"
)

// FLWSrvr is a FourLetterWord helper function. In particular, this function pulls the srvr output
// from the zookeeper instances and parses the output. A slice of *ServerStats structs are returned
// as well as a boolean value to indicate whether this function processed successfully.
//
// If the boolean value is false there was a problem. If the *ServerStats slice is empty or nil,
// then the error happened before we started to obtain 'srvr' values. Otherwise, one of the
// servers had an issue and the "Error" value in the struct should be inspected to determine
// which server had the issue.
func FLWSrvr(servers []string, timeout time.Duration) ([]*ServerStats, bool) {
	// different parts of the regular expression that are required to parse the srvr output
	var (
		zrVer   = `^Zookeeper version: ([A-Za-z0-9\.\-]+), built on (\d\d/\d\d/\d\d\d\d \d\d:\d\d [A-Za-z0-9:\+\-]+)`
		zrLat   = `^Latency min/avg/max: (\d+)/(\d+)/(\d+)`
		zrNet   = `^Received: (\d+).*\n^Sent: (\d+).*\n^Connections: (\d+).*\n^Outstanding: (\d+)`
		zrState = `^Zxid: (0x[A-Za-z0-9]+).*\n^Mode: (\w+).*\n^Node count: (\d+)`
	)

	// build the regex from the pieces above
	re, err := regexp.Compile(fmt.Sprintf(`(?m:\A%v.*\n%v.*\n%v.*\n%v)`, zrVer, zrLat, zrNet, zrState))

	if err != nil {
		return nil, false
	}

	imOk := true
	servers = FormatServers(servers)
	ss := make([]*ServerStats, len(servers))

	for i := range ss {
		response, err := fourLetterWord(servers[i], "srvr", timeout)

		if err != nil {
			ss[i] = &ServerStats{Error: err}
			imOk = false
			continue
		}

		match := re.FindAllStringSubmatch(string(response), -1)[0][1:]

		if match == nil {
			err := fmt.Errorf("unable to parse fields from zookeeper response (no regex matches)")
			ss[i] = &ServerStats{Error: err}
			imOk = false
			continue
		}

		// determine current server
		var srvrMode Mode
		switch match[10] {
		case "leader":
			srvrMode = ModeLeader
		case "follower":
			srvrMode = ModeFollower
		case "standalone":
			srvrMode = ModeStandalone
		default:
			srvrMode = ModeUnknown
		}

		buildTime, err := time.Parse("01/02/2006 15:04 MST", match[1])

		if err != nil {
			ss[i] = &ServerStats{Error: err}
			imOk = false
			continue
		}

		parsedInt, err := strconv.ParseInt(match[9], 0, 64)

		if err != nil {
			ss[i] = &ServerStats{Error: err}
			imOk = false
			continue
		}

		// the ZxID value is an int64 with two int32s packed inside
		// the high int32 is the epoch (i.e., number of leader elections)
		// the low int32 is the counter
		epoch := int32(parsedInt >> 32)
		counter := int32(parsedInt & 0xFFFFFFFF)

		// within the regex above, these values must be numerical
		// so we can avoid useless checking of the error return value
		minLatency, _ := strconv.ParseInt(match[2], 0, 64)
		avgLatency, _ := strconv.ParseInt(match[3], 0, 64)
		maxLatency, _ := strconv.ParseInt(match[4], 0, 64)
		recv, _ := strconv.ParseInt(match[5], 0, 64)
		sent, _ := strconv.ParseInt(match[6], 0, 64)
		cons, _ := strconv.ParseInt(match[7], 0, 64)
		outs, _ := strconv.ParseInt(match[8], 0, 64)
		ncnt, _ := strconv.ParseInt(match[11], 0, 64)

		ss[i] = &ServerStats{
			Sent:        sent,
			Received:    recv,
			NodeCount:   ncnt,
			MinLatency:  minLatency,
			AvgLatency:  avgLatency,
			MaxLatency:  maxLatency,
			Connections: cons,
			Outstanding: outs,
			Epoch:       epoch,
			Counter:     counter,
			BuildTime:   buildTime,
			Mode:        srvrMode,
			Version:     match[0],
		}
	}

	return ss, imOk
}

// FLWRuok is a FourLetterWord helper function. In particular, this function
// pulls the ruok output from each server.
func FLWRuok(servers []string, timeout time.Duration) []bool {
	servers = FormatServers(servers)
	oks := make([]bool, len(servers))

	for i := range oks {
		response, err := fourLetterWord(servers[i], "ruok", timeout)

		if err != nil {
			continue
		}

		if bytes.Equal(response[:4], []byte("imok")) {
			oks[i] = true
		}
	}
	return oks
}

// FLWCons is a FourLetterWord helper function. In particular, this function
// pulls the ruok output from each server.
//
// As with FLWSrvr, the boolean value indicates whether one of the requests had
// an issue. The Clients struct has an Error value that can be checked.
func FLWCons(servers []string, timeout time.Duration) ([]*ServerClients, bool) {
	var (
		zrAddr = `^ /((?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?):(?:\d+))\[\d+\]`
		zrPac  = `\(queued=(\d+),recved=(\d+),sent=(\d+),sid=(0x[A-Za-z0-9]+),lop=(\w+),est=(\d+),to=(\d+),`
		zrSesh = `lcxid=(0x[A-Za-z0-9]+),lzxid=(0x[A-Za-z0-9]+),lresp=(\d+),llat=(\d+),minlat=(\d+),avglat=(\d+),maxlat=(\d+)\)`
	)

	re, err := regexp.Compile(fmt.Sprintf("%v%v%v", zrAddr, zrPac, zrSesh))

	if err != nil {
		return nil, false
	}

	servers = FormatServers(servers)
	sc := make([]*ServerClients, len(servers))
	imOk := true

	for i := range sc {
		response, err := fourLetterWord(servers[i], "cons", timeout)

		if err != nil {
			sc[i] = &ServerClients{Error: err}
			imOk = false
			continue
		}

		scan := bufio.NewScanner(bytes.NewReader(response))

		var clients []*ServerClient

		for scan.Scan() {
			line := scan.Bytes()

			if len(line) == 0 {
				continue
			}

			m := re.FindAllStringSubmatch(string(line), -1)

			if m == nil {
				err := fmt.Errorf("unable to parse fields from zookeeper response (no regex matches)")
				sc[i] = &ServerClients{Error: err}
				imOk = false
				continue
			}

			match := m[0][1:]

			queued, _ := strconv.ParseInt(match[1], 0, 64)
			recvd, _ := strconv.ParseInt(match[2], 0, 64)
			sent, _ := strconv.ParseInt(match[3], 0, 64)
			sid, _ := strconv.ParseInt(match[4], 0, 64)
			est, _ := strconv.ParseInt(match[6], 0, 64)
			timeout, _ := strconv.ParseInt(match[7], 0, 32)
			lresp, _ := strconv.ParseInt(match[10], 0, 64)
			llat, _ := strconv.ParseInt(match[11], 0, 32)
			minlat, _ := strconv.ParseInt(match[12], 0, 32)
			avglat, _ := strconv.ParseInt(match[13], 0, 32)
			maxlat, _ := strconv.ParseInt(match[14], 0, 32)

			// zookeeper returns a value, '0xffffffffffffffff', as the
			// Lzxid for PING requests in the 'cons' output.
			// unfortunately, in Go that is an invalid int64 and is not represented
			// as -1.
			// However, converting the string value to a big.Int and then back to
			// and int64 properly sets the value to -1
			lzxid, ok := new(big.Int).SetString(match[9], 0)

			var errVal error

			if !ok {
				errVal = fmt.Errorf("failed to convert lzxid value to big.Int")
				imOk = false
			}

			lcxid, ok := new(big.Int).SetString(match[8], 0)

			if !ok && errVal == nil {
				errVal = fmt.Errorf("failed to convert lcxid value to big.Int")
				imOk = false
			}

			clients = append(clients, &ServerClient{
				Queued:        queued,
				Received:      recvd,
				Sent:          sent,
				SessionID:     sid,
				Lcxid:         lcxid.Int64(),
				Lzxid:         lzxid.Int64(),
				Timeout:       int32(timeout),
				LastLatency:   int32(llat),
				MinLatency:    int32(minlat),
				AvgLatency:    int32(avglat),
				MaxLatency:    int32(maxlat),
				Established:   time.Unix(est, 0),
				LastResponse:  time.Unix(lresp, 0),
				Addr:          match[0],
				LastOperation: match[5],
				Error:         errVal,
			})
		}

		sc[i] = &ServerClients{Clients: clients}
	}

	return sc, imOk
}

func fourLetterWord(server, command string, timeout time.Duration) ([]byte, error) {
	conn, err := net.DialTimeout("tcp", server, timeout)

	if err != nil {
		return nil, err
	}

	// the zookeeper server should automatically close this socket
	// once the command has been processed, but better safe than sorry
	defer conn.Close()

	conn.SetWriteDeadline(time.Now().Add(timeout))

	_, err = conn.Write([]byte(command))

	if err != nil {
		return nil, err
	}

	conn.SetReadDeadline(time.Now().Add(timeout))

	resp, err := ioutil.ReadAll(conn)

	if err != nil {
		return nil, err
	}

	return resp, nil
}
