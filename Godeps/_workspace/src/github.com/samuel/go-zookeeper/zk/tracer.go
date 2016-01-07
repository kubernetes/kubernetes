package zk

import (
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"sync"
)

var (
	requests     = make(map[int32]int32) // Map of Xid -> Opcode
	requestsLock = &sync.Mutex{}
)

func trace(conn1, conn2 net.Conn, client bool) {
	defer conn1.Close()
	defer conn2.Close()
	buf := make([]byte, 10*1024)
	init := true
	for {
		_, err := io.ReadFull(conn1, buf[:4])
		if err != nil {
			fmt.Println("1>", client, err)
			return
		}

		blen := int(binary.BigEndian.Uint32(buf[:4]))

		_, err = io.ReadFull(conn1, buf[4:4+blen])
		if err != nil {
			fmt.Println("2>", client, err)
			return
		}

		var cr interface{}
		opcode := int32(-1)
		readHeader := true
		if client {
			if init {
				cr = &connectRequest{}
				readHeader = false
			} else {
				xid := int32(binary.BigEndian.Uint32(buf[4:8]))
				opcode = int32(binary.BigEndian.Uint32(buf[8:12]))
				requestsLock.Lock()
				requests[xid] = opcode
				requestsLock.Unlock()
				cr = requestStructForOp(opcode)
				if cr == nil {
					fmt.Printf("Unknown opcode %d\n", opcode)
				}
			}
		} else {
			if init {
				cr = &connectResponse{}
				readHeader = false
			} else {
				xid := int32(binary.BigEndian.Uint32(buf[4:8]))
				zxid := int64(binary.BigEndian.Uint64(buf[8:16]))
				errnum := int32(binary.BigEndian.Uint32(buf[16:20]))
				if xid != -1 || zxid != -1 {
					requestsLock.Lock()
					found := false
					opcode, found = requests[xid]
					if !found {
						opcode = 0
					}
					delete(requests, xid)
					requestsLock.Unlock()
				} else {
					opcode = opWatcherEvent
				}
				cr = responseStructForOp(opcode)
				if cr == nil {
					fmt.Printf("Unknown opcode %d\n", opcode)
				}
				if errnum != 0 {
					cr = &struct{}{}
				}
			}
		}
		opname := "."
		if opcode != -1 {
			opname = opNames[opcode]
		}
		if cr == nil {
			fmt.Printf("%+v %s %+v\n", client, opname, buf[4:4+blen])
		} else {
			n := 4
			hdrStr := ""
			if readHeader {
				var hdr interface{}
				if client {
					hdr = &requestHeader{}
				} else {
					hdr = &responseHeader{}
				}
				if n2, err := decodePacket(buf[n:n+blen], hdr); err != nil {
					fmt.Println(err)
				} else {
					n += n2
				}
				hdrStr = fmt.Sprintf(" %+v", hdr)
			}
			if _, err := decodePacket(buf[n:n+blen], cr); err != nil {
				fmt.Println(err)
			}
			fmt.Printf("%+v %s%s %+v\n", client, opname, hdrStr, cr)
		}

		init = false

		written, err := conn2.Write(buf[:4+blen])
		if err != nil {
			fmt.Println("3>", client, err)
			return
		} else if written != 4+blen {
			fmt.Printf("Written != read: %d != %d\n", written, blen)
			return
		}
	}
}

func handleConnection(addr string, conn net.Conn) {
	zkConn, err := net.Dial("tcp", addr)
	if err != nil {
		fmt.Println(err)
		return
	}
	go trace(conn, zkConn, true)
	trace(zkConn, conn, false)
}

func StartTracer(listenAddr, serverAddr string) {
	ln, err := net.Listen("tcp", listenAddr)
	if err != nil {
		panic(err)
	}
	for {
		conn, err := ln.Accept()
		if err != nil {
			fmt.Println(err)
			continue
		}
		go handleConnection(serverAddr, conn)
	}
}
