package main

import (
	"log"
	"net"
)

func main() {
	addr, _ := net.ResolveTCPAddr("tcp", "127.0.0.1:2003")
	l, err := net.ListenTCP("tcp", addr)
	if nil != err {
		log.Fatalln(err)
	}
	log.Println("listening", l.Addr())
	for {
		c, err := l.AcceptTCP()
		if nil != err {
			log.Fatalln(err)
		}
		log.Println("accepted", c.RemoteAddr())
	}
}
