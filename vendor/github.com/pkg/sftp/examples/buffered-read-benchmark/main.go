// buffered-read-benchmark benchmarks the peformance of reading
// from /dev/zero on the server to a []byte on the client via io.Copy.
package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"time"

	"golang.org/x/crypto/ssh"
	"golang.org/x/crypto/ssh/agent"

	"github.com/pkg/sftp"
)

var (
	USER = flag.String("user", os.Getenv("USER"), "ssh username")
	HOST = flag.String("host", "localhost", "ssh server hostname")
	PORT = flag.Int("port", 22, "ssh server port")
	PASS = flag.String("pass", os.Getenv("SOCKSIE_SSH_PASSWORD"), "ssh password")
	SIZE = flag.Int("s", 1<<15, "set max packet size")
)

func init() {
	flag.Parse()
}

func main() {
	var auths []ssh.AuthMethod
	if aconn, err := net.Dial("unix", os.Getenv("SSH_AUTH_SOCK")); err == nil {
		auths = append(auths, ssh.PublicKeysCallback(agent.NewClient(aconn).Signers))

	}
	if *PASS != "" {
		auths = append(auths, ssh.Password(*PASS))
	}

	config := ssh.ClientConfig{
		User: *USER,
		Auth: auths,
	}
	addr := fmt.Sprintf("%s:%d", *HOST, *PORT)
	conn, err := ssh.Dial("tcp", addr, &config)
	if err != nil {
		log.Fatalf("unable to connect to [%s]: %v", addr, err)
	}
	defer conn.Close()

	c, err := sftp.NewClient(conn, sftp.MaxPacket(*SIZE))
	if err != nil {
		log.Fatalf("unable to start sftp subsytem: %v", err)
	}
	defer c.Close()

	r, err := c.Open("/dev/zero")
	if err != nil {
		log.Fatal(err)
	}
	defer r.Close()

	const size = 1e9

	log.Printf("reading %v bytes", size)
	t1 := time.Now()
	n, err := io.ReadFull(r, make([]byte, size))
	if err != nil {
		log.Fatal(err)
	}
	if n != size {
		log.Fatalf("copy: expected %v bytes, got %d", size, n)
	}
	log.Printf("read %v bytes in %s", size, time.Since(t1))
}
