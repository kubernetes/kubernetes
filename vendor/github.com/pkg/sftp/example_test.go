package sftp_test

import (
	"fmt"
	"log"
	"os"
	"os/exec"

	"github.com/pkg/sftp"
	"golang.org/x/crypto/ssh"
)

func Example(conn *ssh.Client) {
	// open an SFTP session over an existing ssh connection.
	sftp, err := sftp.NewClient(conn)
	if err != nil {
		log.Fatal(err)
	}
	defer sftp.Close()

	// walk a directory
	w := sftp.Walk("/home/user")
	for w.Step() {
		if w.Err() != nil {
			continue
		}
		log.Println(w.Path())
	}

	// leave your mark
	f, err := sftp.Create("hello.txt")
	if err != nil {
		log.Fatal(err)
	}
	if _, err := f.Write([]byte("Hello world!")); err != nil {
		log.Fatal(err)
	}

	// check it's there
	fi, err := sftp.Lstat("hello.txt")
	if err != nil {
		log.Fatal(err)
	}
	log.Println(fi)
}

func ExampleNewClientPipe() {
	// Connect to a remote host and request the sftp subsystem via the 'ssh'
	// command.  This assumes that passwordless login is correctly configured.
	cmd := exec.Command("ssh", "example.com", "-s", "sftp")

	// send errors from ssh to stderr
	cmd.Stderr = os.Stderr

	// get stdin and stdout
	wr, err := cmd.StdinPipe()
	if err != nil {
		log.Fatal(err)
	}
	rd, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatal(err)
	}

	// start the process
	if err := cmd.Start(); err != nil {
		log.Fatal(err)
	}
	defer cmd.Wait()

	// open the SFTP session
	client, err := sftp.NewClientPipe(rd, wr)
	if err != nil {
		log.Fatal(err)
	}

	// read a directory
	list, err := client.ReadDir("/")
	if err != nil {
		log.Fatal(err)
	}

	// print contents
	for _, item := range list {
		fmt.Println(item.Name())
	}

	// close the connection
	client.Close()
}
