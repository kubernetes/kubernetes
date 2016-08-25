// Copyright © 2015 Jerry Jacobs <jerry.jacobs@xor-gate.org>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package afero

import (
	"testing"
	"os"
	"log"
	"fmt"
	"net"
	"flag"
	"time"
	"io/ioutil"
	"crypto/rsa"
	_rand "crypto/rand"
	"encoding/pem"
	"crypto/x509"

	"golang.org/x/crypto/ssh"
	"github.com/pkg/sftp"
)

type SftpFsContext struct {
	sshc   *ssh.Client
	sshcfg *ssh.ClientConfig
	sftpc  *sftp.Client
}

// TODO we only connect with hardcoded user+pass for now
// it should be possible to use $HOME/.ssh/id_rsa to login into the stub sftp server
func SftpConnect(user, password, host string) (*SftpFsContext, error) {
/*
	pemBytes, err := ioutil.ReadFile(os.Getenv("HOME") + "/.ssh/id_rsa")
	if err != nil {
		return nil,err
	}

	signer, err := ssh.ParsePrivateKey(pemBytes)
	if err != nil {
		return nil,err
	}

	sshcfg := &ssh.ClientConfig{
		User: user,
		Auth: []ssh.AuthMethod{
			ssh.Password(password),
			ssh.PublicKeys(signer),
		},
	}
*/

	sshcfg := &ssh.ClientConfig{
		User: user,
		Auth: []ssh.AuthMethod{
			ssh.Password(password),
		},
	}

	sshc, err := ssh.Dial("tcp", host, sshcfg)
	if err != nil {
		return nil,err
	}

	sftpc, err := sftp.NewClient(sshc)
	if err != nil {
		return nil,err
	}

	ctx := &SftpFsContext{
		sshc: sshc,
		sshcfg: sshcfg,
		sftpc: sftpc,
	}

	return ctx,nil
}

func (ctx *SftpFsContext) Disconnect() error {
	ctx.sftpc.Close()
	ctx.sshc.Close()
	return nil
}

// TODO for such a weird reason rootpath is "." when writing "file1" with afero sftp backend
func RunSftpServer(rootpath string) {
	var (
		readOnly      bool
		debugLevelStr string
		debugLevel    int
		debugStderr   bool
		rootDir       string
	)

	flag.BoolVar(&readOnly, "R", false, "read-only server")
	flag.BoolVar(&debugStderr, "e", true, "debug to stderr")
	flag.StringVar(&debugLevelStr, "l", "none", "debug level")
	flag.StringVar(&rootDir, "root", rootpath, "root directory")
	flag.Parse()

	debugStream := ioutil.Discard
	if debugStderr {
		debugStream = os.Stderr
		debugLevel = 1
	}

	// An SSH server is represented by a ServerConfig, which holds
	// certificate details and handles authentication of ServerConns.
	config := &ssh.ServerConfig{
		PasswordCallback: func(c ssh.ConnMetadata, pass []byte) (*ssh.Permissions, error) {
			// Should use constant-time compare (or better, salt+hash) in
			// a production setting.
			fmt.Fprintf(debugStream, "Login: %s\n", c.User())
			if c.User() == "test" && string(pass) == "test" {
				return nil, nil
			}
			return nil, fmt.Errorf("password rejected for %q", c.User())
		},
	}

	privateBytes, err := ioutil.ReadFile("./test/id_rsa")
	if err != nil {
		log.Fatal("Failed to load private key", err)
	}

	private, err := ssh.ParsePrivateKey(privateBytes)
	if err != nil {
		log.Fatal("Failed to parse private key", err)
	}

	config.AddHostKey(private)

	// Once a ServerConfig has been configured, connections can be
	// accepted.
	listener, err := net.Listen("tcp", "0.0.0.0:2022")
	if err != nil {
		log.Fatal("failed to listen for connection", err)
	}
	fmt.Printf("Listening on %v\n", listener.Addr())

	nConn, err := listener.Accept()
	if err != nil {
		log.Fatal("failed to accept incoming connection", err)
	}

	// Before use, a handshake must be performed on the incoming
	// net.Conn.
	_, chans, reqs, err := ssh.NewServerConn(nConn, config)
	if err != nil {
		log.Fatal("failed to handshake", err)
	}
	fmt.Fprintf(debugStream, "SSH server established\n")

	// The incoming Request channel must be serviced.
	go ssh.DiscardRequests(reqs)

	// Service the incoming Channel channel.
	for newChannel := range chans {
		// Channels have a type, depending on the application level
		// protocol intended. In the case of an SFTP session, this is "subsystem"
		// with a payload string of "<length=4>sftp"
		fmt.Fprintf(debugStream, "Incoming channel: %s\n", newChannel.ChannelType())
		if newChannel.ChannelType() != "session" {
			newChannel.Reject(ssh.UnknownChannelType, "unknown channel type")
			fmt.Fprintf(debugStream, "Unknown channel type: %s\n", newChannel.ChannelType())
			continue
		}
		channel, requests, err := newChannel.Accept()
		if err != nil {
			log.Fatal("could not accept channel.", err)
		}
		fmt.Fprintf(debugStream, "Channel accepted\n")

		// Sessions have out-of-band requests such as "shell",
		// "pty-req" and "env".  Here we handle only the
		// "subsystem" request.
		go func(in <-chan *ssh.Request) {
			for req := range in {
				fmt.Fprintf(debugStream, "Request: %v\n", req.Type)
				ok := false
				switch req.Type {
				case "subsystem":
					fmt.Fprintf(debugStream, "Subsystem: %s\n", req.Payload[4:])
					if string(req.Payload[4:]) == "sftp" {
						ok = true
					}
				}
				fmt.Fprintf(debugStream, " - accepted: %v\n", ok)
				req.Reply(ok, nil)
			}
		}(requests)

		server, err := sftp.NewServer(channel, channel, debugStream, debugLevel, readOnly, rootpath)
		if err != nil {
			log.Fatal(err)
		}
		if err := server.Serve(); err != nil {
			log.Fatal("sftp server completed with error:", err)
		}
	}
}

// MakeSSHKeyPair make a pair of public and private keys for SSH access.
// Public key is encoded in the format for inclusion in an OpenSSH authorized_keys file.
// Private Key generated is PEM encoded
func MakeSSHKeyPair(bits int, pubKeyPath, privateKeyPath string) error {
	privateKey, err := rsa.GenerateKey(_rand.Reader, bits)
	if err != nil {
		return err
	}

	// generate and write private key as PEM
	privateKeyFile, err := os.Create(privateKeyPath)
	defer privateKeyFile.Close()
	if err != nil {
		return err
	}

	privateKeyPEM := &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(privateKey)}
	if err := pem.Encode(privateKeyFile, privateKeyPEM); err != nil {
		return err
	}

	// generate and write public key
	pub, err := ssh.NewPublicKey(&privateKey.PublicKey)
	if err != nil {
		return err
	}

	return ioutil.WriteFile(pubKeyPath, ssh.MarshalAuthorizedKey(pub), 0655)
}

func TestSftpCreate(t *testing.T) {
	os.Mkdir("./test", 0777)
	MakeSSHKeyPair(1024, "./test/id_rsa.pub", "./test/id_rsa")

	go RunSftpServer("./test/")
	time.Sleep(5 * time.Second)

	ctx, err := SftpConnect("test", "test", "localhost:2022")
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Disconnect()

	var AppFs Fs = SftpFs{
		SftpClient: ctx.sftpc,
	}

	AppFs.MkdirAll("test/dir1/dir2/dir3", os.FileMode(0777))
	AppFs.Mkdir("test/foo", os.FileMode(0000))
	AppFs.Chmod("test/foo", os.FileMode(0700))
	AppFs.Mkdir("test/bar", os.FileMode(0777))

	file, err := AppFs.Create("file1")
	if err != nil {
		t.Error(err)
	}
	defer file.Close()

	file.Write([]byte("hello\t"))
	file.WriteString("world!\n")

	f1, err := AppFs.Open("file1")
	if err != nil {
		log.Fatalf("open: %v", err)
	}
	defer f1.Close()

	b := make([]byte, 100)

	_, err = f1.Read(b)
	fmt.Println(string(b))

	// TODO check here if "hello\tworld\n" is in buffer b
}
