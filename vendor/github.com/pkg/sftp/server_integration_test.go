package sftp

// sftp server integration tests
// enable with -integration
// example invokation (darwin): gofmt -w `find . -name \*.go` && (cd server_standalone/ ; go build -tags debug) && go test -tags debug github.com/pkg/sftp -integration -v -sftp /usr/libexec/sftp-server -run ServerCompareSubsystems

import (
	"bytes"
	"encoding/hex"
	"flag"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/kr/fs"
	"golang.org/x/crypto/ssh"
)

var testSftpClientBin = flag.String("sftp_client", "/usr/bin/sftp", "location of the sftp client binary")
var sshServerDebugStream = ioutil.Discard
var sftpServerDebugStream = ioutil.Discard
var sftpClientDebugStream = ioutil.Discard

const (
	GOLANG_SFTP  = true
	OPENSSH_SFTP = false
)

var (
	hostPrivateKeySigner ssh.Signer
	privKey              = []byte(`
-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEArhp7SqFnXVZAgWREL9Ogs+miy4IU/m0vmdkoK6M97G9NX/Pj
wf8I/3/ynxmcArbt8Rc4JgkjT2uxx/NqR0yN42N1PjO5Czu0dms1PSqcKIJdeUBV
7gdrKSm9Co4d2vwfQp5mg47eG4w63pz7Drk9+VIyi9YiYH4bve7WnGDswn4ycvYZ
slV5kKnjlfCdPig+g5P7yQYud0cDWVwyA0+kxvL6H3Ip+Fu8rLDZn4/P1WlFAIuc
PAf4uEKDGGmC2URowi5eesYR7f6GN/HnBs2776laNlAVXZUmYTUfOGagwLsEkx8x
XdNqntfbs2MOOoK+myJrNtcB9pCrM0H6um19uQIDAQABAoIBABkWr9WdVKvalgkP
TdQmhu3mKRNyd1wCl+1voZ5IM9Ayac/98UAvZDiNU4Uhx52MhtVLJ0gz4Oa8+i16
IkKMAZZW6ro/8dZwkBzQbieWUFJ2Fso2PyvB3etcnGU8/Yhk9IxBDzy+BbuqhYE2
1ebVQtz+v1HvVZzaD11bYYm/Xd7Y28QREVfFen30Q/v3dv7dOteDE/RgDS8Czz7w
jMW32Q8JL5grz7zPkMK39BLXsTcSYcaasT2ParROhGJZDmbgd3l33zKCVc1zcj9B
SA47QljGd09Tys958WWHgtj2o7bp9v1Ufs4LnyKgzrB80WX1ovaSQKvd5THTLchO
kLIhUAECgYEA2doGXy9wMBmTn/hjiVvggR1aKiBwUpnB87Hn5xCMgoECVhFZlT6l
WmZe7R2klbtG1aYlw+y+uzHhoVDAJW9AUSV8qoDUwbRXvBVlp+In5wIqJ+VjfivK
zgIfzomL5NvDz37cvPmzqIeySTowEfbQyq7CUQSoDtE9H97E2wWZhDkCgYEAzJdJ
k+NSFoTkHhfD3L0xCDHpRV3gvaOeew8524fVtVUq53X8m91ng4AX1r74dCUYwwiF
gqTtSSJfx2iH1xKnNq28M9uKg7wOrCKrRqNPnYUO3LehZEC7rwUr26z4iJDHjjoB
uBcS7nw0LJ+0Zeg1IF+aIdZGV3MrAKnrzWPixYECgYBsffX6ZWebrMEmQ89eUtFF
u9ZxcGI/4K8ErC7vlgBD5ffB4TYZ627xzFWuBLs4jmHCeNIJ9tct5rOVYN+wRO1k
/CRPzYUnSqb+1jEgILL6istvvv+DkE+ZtNkeRMXUndWwel94BWsBnUKe0UmrSJ3G
sq23J3iCmJW2T3z+DpXbkQKBgQCK+LUVDNPE0i42NsRnm+fDfkvLP7Kafpr3Umdl
tMY474o+QYn+wg0/aPJIf9463rwMNyyhirBX/k57IIktUdFdtfPicd2MEGETElWv
nN1GzYxD50Rs2f/jKisZhEwqT9YNyV9DkgDdGGdEbJNYqbv0qpwDIg8T9foe8E1p
bdErgQKBgAt290I3L316cdxIQTkJh1DlScN/unFffITwu127WMr28Jt3mq3cZpuM
Aecey/eEKCj+Rlas5NDYKsB18QIuAw+qqWyq0LAKLiAvP1965Rkc4PLScl3MgJtO
QYa37FK0p8NcDeUuF86zXBVutwS5nJLchHhKfd590ks57OROtm29
-----END RSA PRIVATE KEY-----
`)
)

func init() {
	var err error
	hostPrivateKeySigner, err = ssh.ParsePrivateKey(privKey)
	if err != nil {
		panic(err)
	}
}

func keyAuth(conn ssh.ConnMetadata, key ssh.PublicKey) (*ssh.Permissions, error) {
	permissions := &ssh.Permissions{
		CriticalOptions: map[string]string{},
		Extensions:      map[string]string{},
	}
	return permissions, nil
}

func pwAuth(conn ssh.ConnMetadata, pw []byte) (*ssh.Permissions, error) {
	permissions := &ssh.Permissions{
		CriticalOptions: map[string]string{},
		Extensions:      map[string]string{},
	}
	return permissions, nil
}

func basicServerConfig() *ssh.ServerConfig {
	config := ssh.ServerConfig{
		Config: ssh.Config{
			MACs: []string{"hmac-sha1"},
		},
		PasswordCallback:  pwAuth,
		PublicKeyCallback: keyAuth,
	}
	config.AddHostKey(hostPrivateKeySigner)
	return &config
}

type sshServer struct {
	useSubsystem bool
	conn         net.Conn
	config       *ssh.ServerConfig
	sshConn      *ssh.ServerConn
	newChans     <-chan ssh.NewChannel
	newReqs      <-chan *ssh.Request
}

func sshServerFromConn(conn net.Conn, useSubsystem bool, config *ssh.ServerConfig) (*sshServer, error) {
	// From a standard TCP connection to an encrypted SSH connection
	sshConn, newChans, newReqs, err := ssh.NewServerConn(conn, config)
	if err != nil {
		return nil, err
	}

	svr := &sshServer{useSubsystem, conn, config, sshConn, newChans, newReqs}
	svr.listenChannels()
	return svr, nil
}

func (svr *sshServer) Wait() error {
	return svr.sshConn.Wait()
}

func (svr *sshServer) Close() error {
	return svr.sshConn.Close()
}

func (svr *sshServer) listenChannels() {
	go func() {
		for chanReq := range svr.newChans {
			go svr.handleChanReq(chanReq)
		}
	}()
	go func() {
		for req := range svr.newReqs {
			go svr.handleReq(req)
		}
	}()
}

func (svr *sshServer) handleReq(req *ssh.Request) {
	switch req.Type {
	default:
		rejectRequest(req)
	}
}

type sshChannelServer struct {
	svr     *sshServer
	chanReq ssh.NewChannel
	ch      ssh.Channel
	newReqs <-chan *ssh.Request
}

type sshSessionChannelServer struct {
	*sshChannelServer
	env []string
}

func (svr *sshServer) handleChanReq(chanReq ssh.NewChannel) {
	fmt.Fprintf(sshServerDebugStream, "channel request: %v, extra: '%v'\n", chanReq.ChannelType(), hex.EncodeToString(chanReq.ExtraData()))
	switch chanReq.ChannelType() {
	case "session":
		if ch, reqs, err := chanReq.Accept(); err != nil {
			fmt.Fprintf(sshServerDebugStream, "fail to accept channel request: %v\n", err)
			chanReq.Reject(ssh.ResourceShortage, "channel accept failure")
		} else {
			chsvr := &sshSessionChannelServer{
				sshChannelServer: &sshChannelServer{svr, chanReq, ch, reqs},
				env:              append([]string{}, os.Environ()...),
			}
			chsvr.handle()
		}
	default:
		chanReq.Reject(ssh.UnknownChannelType, "channel type is not a session")
	}
}

func (chsvr *sshSessionChannelServer) handle() {
	// should maybe do something here...
	go chsvr.handleReqs()
}

func (chsvr *sshSessionChannelServer) handleReqs() {
	for req := range chsvr.newReqs {
		chsvr.handleReq(req)
	}
	fmt.Fprintf(sshServerDebugStream, "ssh server session channel complete\n")
}

func (chsvr *sshSessionChannelServer) handleReq(req *ssh.Request) {
	switch req.Type {
	case "env":
		chsvr.handleEnv(req)
	case "subsystem":
		chsvr.handleSubsystem(req)
	default:
		rejectRequest(req)
	}
}

func rejectRequest(req *ssh.Request) error {
	fmt.Fprintf(sshServerDebugStream, "ssh rejecting request, type: %s\n", req.Type)
	err := req.Reply(false, []byte{})
	if err != nil {
		fmt.Fprintf(sshServerDebugStream, "ssh request reply had error: %v\n", err)
	}
	return err
}

func rejectRequestUnmarshalError(req *ssh.Request, s interface{}, err error) error {
	fmt.Fprintf(sshServerDebugStream, "ssh request unmarshaling error, type '%T': %v\n", s, err)
	rejectRequest(req)
	return err
}

// env request form:
type sshEnvRequest struct {
	Envvar string
	Value  string
}

func (chsvr *sshSessionChannelServer) handleEnv(req *ssh.Request) error {
	envReq := &sshEnvRequest{}
	if err := ssh.Unmarshal(req.Payload, envReq); err != nil {
		return rejectRequestUnmarshalError(req, envReq, err)
	}
	req.Reply(true, nil)

	found := false
	for i, envstr := range chsvr.env {
		if strings.HasPrefix(envstr, envReq.Envvar+"=") {
			found = true
			chsvr.env[i] = envReq.Envvar + "=" + envReq.Value
		}
	}
	if !found {
		chsvr.env = append(chsvr.env, envReq.Envvar+"="+envReq.Value)
	}

	return nil
}

// Payload: int: command size, string: command
type sshSubsystemRequest struct {
	Name string
}

type sshSubsystemExitStatus struct {
	Status uint32
}

func (chsvr *sshSessionChannelServer) handleSubsystem(req *ssh.Request) error {
	defer func() {
		err1 := chsvr.ch.CloseWrite()
		err2 := chsvr.ch.Close()
		fmt.Fprintf(sshServerDebugStream, "ssh server subsystem request complete, err: %v %v\n", err1, err2)
	}()

	subsystemReq := &sshSubsystemRequest{}
	if err := ssh.Unmarshal(req.Payload, subsystemReq); err != nil {
		return rejectRequestUnmarshalError(req, subsystemReq, err)
	}

	// reply to the ssh client

	// no idea if this is actually correct spec-wise.
	// just enough for an sftp server to start.
	if subsystemReq.Name != "sftp" {
		return req.Reply(false, nil)
	}

	req.Reply(true, nil)

	if !chsvr.svr.useSubsystem {
		// use the openssh sftp server backend; this is to test the ssh code, not the sftp code,
		// or is used for comparison between our sftp subsystem and the openssh sftp subsystem
		cmd := exec.Command(*testSftp, "-e", "-l", "DEBUG") // log to stderr
		cmd.Stdin = chsvr.ch
		cmd.Stdout = chsvr.ch
		cmd.Stderr = sftpServerDebugStream
		if err := cmd.Start(); err != nil {
			return err
		}
		return cmd.Wait()
	}

	sftpServer, err := NewServer(
		chsvr.ch,
		WithDebug(sftpServerDebugStream),
	)
	if err != nil {
		return err
	}

	// wait for the session to close
	runErr := sftpServer.Serve()
	exitStatus := uint32(1)
	if runErr == nil {
		exitStatus = uint32(0)
	}

	_, exitStatusErr := chsvr.ch.SendRequest("exit-status", false, ssh.Marshal(sshSubsystemExitStatus{exitStatus}))
	return exitStatusErr
}

// starts an ssh server to test. returns: host string and port
func testServer(t *testing.T, useSubsystem bool, readonly bool) (net.Listener, string, int) {
	if !*testIntegration {
		t.Skip("skipping intergration test")
	}

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}

	host, portStr, err := net.SplitHostPort(listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatal(err)
	}

	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				fmt.Fprintf(sshServerDebugStream, "ssh server socket closed: %v\n", err)
				break
			}

			go func() {
				defer conn.Close()
				sshSvr, err := sshServerFromConn(conn, useSubsystem, basicServerConfig())
				if err != nil {
					t.Error(err)
					return
				}
				err = sshSvr.Wait()
				fmt.Fprintf(sshServerDebugStream, "ssh server finished, err: %v\n", err)
			}()
		}
	}()

	return listener, host, port
}

func runSftpClient(t *testing.T, script string, path string, host string, port int) (string, error) {
	// if sftp client binary is unavailable, skip test
	if _, err := os.Stat(*testSftpClientBin); err != nil {
		t.Skip("sftp client binary unavailable")
	}
	args := []string{
		// "-vvvv",
		"-b", "-",
		"-o", "StrictHostKeyChecking=no",
		"-o", "LogLevel=ERROR",
		"-o", "UserKnownHostsFile /dev/null",
		"-P", fmt.Sprintf("%d", port), fmt.Sprintf("%s:%s", host, path),
	}
	cmd := exec.Command(*testSftpClientBin, args...)
	var stdout bytes.Buffer
	cmd.Stdin = bytes.NewBufferString(script)
	cmd.Stdout = &stdout
	cmd.Stderr = sftpClientDebugStream
	if err := cmd.Start(); err != nil {
		return "", err
	}
	err := cmd.Wait()
	return string(stdout.Bytes()), err
}

func TestServerCompareSubsystems(t *testing.T) {
	listenerGo, hostGo, portGo := testServer(t, GOLANG_SFTP, READONLY)
	listenerOp, hostOp, portOp := testServer(t, OPENSSH_SFTP, READONLY)
	defer listenerGo.Close()
	defer listenerOp.Close()

	script := `
ls /
ls -l /
ls /dev/
ls -l /dev/
ls -l /etc/
ls -l /bin/
ls -l /usr/bin/
`
	outputGo, err := runSftpClient(t, script, "/", hostGo, portGo)
	if err != nil {
		t.Fatal(err)
	}

	outputOp, err := runSftpClient(t, script, "/", hostOp, portOp)
	if err != nil {
		t.Fatal(err)
	}

	newlineRegex := regexp.MustCompile(`\r*\n`)
	spaceRegex := regexp.MustCompile(`\s+`)
	outputGoLines := newlineRegex.Split(outputGo, -1)
	outputOpLines := newlineRegex.Split(outputOp, -1)

	for i, goLine := range outputGoLines {
		if i > len(outputOpLines) {
			t.Fatalf("output line count differs")
		}
		opLine := outputOpLines[i]
		bad := false
		if goLine != opLine {
			goWords := spaceRegex.Split(goLine, -1)
			opWords := spaceRegex.Split(opLine, -1)
			// allow words[2] and [3] to be different as these are users & groups
			// also allow words[1] to differ as the link count for directories like
			// proc is unstable during testing as processes are created/destroyed.
			for j, goWord := range goWords {
				if j > len(opWords) {
					bad = true
				}
				opWord := opWords[j]
				if goWord != opWord && j != 1 && j != 2 && j != 3 {
					bad = true
				}
			}
		}

		if bad {
			t.Errorf("outputs differ, go:\n%v\nopenssh:\n%v\n", goLine, opLine)
		}
	}
}

var rng = rand.New(rand.NewSource(time.Now().Unix()))

func randData(length int) []byte {
	data := make([]byte, length)
	for i := 0; i < length; i++ {
		data[i] = byte(rng.Uint32())
	}
	return data
}

func randName() string {
	return "sftp." + hex.EncodeToString(randData(16))
}

func TestServerMkdirRmdir(t *testing.T) {
	listenerGo, hostGo, portGo := testServer(t, GOLANG_SFTP, READONLY)
	defer listenerGo.Close()

	tmpDir := "/tmp/" + randName()
	defer os.RemoveAll(tmpDir)

	// mkdir remote
	if _, err := runSftpClient(t, "mkdir "+tmpDir, "/", hostGo, portGo); err != nil {
		t.Fatal(err)
	}

	// directory should now exist
	if _, err := os.Stat(tmpDir); err != nil {
		t.Fatal(err)
	}

	// now remove the directory
	if _, err := runSftpClient(t, "rmdir "+tmpDir, "/", hostGo, portGo); err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(tmpDir); err == nil {
		t.Fatal("should have error after deleting the directory")
	}
}

func TestServerSymlink(t *testing.T) {
	listenerGo, hostGo, portGo := testServer(t, GOLANG_SFTP, READONLY)
	defer listenerGo.Close()

	link := "/tmp/" + randName()
	defer os.RemoveAll(link)

	// now create a symbolic link within the new directory
	if output, err := runSftpClient(t, "symlink /bin/sh "+link, "/", hostGo, portGo); err != nil {
		t.Fatalf("failed: %v %v", err, string(output))
	}

	// symlink should now exist
	if stat, err := os.Lstat(link); err != nil {
		t.Fatal(err)
	} else if (stat.Mode() & os.ModeSymlink) != os.ModeSymlink {
		t.Fatalf("is not a symlink: %v", stat.Mode())
	}
}

func TestServerPut(t *testing.T) {
	listenerGo, hostGo, portGo := testServer(t, GOLANG_SFTP, READONLY)
	defer listenerGo.Close()

	tmpFileLocal := "/tmp/" + randName()
	tmpFileRemote := "/tmp/" + randName()
	defer os.RemoveAll(tmpFileLocal)
	defer os.RemoveAll(tmpFileRemote)

	t.Logf("put: local %v remote %v", tmpFileLocal, tmpFileRemote)

	// create a file with random contents. This will be the local file pushed to the server
	tmpFileLocalData := randData(10 * 1024 * 1024)
	if err := ioutil.WriteFile(tmpFileLocal, tmpFileLocalData, 0644); err != nil {
		t.Fatal(err)
	}

	// sftp the file to the server
	if output, err := runSftpClient(t, "put "+tmpFileLocal+" "+tmpFileRemote, "/", hostGo, portGo); err != nil {
		t.Fatalf("runSftpClient failed: %v, output\n%v\n", err, output)
	}

	// tmpFile2 should now exist, with the same contents
	if tmpFileRemoteData, err := ioutil.ReadFile(tmpFileRemote); err != nil {
		t.Fatal(err)
	} else if string(tmpFileLocalData) != string(tmpFileRemoteData) {
		t.Fatal("contents of file incorrect after put")
	}
}

func TestServerGet(t *testing.T) {
	listenerGo, hostGo, portGo := testServer(t, GOLANG_SFTP, READONLY)
	defer listenerGo.Close()

	tmpFileLocal := "/tmp/" + randName()
	tmpFileRemote := "/tmp/" + randName()
	defer os.RemoveAll(tmpFileLocal)
	defer os.RemoveAll(tmpFileRemote)

	t.Logf("get: local %v remote %v", tmpFileLocal, tmpFileRemote)

	// create a file with random contents. This will be the remote file pulled from the server
	tmpFileRemoteData := randData(10 * 1024 * 1024)
	if err := ioutil.WriteFile(tmpFileRemote, tmpFileRemoteData, 0644); err != nil {
		t.Fatal(err)
	}

	// sftp the file to the server
	if output, err := runSftpClient(t, "get "+tmpFileRemote+" "+tmpFileLocal, "/", hostGo, portGo); err != nil {
		t.Fatalf("runSftpClient failed: %v, output\n%v\n", err, output)
	}

	// tmpFile2 should now exist, with the same contents
	if tmpFileLocalData, err := ioutil.ReadFile(tmpFileLocal); err != nil {
		t.Fatal(err)
	} else if string(tmpFileLocalData) != string(tmpFileRemoteData) {
		t.Fatal("contents of file incorrect after put")
	}
}

func compareDirectoriesRecursive(t *testing.T, aroot, broot string) {
	walker := fs.Walk(aroot)
	for walker.Step() {
		if err := walker.Err(); err != nil {
			t.Fatal(err)
		}
		// find paths
		aPath := walker.Path()
		aRel, err := filepath.Rel(aroot, aPath)
		if err != nil {
			t.Fatalf("could not find relative path for %v: %v", aPath, err)
		}
		bPath := path.Join(broot, aRel)

		if aRel == "." {
			continue
		}

		//t.Logf("comparing: %v a: %v b %v", aRel, aPath, bPath)

		// if a is a link, the sftp recursive copy won't have copied it. ignore
		aLink, err := os.Lstat(aPath)
		if err != nil {
			t.Fatalf("could not lstat %v: %v", aPath, err)
		}
		if aLink.Mode()&os.ModeSymlink != 0 {
			continue
		}

		// stat the files
		aFile, err := os.Stat(aPath)
		if err != nil {
			t.Fatalf("could not stat %v: %v", aPath, err)
		}
		bFile, err := os.Stat(bPath)
		if err != nil {
			t.Fatalf("could not stat %v: %v", bPath, err)
		}

		// compare stats, with some leniency for the timestamp
		if aFile.Mode() != bFile.Mode() {
			t.Fatalf("modes different for %v: %v vs %v", aRel, aFile.Mode(), bFile.Mode())
		}
		if !aFile.IsDir() {
			if aFile.Size() != bFile.Size() {
				t.Fatalf("sizes different for %v: %v vs %v", aRel, aFile.Size(), bFile.Size())
			}
		}
		timeDiff := aFile.ModTime().Sub(bFile.ModTime())
		if timeDiff > time.Second || timeDiff < -time.Second {
			t.Fatalf("mtimes different for %v: %v vs %v", aRel, aFile.ModTime(), bFile.ModTime())
		}

		// compare contents
		if !aFile.IsDir() {
			if aContents, err := ioutil.ReadFile(aPath); err != nil {
				t.Fatal(err)
			} else if bContents, err := ioutil.ReadFile(bPath); err != nil {
				t.Fatal(err)
			} else if string(aContents) != string(bContents) {
				t.Fatalf("contents different for %v", aRel)
			}
		}
	}
}

func TestServerPutRecursive(t *testing.T) {
	listenerGo, hostGo, portGo := testServer(t, GOLANG_SFTP, READONLY)
	defer listenerGo.Close()

	dirLocal, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	tmpDirRemote := "/tmp/" + randName()
	defer os.RemoveAll(tmpDirRemote)

	t.Logf("put recursive: local %v remote %v", dirLocal, tmpDirRemote)

	// push this directory (source code etc) recursively to the server
	if output, err := runSftpClient(t, "mkdir "+tmpDirRemote+"\r\nput -r -P "+dirLocal+"/ "+tmpDirRemote+"/", "/", hostGo, portGo); err != nil {
		t.Fatalf("runSftpClient failed: %v, output\n%v\n", err, output)
	}

	compareDirectoriesRecursive(t, dirLocal, path.Join(tmpDirRemote, path.Base(dirLocal)))
}

func TestServerGetRecursive(t *testing.T) {
	listenerGo, hostGo, portGo := testServer(t, GOLANG_SFTP, READONLY)
	defer listenerGo.Close()

	dirRemote, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	tmpDirLocal := "/tmp/" + randName()
	defer os.RemoveAll(tmpDirLocal)

	t.Logf("get recursive: local %v remote %v", tmpDirLocal, dirRemote)

	// pull this directory (source code etc) recursively from the server
	if output, err := runSftpClient(t, "lmkdir "+tmpDirLocal+"\r\nget -r -P "+dirRemote+"/ "+tmpDirLocal+"/", "/", hostGo, portGo); err != nil {
		t.Fatalf("runSftpClient failed: %v, output\n%v\n", err, output)
	}

	compareDirectoriesRecursive(t, dirRemote, path.Join(tmpDirLocal, path.Base(dirRemote)))
}
