package testutil

import (
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestIsKilledFalseWithNonKilledProcess(t *testing.T) {
	var lsCmd *exec.Cmd
	if runtime.GOOS != "windows" {
		lsCmd = exec.Command("ls")
	} else {
		lsCmd = exec.Command("cmd", "/c", "dir")
	}

	err := lsCmd.Run()
	if IsKilled(err) {
		t.Fatalf("Expected the ls command to not be killed, was.")
	}
}

func TestIsKilledTrueWithKilledProcess(t *testing.T) {
	var longCmd *exec.Cmd
	if runtime.GOOS != "windows" {
		longCmd = exec.Command("top")
	} else {
		longCmd = exec.Command("powershell", "while ($true) { sleep 1 }")
	}

	// Start a command
	err := longCmd.Start()
	if err != nil {
		t.Fatal(err)
	}
	// Capture the error when *dying*
	done := make(chan error, 1)
	go func() {
		done <- longCmd.Wait()
	}()
	// Then kill it
	longCmd.Process.Kill()
	// Get the error
	err = <-done
	if !IsKilled(err) {
		t.Fatalf("Expected the command to be killed, was not.")
	}
}

func TestRunCommandPipelineWithOutputWithNotEnoughCmds(t *testing.T) {
	_, _, err := RunCommandPipelineWithOutput(exec.Command("ls"))
	expectedError := "pipeline does not have multiple cmds"
	if err == nil || err.Error() != expectedError {
		t.Fatalf("Expected an error with %s, got err:%s", expectedError, err)
	}
}

func TestRunCommandPipelineWithOutputErrors(t *testing.T) {
	p := "$PATH"
	if runtime.GOOS == "windows" {
		p = "%PATH%"
	}
	cmd1 := exec.Command("ls")
	cmd1.Stdout = os.Stdout
	cmd2 := exec.Command("anything really")
	_, _, err := RunCommandPipelineWithOutput(cmd1, cmd2)
	if err == nil || err.Error() != "cannot set stdout pipe for anything really: exec: Stdout already set" {
		t.Fatalf("Expected an error, got %v", err)
	}

	cmdWithError := exec.Command("doesnotexists")
	cmdCat := exec.Command("cat")
	_, _, err = RunCommandPipelineWithOutput(cmdWithError, cmdCat)
	if err == nil || err.Error() != `starting doesnotexists failed with error: exec: "doesnotexists": executable file not found in `+p {
		t.Fatalf("Expected an error, got %v", err)
	}
}

func TestRunCommandPipelineWithOutput(t *testing.T) {
	//TODO: Should run on Solaris
	if runtime.GOOS == "solaris" {
		t.Skip()
	}
	cmds := []*exec.Cmd{
		// Print 2 characters
		exec.Command("echo", "-n", "11"),
		// Count the number or char from stdin (previous command)
		exec.Command("wc", "-m"),
	}
	out, exitCode, err := RunCommandPipelineWithOutput(cmds...)
	expectedOutput := "2\n"
	if out != expectedOutput || exitCode != 0 || err != nil {
		t.Fatalf("Expected %s for commands %v, got out:%s, exitCode:%d, err:%v", expectedOutput, cmds, out, exitCode, err)
	}
}

func TestConvertSliceOfStringsToMap(t *testing.T) {
	input := []string{"a", "b"}
	actual := ConvertSliceOfStringsToMap(input)
	for _, key := range input {
		if _, ok := actual[key]; !ok {
			t.Fatalf("Expected output to contains key %s, did not: %v", key, actual)
		}
	}
}

func TestCompareDirectoryEntries(t *testing.T) {
	tmpFolder, err := ioutil.TempDir("", "integration-cli-utils-compare-directories")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpFolder)

	file1 := filepath.Join(tmpFolder, "file1")
	file2 := filepath.Join(tmpFolder, "file2")
	os.Create(file1)
	os.Create(file2)

	fi1, err := os.Stat(file1)
	if err != nil {
		t.Fatal(err)
	}
	fi1bis, err := os.Stat(file1)
	if err != nil {
		t.Fatal(err)
	}
	fi2, err := os.Stat(file2)
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		e1          []os.FileInfo
		e2          []os.FileInfo
		shouldError bool
	}{
		// Empty directories
		{
			[]os.FileInfo{},
			[]os.FileInfo{},
			false,
		},
		// Same FileInfos
		{
			[]os.FileInfo{fi1},
			[]os.FileInfo{fi1},
			false,
		},
		// Different FileInfos but same names
		{
			[]os.FileInfo{fi1},
			[]os.FileInfo{fi1bis},
			false,
		},
		// Different FileInfos, different names
		{
			[]os.FileInfo{fi1},
			[]os.FileInfo{fi2},
			true,
		},
	}
	for _, elt := range cases {
		err := CompareDirectoryEntries(elt.e1, elt.e2)
		if elt.shouldError && err == nil {
			t.Fatalf("Should have return an error, did not with %v and %v", elt.e1, elt.e2)
		}
		if !elt.shouldError && err != nil {
			t.Fatalf("Should have not returned an error, but did : %v with %v and %v", err, elt.e1, elt.e2)
		}
	}
}

// FIXME make an "unhappy path" test for ListTar without "panicking" :-)
func TestListTar(t *testing.T) {
	// TODO Windows: Figure out why this fails. Should be portable.
	if runtime.GOOS == "windows" {
		t.Skip("Failing on Windows - needs further investigation")
	}
	tmpFolder, err := ioutil.TempDir("", "integration-cli-utils-list-tar")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpFolder)

	// Let's create a Tar file
	srcFile := filepath.Join(tmpFolder, "src")
	tarFile := filepath.Join(tmpFolder, "src.tar")
	os.Create(srcFile)
	cmd := exec.Command("sh", "-c", "tar cf "+tarFile+" "+srcFile)
	_, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}

	reader, err := os.Open(tarFile)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()

	entries, err := ListTar(reader)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 && entries[0] != "src" {
		t.Fatalf("Expected a tar file with 1 entry (%s), got %v", srcFile, entries)
	}
}

func TestRandomTmpDirPath(t *testing.T) {
	path := RandomTmpDirPath("something", runtime.GOOS)

	prefix := "/tmp/something"
	if runtime.GOOS == "windows" {
		prefix = os.Getenv("TEMP") + `\something`
	}
	expectedSize := len(prefix) + 11

	if !strings.HasPrefix(path, prefix) {
		t.Fatalf("Expected generated path to have '%s' as prefix, got %s'", prefix, path)
	}
	if len(path) != expectedSize {
		t.Fatalf("Expected generated path to be %d, got %d", expectedSize, len(path))
	}
}

func TestConsumeWithSpeed(t *testing.T) {
	reader := strings.NewReader("1234567890")
	chunksize := 2

	bytes1, err := ConsumeWithSpeed(reader, chunksize, 10*time.Millisecond, nil)
	if err != nil {
		t.Fatal(err)
	}

	if bytes1 != 10 {
		t.Fatalf("Expected to have read 10 bytes, got %d", bytes1)
	}

}

func TestConsumeWithSpeedWithStop(t *testing.T) {
	reader := strings.NewReader("1234567890")
	chunksize := 2

	stopIt := make(chan bool)

	go func() {
		time.Sleep(1 * time.Millisecond)
		stopIt <- true
	}()

	bytes1, err := ConsumeWithSpeed(reader, chunksize, 20*time.Millisecond, stopIt)
	if err != nil {
		t.Fatal(err)
	}

	if bytes1 != 2 {
		t.Fatalf("Expected to have read 2 bytes, got %d", bytes1)
	}

}

func TestParseCgroupPathsEmpty(t *testing.T) {
	cgroupMap := ParseCgroupPaths("")
	if len(cgroupMap) != 0 {
		t.Fatalf("Expected an empty map, got %v", cgroupMap)
	}
	cgroupMap = ParseCgroupPaths("\n")
	if len(cgroupMap) != 0 {
		t.Fatalf("Expected an empty map, got %v", cgroupMap)
	}
	cgroupMap = ParseCgroupPaths("something:else\nagain:here")
	if len(cgroupMap) != 0 {
		t.Fatalf("Expected an empty map, got %v", cgroupMap)
	}
}

func TestParseCgroupPaths(t *testing.T) {
	cgroupMap := ParseCgroupPaths("2:memory:/a\n1:cpuset:/b")
	if len(cgroupMap) != 2 {
		t.Fatalf("Expected a map with 2 entries, got %v", cgroupMap)
	}
	if value, ok := cgroupMap["memory"]; !ok || value != "/a" {
		t.Fatalf("Expected cgroupMap to contains an entry for 'memory' with value '/a', got %v", cgroupMap)
	}
	if value, ok := cgroupMap["cpuset"]; !ok || value != "/b" {
		t.Fatalf("Expected cgroupMap to contains an entry for 'cpuset' with value '/b', got %v", cgroupMap)
	}
}

func TestChannelBufferTimeout(t *testing.T) {
	expected := "11"

	buf := &ChannelBuffer{make(chan []byte, 1)}
	defer buf.Close()

	done := make(chan struct{}, 1)
	go func() {
		time.Sleep(100 * time.Millisecond)
		io.Copy(buf, strings.NewReader(expected))
		done <- struct{}{}
	}()

	// Wait long enough
	b := make([]byte, 2)
	_, err := buf.ReadTimeout(b, 50*time.Millisecond)
	if err == nil && err.Error() != "timeout reading from channel" {
		t.Fatalf("Expected an error, got %s", err)
	}
	<-done
}

func TestChannelBuffer(t *testing.T) {
	expected := "11"

	buf := &ChannelBuffer{make(chan []byte, 1)}
	defer buf.Close()

	go func() {
		time.Sleep(100 * time.Millisecond)
		io.Copy(buf, strings.NewReader(expected))
	}()

	// Wait long enough
	b := make([]byte, 2)
	_, err := buf.ReadTimeout(b, 200*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}

	if string(b) != expected {
		t.Fatalf("Expected '%s', got '%s'", expected, string(b))
	}
}
