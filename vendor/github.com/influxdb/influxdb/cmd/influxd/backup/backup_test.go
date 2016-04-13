package backup_test

/*
import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/influxdb/influxdb"
	"github.com/influxdb/influxdb/cmd/influxd"
)

// Ensure the backup can download from the server and save to disk.
func TestBackupCommand(t *testing.T) {
	// Mock the backup endpoint.
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/data/snapshot" {
			t.Fatalf("unexpected url path: %s", r.URL.Path)
		}

		// Write a simple snapshot to the buffer.
		sw := influxdb.NewSnapshotWriter()
		sw.Snapshot = &influxdb.Snapshot{Files: []influxdb.SnapshotFile{
			{Name: "meta", Size: 5, Index: 10},
		}}
		sw.FileWriters["meta"] = influxdb.NopWriteToCloser(bytes.NewBufferString("55555"))
		if _, err := sw.WriteTo(w); err != nil {
			t.Fatal(err)
		}
	}))
	defer s.Close()

	// Create a temp path and remove incremental backups at the end.
	path := tempfile()
	defer os.Remove(path)
	defer os.Remove(path + ".0")
	defer os.Remove(path + ".1")

	// Execute the backup against the mock server.
	for i := 0; i < 3; i++ {
		if err := NewBackupCommand().Run("-host", s.URL, path); err != nil {
			t.Fatal(err)
		}
	}

	// Verify snapshot and two incremental snapshots were written.
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("snapshot not found: %s", err)
	} else if _, err = os.Stat(path + ".0"); err != nil {
		t.Fatalf("incremental snapshot(0) not found: %s", err)
	} else if _, err = os.Stat(path + ".1"); err != nil {
		t.Fatalf("incremental snapshot(1) not found: %s", err)
	}
}

// Ensure the backup command returns an error if flags cannot be parsed.
func TestBackupCommand_ErrFlagParse(t *testing.T) {
	cmd := NewBackupCommand()
	if err := cmd.Run("-bad-flag"); err == nil || err.Error() != `flag provided but not defined: -bad-flag` {
		t.Fatal(err)
	} else if !strings.Contains(cmd.Stderr.String(), "usage") {
		t.Fatal("usage message not displayed")
	}
}

// Ensure the backup command returns an error if the host cannot be parsed.
func TestBackupCommand_ErrInvalidHostURL(t *testing.T) {
	if err := NewBackupCommand().Run("-host", "http://%f"); err == nil || err.Error() != `parse host url: parse http://%f: hexadecimal escape in host` {
		t.Fatal(err)
	}
}

// Ensure the backup command returns an error if the output path is not specified.
func TestBackupCommand_ErrPathRequired(t *testing.T) {
	if err := NewBackupCommand().Run("-host", "//localhost"); err == nil || err.Error() != `snapshot path required` {
		t.Fatal(err)
	}
}

// Ensure the backup returns an error if it cannot connect to the server.
func TestBackupCommand_ErrConnectionRefused(t *testing.T) {
	// Start and immediately stop a server so we have a dead port.
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	s.Close()

	// Execute the backup command.
	path := tempfile()
	defer os.Remove(path)
	if err := NewBackupCommand().Run("-host", s.URL, path); err == nil ||
		!(strings.Contains(err.Error(), `connection refused`) || strings.Contains(err.Error(), `No connection could be made`)) {
		t.Fatal(err)
	}
}

// Ensure the backup returns any non-200 status codes.
func TestBackupCommand_ErrServerError(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer s.Close()

	// Execute the backup command.
	path := tempfile()
	defer os.Remove(path)
	if err := NewBackupCommand().Run("-host", s.URL, path); err == nil || err.Error() != `download: snapshot error: status=500` {
		t.Fatal(err)
	}
}

// BackupCommand is a test wrapper for main.BackupCommand.
type BackupCommand struct {
	*main.BackupCommand
	Stderr bytes.Buffer
}

// NewBackupCommand returns a new instance of BackupCommand.
func NewBackupCommand() *BackupCommand {
	cmd := &BackupCommand{BackupCommand: main.NewBackupCommand()}
	cmd.BackupCommand.Stderr = &cmd.Stderr
	return cmd
}
*/
