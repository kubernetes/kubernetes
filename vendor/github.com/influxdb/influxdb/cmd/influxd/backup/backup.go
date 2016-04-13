package backup

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"os"

	"github.com/influxdb/influxdb/services/snapshotter"
	"github.com/influxdb/influxdb/snapshot"
)

// Suffix is a suffix added to the backup while it's in-process.
const Suffix = ".pending"

// Command represents the program execution for "influxd backup".
type Command struct {
	// The logger passed to the ticker during execution.
	Logger *log.Logger

	// Standard input/output, overridden for testing.
	Stderr io.Writer
}

// NewCommand returns a new instance of Command with default settings.
func NewCommand() *Command {
	return &Command{
		Stderr: os.Stderr,
	}
}

// Run executes the program.
func (cmd *Command) Run(args ...string) error {
	// Set up logger.
	cmd.Logger = log.New(cmd.Stderr, "", log.LstdFlags)
	cmd.Logger.Printf("influxdb backup")

	// Parse command line arguments.
	host, path, err := cmd.parseFlags(args)
	if err != nil {
		return err
	}

	// Retrieve snapshot from local file.
	m, err := snapshot.ReadFileManifest(path)
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("read file snapshot: %s", err)
	}

	// Determine temporary path to download to.
	tmppath := path + Suffix

	// Calculate path of next backup file.
	// This uses the path if it doesn't exist.
	// Otherwise it appends an autoincrementing number.
	path, err = cmd.nextPath(path)
	if err != nil {
		return fmt.Errorf("next path: %s", err)
	}

	// Retrieve snapshot.
	if err := cmd.download(host, m, tmppath); err != nil {
		return fmt.Errorf("download: %s", err)
	}

	// Rename temporary file to final path.
	if err := os.Rename(tmppath, path); err != nil {
		return fmt.Errorf("rename: %s", err)
	}

	// TODO: Check file integrity.

	// Notify user of completion.
	cmd.Logger.Println("backup complete")

	return nil
}

// parseFlags parses and validates the command line arguments.
func (cmd *Command) parseFlags(args []string) (host string, path string, err error) {
	fs := flag.NewFlagSet("", flag.ContinueOnError)
	fs.StringVar(&host, "host", "localhost:8088", "")
	fs.SetOutput(cmd.Stderr)
	fs.Usage = cmd.printUsage
	if err := fs.Parse(args); err != nil {
		return "", "", err
	}

	// Ensure that only one arg is specified.
	if fs.NArg() == 0 {
		return "", "", errors.New("snapshot path required")
	} else if fs.NArg() != 1 {
		return "", "", errors.New("only one snapshot path allowed")
	}
	path = fs.Arg(0)

	return host, path, nil
}

// nextPath returns the next file to write to.
func (cmd *Command) nextPath(path string) (string, error) {
	// Use base path if it doesn't exist.
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return path, nil
	} else if err != nil {
		return "", err
	}

	// Otherwise iterate through incremental files until one is available.
	for i := 0; ; i++ {
		s := fmt.Sprintf(path+".%d", i)
		if _, err := os.Stat(s); os.IsNotExist(err) {
			return s, nil
		} else if err != nil {
			return "", err
		}
	}
}

// download downloads a snapshot from a host to a given path.
func (cmd *Command) download(host string, m *snapshot.Manifest, path string) error {
	// Create local file to write to.
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("open temp file: %s", err)
	}
	defer f.Close()

	// Connect to snapshotter service.
	conn, err := net.Dial("tcp", host)
	if err != nil {
		return err
	}
	defer conn.Close()

	// Send snapshotter marker byte.
	if _, err := conn.Write([]byte{snapshotter.MuxHeader}); err != nil {
		return fmt.Errorf("write snapshot header byte: %s", err)
	}

	// Write the manifest we currently have.
	if err := json.NewEncoder(conn).Encode(m); err != nil {
		return fmt.Errorf("encode snapshot manifest: %s", err)
	}

	// Read snapshot from the connection.
	if _, err := io.Copy(f, conn); err != nil {
		return fmt.Errorf("copy snapshot to file: %s", err)
	}

	// FIXME(benbjohnson): Verify integrity of snapshot.

	return nil
}

// printUsage prints the usage message to STDERR.
func (cmd *Command) printUsage() {
	fmt.Fprintf(cmd.Stderr, `usage: influxd backup [flags] PATH

backup downloads a snapshot of a data node and saves it to disk.

        -host <host:port>
                          The host to connect to snapshot.
                          Defaults to 127.0.0.1:8088.
`)
}
