package restore

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"

	"github.com/BurntSushi/toml"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/snapshot"
	"github.com/influxdb/influxdb/tsdb"
)

// Command represents the program execution for "influxd restore".
type Command struct {
	Stdout io.Writer
	Stderr io.Writer
}

// NewCommand returns a new instance of Command with default settings.
func NewCommand() *Command {
	return &Command{
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	}
}

// Run executes the program.
func (cmd *Command) Run(args ...string) error {
	config, path, err := cmd.parseFlags(args)
	if err != nil {
		return err
	}

	return cmd.Restore(config, path)
}

// Restore restores a database snapshot
func (cmd *Command) Restore(config *Config, path string) error {
	// Remove meta and data directories.
	if err := os.RemoveAll(config.Meta.Dir); err != nil {
		return fmt.Errorf("remove meta dir: %s", err)
	} else if err := os.RemoveAll(config.Data.Dir); err != nil {
		return fmt.Errorf("remove data dir: %s", err)
	}

	// Open snapshot file and all incremental backups.
	mr, files, err := snapshot.OpenFileMultiReader(path)
	if err != nil {
		return fmt.Errorf("open multireader: %s", err)
	}
	defer closeAll(files)

	// Unpack files from archive.
	if err := cmd.unpack(mr, config); err != nil {
		return fmt.Errorf("unpack: %s", err)
	}

	// Notify user of completion.
	fmt.Fprintf(os.Stdout, "restore complete using %s", path)
	return nil
}

// parseFlags parses and validates the command line arguments.
func (cmd *Command) parseFlags(args []string) (*Config, string, error) {
	fs := flag.NewFlagSet("", flag.ContinueOnError)
	configPath := fs.String("config", "", "")
	fs.SetOutput(cmd.Stderr)
	fs.Usage = cmd.printUsage
	if err := fs.Parse(args); err != nil {
		return nil, "", err
	}

	// Parse configuration file from disk.
	if *configPath == "" {
		return nil, "", fmt.Errorf("config required")
	}

	// Parse config.
	config := Config{
		Meta: meta.NewConfig(),
		Data: tsdb.NewConfig(),
	}
	if _, err := toml.DecodeFile(*configPath, &config); err != nil {
		return nil, "", err
	}

	// Require output path.
	path := fs.Arg(0)
	if path == "" {
		return nil, "", fmt.Errorf("snapshot path required")
	}

	return &config, path, nil
}

func closeAll(a []io.Closer) {
	for _, c := range a {
		_ = c.Close()
	}
}

// unpack expands the files in the snapshot archive into a directory.
func (cmd *Command) unpack(mr *snapshot.MultiReader, config *Config) error {
	// Loop over files and extract.
	for {
		// Read entry header.
		sf, err := mr.Next()
		if err == io.EOF {
			break
		} else if err != nil {
			return fmt.Errorf("next: entry=%s, err=%s", sf.Name, err)
		}

		// Log progress.
		fmt.Fprintf(os.Stdout, "unpacking: %s (%d bytes)\n", sf.Name, sf.Size)

		// Handle meta and tsdb files separately.
		switch sf.Name {
		case "meta":
			if err := cmd.unpackMeta(mr, sf, config); err != nil {
				return fmt.Errorf("meta: %s", err)
			}
		default:
			if err := cmd.unpackData(mr, sf, config); err != nil {
				return fmt.Errorf("data: %s", err)
			}
		}
	}

	return nil
}

// unpackMeta reads the metadata from the snapshot and initializes a raft
// cluster and replaces the root metadata.
func (cmd *Command) unpackMeta(mr *snapshot.MultiReader, sf snapshot.File, config *Config) error {
	// Read meta into buffer.
	var buf bytes.Buffer
	if _, err := io.CopyN(&buf, mr, sf.Size); err != nil {
		return fmt.Errorf("copy: %s", err)
	}

	// Unpack into metadata.
	var data meta.Data
	if err := data.UnmarshalBinary(buf.Bytes()); err != nil {
		return fmt.Errorf("unmarshal: %s", err)
	}

	// Copy meta config and remove peers so it starts in single mode.
	c := config.Meta
	c.Peers = nil

	// Initialize meta store.
	store := meta.NewStore(config.Meta)
	store.RaftListener = newNopListener()
	store.ExecListener = newNopListener()
	store.RPCListener = newNopListener()

	// Determine advertised address.
	_, port, err := net.SplitHostPort(config.Meta.BindAddress)
	if err != nil {
		return fmt.Errorf("split bind address: %s", err)
	}
	hostport := net.JoinHostPort(config.Meta.Hostname, port)

	// Resolve address.
	addr, err := net.ResolveTCPAddr("tcp", hostport)
	if err != nil {
		return fmt.Errorf("resolve tcp: addr=%s, err=%s", hostport, err)
	}
	store.Addr = addr
	store.RemoteAddr = addr

	// Open the meta store.
	if err := store.Open(); err != nil {
		return fmt.Errorf("open store: %s", err)
	}
	defer store.Close()

	// Wait for the store to be ready or error.
	select {
	case <-store.Ready():
	case err := <-store.Err():
		return err
	}

	// Force set the full metadata.
	if err := store.SetData(&data); err != nil {
		return fmt.Errorf("set data: %s", err)
	}

	return nil
}

func (cmd *Command) unpackData(mr *snapshot.MultiReader, sf snapshot.File, config *Config) error {
	path := filepath.Join(config.Data.Dir, sf.Name)
	// Create parent directory for output file.
	if err := os.MkdirAll(filepath.Dir(path), 0777); err != nil {
		return fmt.Errorf("mkdir: entry=%s, err=%s", sf.Name, err)
	}

	// Create output file.
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create: entry=%s, err=%s", sf.Name, err)
	}
	defer f.Close()

	// Copy contents from reader.
	if _, err := io.CopyN(f, mr, sf.Size); err != nil {
		return fmt.Errorf("copy: entry=%s, err=%s", sf.Name, err)
	}

	return nil
}

// printUsage prints the usage message to STDERR.
func (cmd *Command) printUsage() {
	fmt.Fprintf(cmd.Stderr, `usage: influxd restore [flags] PATH

restore uses a snapshot of a data node to rebuild a cluster.

        -config <path>
                          Set the path to the configuration file.
`)
}

// Config represents a partial config for rebuilding the server.
type Config struct {
	Meta *meta.Config `toml:"meta"`
	Data tsdb.Config  `toml:"data"`
}

type nopListener struct {
	closing chan struct{}
}

func newNopListener() *nopListener {
	return &nopListener{make(chan struct{})}
}

func (ln *nopListener) Accept() (net.Conn, error) {
	<-ln.closing
	return nil, errors.New("listener closing")
}

func (ln *nopListener) Close() error {
	if ln.closing != nil {
		close(ln.closing)
		ln.closing = nil
	}
	return nil
}

func (ln *nopListener) Addr() net.Addr { return nil }
