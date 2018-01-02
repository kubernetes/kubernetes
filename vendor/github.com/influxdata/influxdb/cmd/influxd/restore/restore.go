package restore

import (
	"archive/tar"
	"bytes"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"sync"

	"github.com/influxdata/influxdb/cmd/influxd/backup"
	"github.com/influxdata/influxdb/services/meta"
	"github.com/influxdata/influxdb/services/snapshotter"
)

// Command represents the program execution for "influxd restore".
type Command struct {
	Stdout io.Writer
	Stderr io.Writer

	backupFilesPath string
	metadir         string
	datadir         string
	database        string
	retention       string
	shard           string

	// TODO: when the new meta stuff is done this should not be exported or be gone
	MetaConfig *meta.Config
}

// NewCommand returns a new instance of Command with default settings.
func NewCommand() *Command {
	return &Command{
		Stdout:     os.Stdout,
		Stderr:     os.Stderr,
		MetaConfig: meta.NewConfig(),
	}
}

// Run executes the program.
func (cmd *Command) Run(args ...string) error {
	if err := cmd.parseFlags(args); err != nil {
		return err
	}

	if cmd.metadir != "" {
		if err := cmd.unpackMeta(); err != nil {
			return err
		}
	}

	if cmd.shard != "" {
		return cmd.unpackShard(cmd.shard)
	} else if cmd.retention != "" {
		return cmd.unpackRetention()
	} else if cmd.datadir != "" {
		return cmd.unpackDatabase()
	}
	return nil
}

// parseFlags parses and validates the command line arguments.
func (cmd *Command) parseFlags(args []string) error {
	fs := flag.NewFlagSet("", flag.ContinueOnError)
	fs.StringVar(&cmd.metadir, "metadir", "", "")
	fs.StringVar(&cmd.datadir, "datadir", "", "")
	fs.StringVar(&cmd.database, "database", "", "")
	fs.StringVar(&cmd.retention, "retention", "", "")
	fs.StringVar(&cmd.shard, "shard", "", "")
	fs.SetOutput(cmd.Stdout)
	fs.Usage = cmd.printUsage
	if err := fs.Parse(args); err != nil {
		return err
	}

	cmd.MetaConfig = meta.NewConfig()
	cmd.MetaConfig.Dir = cmd.metadir

	// Require output path.
	cmd.backupFilesPath = fs.Arg(0)
	if cmd.backupFilesPath == "" {
		return fmt.Errorf("path with backup files required")
	}

	// validate the arguments
	if cmd.metadir == "" && cmd.database == "" {
		return fmt.Errorf("-metadir or -database are required to restore")
	}

	if cmd.database != "" && cmd.datadir == "" {
		return fmt.Errorf("-datadir is required to restore")
	}

	if cmd.shard != "" {
		if cmd.database == "" {
			return fmt.Errorf("-database is required to restore shard")
		}
		if cmd.retention == "" {
			return fmt.Errorf("-retention is required to restore shard")
		}
	} else if cmd.retention != "" && cmd.database == "" {
		return fmt.Errorf("-database is required to restore retention policy")
	}

	return nil
}

// unpackMeta reads the metadata from the backup directory and initializes a raft
// cluster and replaces the root metadata.
func (cmd *Command) unpackMeta() error {
	// find the meta file
	metaFiles, err := filepath.Glob(filepath.Join(cmd.backupFilesPath, backup.Metafile+".*"))
	if err != nil {
		return err
	}

	if len(metaFiles) == 0 {
		return fmt.Errorf("no metastore backups in %s", cmd.backupFilesPath)
	}

	latest := metaFiles[len(metaFiles)-1]

	fmt.Fprintf(cmd.Stdout, "Using metastore snapshot: %v\n", latest)
	// Read the metastore backup
	f, err := os.Open(latest)
	if err != nil {
		return err
	}

	var buf bytes.Buffer
	if _, err := io.Copy(&buf, f); err != nil {
		return fmt.Errorf("copy: %s", err)
	}

	b := buf.Bytes()
	var i int

	// Make sure the file is actually a meta store backup file
	magic := binary.BigEndian.Uint64(b[:8])
	if magic != snapshotter.BackupMagicHeader {
		return fmt.Errorf("invalid metadata file")
	}
	i += 8

	// Size of the meta store bytes
	length := int(binary.BigEndian.Uint64(b[i : i+8]))
	i += 8
	metaBytes := b[i : i+length]
	i += int(length)

	// Size of the node.json bytes
	length = int(binary.BigEndian.Uint64(b[i : i+8]))
	i += 8
	nodeBytes := b[i:]

	// Unpack into metadata.
	var data meta.Data
	if err := data.UnmarshalBinary(metaBytes); err != nil {
		return fmt.Errorf("unmarshal: %s", err)
	}

	// Copy meta config and remove peers so it starts in single mode.
	c := cmd.MetaConfig
	c.Dir = cmd.metadir

	// Create the meta dir
	if os.MkdirAll(c.Dir, 0700); err != nil {
		return err
	}

	// Write node.json back to meta dir
	if err := ioutil.WriteFile(filepath.Join(c.Dir, "node.json"), nodeBytes, 0655); err != nil {
		return err
	}

	client := meta.NewClient(c)
	client.SetLogOutput(ioutil.Discard)
	if err := client.Open(); err != nil {
		return err
	}
	defer client.Close()

	// Force set the full metadata.
	if err := client.SetData(&data); err != nil {
		return fmt.Errorf("set data: %s", err)
	}

	// remove the raft.db file if it exists
	err = os.Remove(filepath.Join(cmd.metadir, "raft.db"))
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	// remove the node.json file if it exists
	err = os.Remove(filepath.Join(cmd.metadir, "node.json"))
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	return nil
}

// unpackShard will look for all backup files in the path matching this shard ID
// and restore them to the data dir
func (cmd *Command) unpackShard(shardID string) error {
	// make sure the shard isn't already there so we don't clobber anything
	restorePath := filepath.Join(cmd.datadir, cmd.database, cmd.retention, shardID)
	if _, err := os.Stat(restorePath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("shard already present: %s", restorePath)
	}

	id, err := strconv.ParseUint(shardID, 10, 64)
	if err != nil {
		return err
	}

	// find the shard backup files
	pat := filepath.Join(cmd.backupFilesPath, fmt.Sprintf(backup.BackupFilePattern, cmd.database, cmd.retention, id))
	return cmd.unpackFiles(pat + ".*")
}

// unpackDatabase will look for all backup files in the path matching this database
// and restore them to the data dir
func (cmd *Command) unpackDatabase() error {
	// make sure the shard isn't already there so we don't clobber anything
	restorePath := filepath.Join(cmd.datadir, cmd.database)
	if _, err := os.Stat(restorePath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("database already present: %s", restorePath)
	}

	// find the database backup files
	pat := filepath.Join(cmd.backupFilesPath, cmd.database)
	return cmd.unpackFiles(pat + ".*")
}

// unpackRetention will look for all backup files in the path matching this retention
// and restore them to the data dir
func (cmd *Command) unpackRetention() error {
	// make sure the shard isn't already there so we don't clobber anything
	restorePath := filepath.Join(cmd.datadir, cmd.database, cmd.retention)
	if _, err := os.Stat(restorePath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("retention already present: %s", restorePath)
	}

	// find the retention backup files
	pat := filepath.Join(cmd.backupFilesPath, cmd.database)
	return cmd.unpackFiles(fmt.Sprintf("%s.%s.*", pat, cmd.retention))
}

// unpackFiles will look for backup files matching the pattern and restore them to the data dir
func (cmd *Command) unpackFiles(pat string) error {
	fmt.Printf("Restoring from backup %s\n", pat)

	backupFiles, err := filepath.Glob(pat)
	if err != nil {
		return err
	}

	if len(backupFiles) == 0 {
		return fmt.Errorf("no backup files for %s in %s", pat, cmd.backupFilesPath)
	}

	for _, fn := range backupFiles {
		if err := cmd.unpackTar(fn); err != nil {
			return err
		}
	}

	return nil
}

// unpackTar will restore a single tar archive to the data dir
func (cmd *Command) unpackTar(tarFile string) error {
	f, err := os.Open(tarFile)
	if err != nil {
		return err
	}
	defer f.Close()

	tr := tar.NewReader(f)

	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil
		} else if err != nil {
			return err
		}

		if err := cmd.unpackFile(tr, hdr.Name); err != nil {
			return err
		}
	}
}

// unpackFile will copy the current file from the tar archive to the data dir
func (cmd *Command) unpackFile(tr *tar.Reader, fileName string) error {
	fn := filepath.Join(cmd.datadir, fileName)
	fmt.Printf("unpacking %s\n", fn)

	if err := os.MkdirAll(filepath.Dir(fn), 0777); err != nil {
		return fmt.Errorf("error making restore dir: %s", err.Error())
	}

	ff, err := os.Create(fn)
	if err != nil {
		return err
	}
	defer ff.Close()

	if _, err := io.Copy(ff, tr); err != nil {
		return err
	}

	return nil
}

// printUsage prints the usage message to STDERR.
func (cmd *Command) printUsage() {
	fmt.Fprintf(cmd.Stdout, `Uses backups from the PATH to restore the metastore, databases,
retention policies, or specific shards. The InfluxDB process must not be
running during a restore.

Usage: influxd restore [flags] PATH

    -metadir <path>
            Optional. If set the metastore will be recovered to the given path.
    -datadir <path>
            Optional. If set the restore process will recover the specified
            database, retention policy or shard to the given directory.
    -database <name>
            Optional. Required if no metadir given. Will restore the database
            TSM files.
    -retention <name>
            Optional. If given, database is required. Will restore the retention policy's
            TSM files.
    -shard <id>
            Optional. If given, database and retention are required. Will restore the shard's
            TSM files.

`)
}

type nopListener struct {
	mu      sync.Mutex
	closing chan struct{}
}

func newNopListener() *nopListener {
	return &nopListener{closing: make(chan struct{})}
}

func (ln *nopListener) Accept() (net.Conn, error) {
	ln.mu.Lock()
	defer ln.mu.Unlock()

	<-ln.closing
	return nil, errors.New("listener closing")
}

func (ln *nopListener) Close() error {
	if ln.closing != nil {
		close(ln.closing)
		ln.mu.Lock()
		defer ln.mu.Unlock()

		ln.closing = nil
	}
	return nil
}

func (ln *nopListener) Addr() net.Addr { return &net.TCPAddr{} }
