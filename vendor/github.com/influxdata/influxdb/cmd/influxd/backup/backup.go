package backup

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/influxdata/influxdb/services/snapshotter"
	"github.com/influxdata/influxdb/tcp"
)

const (
	// Suffix is a suffix added to the backup while it's in-process.
	Suffix = ".pending"

	// Metafile is the base name given to the metastore backups.
	Metafile = "meta"

	// BackupFilePattern is the beginning of the pattern for a backup
	// file. They follow the scheme <database>.<retention>.<shardID>.<increment>
	BackupFilePattern = "%s.%s.%05d"
)

// Command represents the program execution for "influxd backup".
type Command struct {
	// The logger passed to the ticker during execution.
	Logger *log.Logger

	// Standard input/output, overridden for testing.
	Stderr io.Writer
	Stdout io.Writer

	host     string
	path     string
	database string
}

// NewCommand returns a new instance of Command with default settings.
func NewCommand() *Command {
	return &Command{
		Stderr: os.Stderr,
		Stdout: os.Stdout,
	}
}

// Run executes the program.
func (cmd *Command) Run(args ...string) error {
	// Set up logger.
	cmd.Logger = log.New(cmd.Stderr, "", log.LstdFlags)

	// Parse command line arguments.
	retentionPolicy, shardID, since, err := cmd.parseFlags(args)
	if err != nil {
		return err
	}

	// based on the arguments passed in we only backup the minimum
	if shardID != "" {
		// always backup the metastore
		if err := cmd.backupMetastore(); err != nil {
			return err
		}
		err = cmd.backupShard(retentionPolicy, shardID, since)
	} else if retentionPolicy != "" {
		err = cmd.backupRetentionPolicy(retentionPolicy, since)
	} else if cmd.database != "" {
		err = cmd.backupDatabase(since)
	} else {
		err = cmd.backupMetastore()
	}

	if err != nil {
		cmd.Logger.Printf("backup failed: %v", err)
		return err
	}

	cmd.Logger.Println("backup complete")

	return nil
}

// parseFlags parses and validates the command line arguments into a request object.
func (cmd *Command) parseFlags(args []string) (retentionPolicy, shardID string, since time.Time, err error) {
	fs := flag.NewFlagSet("", flag.ContinueOnError)

	fs.StringVar(&cmd.host, "host", "localhost:8088", "")
	fs.StringVar(&cmd.database, "database", "", "")
	fs.StringVar(&retentionPolicy, "retention", "", "")
	fs.StringVar(&shardID, "shard", "", "")
	var sinceArg string
	fs.StringVar(&sinceArg, "since", "", "")

	fs.SetOutput(cmd.Stderr)
	fs.Usage = cmd.printUsage

	err = fs.Parse(args)
	if err != nil {
		return
	}
	if sinceArg != "" {
		since, err = time.Parse(time.RFC3339, sinceArg)
		if err != nil {
			return
		}
	}

	// Ensure that only one arg is specified.
	if fs.NArg() == 0 {
		return "", "", time.Unix(0, 0), errors.New("backup destination path required")
	} else if fs.NArg() != 1 {
		return "", "", time.Unix(0, 0), errors.New("only one backup path allowed")
	}
	cmd.path = fs.Arg(0)

	err = os.MkdirAll(cmd.path, 0700)

	return
}

// backupShard will write a tar archive of the passed in shard with any TSM files that have been
// created since the time passed in
func (cmd *Command) backupShard(retentionPolicy string, shardID string, since time.Time) error {
	id, err := strconv.ParseUint(shardID, 10, 64)
	if err != nil {
		return err
	}

	shardArchivePath, err := cmd.nextPath(filepath.Join(cmd.path, fmt.Sprintf(BackupFilePattern, cmd.database, retentionPolicy, id)))
	if err != nil {
		return err
	}

	cmd.Logger.Printf("backing up db=%v rp=%v shard=%v to %s since %s",
		cmd.database, retentionPolicy, shardID, shardArchivePath, since)

	req := &snapshotter.Request{
		Type:            snapshotter.RequestShardBackup,
		Database:        cmd.database,
		RetentionPolicy: retentionPolicy,
		ShardID:         id,
		Since:           since,
	}

	// TODO: verify shard backup data
	return cmd.downloadAndVerify(req, shardArchivePath, nil)
}

// backupDatabase will request the database information from the server and then backup the metastore and
// every shard in every retention policy in the database. Each shard will be written to a separate tar.
func (cmd *Command) backupDatabase(since time.Time) error {
	cmd.Logger.Printf("backing up db=%s since %s", cmd.database, since)

	req := &snapshotter.Request{
		Type:     snapshotter.RequestDatabaseInfo,
		Database: cmd.database,
	}

	response, err := cmd.requestInfo(req)
	if err != nil {
		return err
	}

	return cmd.backupResponsePaths(response, since)
}

// backupRetentionPolicy will request the retention policy information from the server and then backup
// the metastore and every shard in the retention policy. Each shard will be written to a separate tar.
func (cmd *Command) backupRetentionPolicy(retentionPolicy string, since time.Time) error {
	cmd.Logger.Printf("backing up rp=%s since %s", retentionPolicy, since)

	req := &snapshotter.Request{
		Type:            snapshotter.RequestRetentionPolicyInfo,
		Database:        cmd.database,
		RetentionPolicy: retentionPolicy,
	}

	response, err := cmd.requestInfo(req)
	if err != nil {
		return err
	}

	return cmd.backupResponsePaths(response, since)
}

// backupResponsePaths will backup the metastore and all shard paths in the response struct
func (cmd *Command) backupResponsePaths(response *snapshotter.Response, since time.Time) error {
	if err := cmd.backupMetastore(); err != nil {
		return err
	}

	// loop through the returned paths and back up each shard
	for _, path := range response.Paths {
		rp, id, err := retentionAndShardFromPath(path)
		if err != nil {
			return err
		}

		if err := cmd.backupShard(rp, id, since); err != nil {
			return err
		}
	}

	return nil
}

// backupMetastore will backup the metastore on the host to the passed in path. Database and retention policy backups
// will force a backup of the metastore as well as requesting a specific shard backup from the command line
func (cmd *Command) backupMetastore() error {
	metastoreArchivePath, err := cmd.nextPath(filepath.Join(cmd.path, Metafile))
	if err != nil {
		return err
	}

	cmd.Logger.Printf("backing up metastore to %s", metastoreArchivePath)

	req := &snapshotter.Request{
		Type: snapshotter.RequestMetastoreBackup,
	}

	return cmd.downloadAndVerify(req, metastoreArchivePath, func(file string) error {
		binData, err := ioutil.ReadFile(file)
		if err != nil {
			return err
		}

		magic := binary.BigEndian.Uint64(binData[:8])
		if magic != snapshotter.BackupMagicHeader {
			cmd.Logger.Println("Invalid metadata blob, ensure the metadata service is running (default port 8088)")
			return errors.New("invalid metadata received")
		}

		return nil
	})
}

// nextPath returns the next file to write to.
func (cmd *Command) nextPath(path string) (string, error) {
	// Iterate through incremental files until one is available.
	for i := 0; ; i++ {
		s := fmt.Sprintf(path+".%02d", i)
		if _, err := os.Stat(s); os.IsNotExist(err) {
			return s, nil
		} else if err != nil {
			return "", err
		}
	}
}

// downloadAndVerify will download either the metastore or shard to a temp file and then
// rename it to a good backup file name after complete
func (cmd *Command) downloadAndVerify(req *snapshotter.Request, path string, validator func(string) error) error {
	tmppath := path + Suffix
	if err := cmd.download(req, tmppath); err != nil {
		return err
	}

	if validator != nil {
		if err := validator(tmppath); err != nil {
			if rmErr := os.Remove(tmppath); rmErr != nil {
				cmd.Logger.Printf("Error cleaning up temporary file: %v", rmErr)
			}
			return err
		}
	}

	f, err := os.Stat(tmppath)
	if err != nil {
		return err
	}

	// There was nothing downloaded, don't create an empty backup file.
	if f.Size() == 0 {
		return os.Remove(tmppath)
	}

	// Rename temporary file to final path.
	if err := os.Rename(tmppath, path); err != nil {
		return fmt.Errorf("rename: %s", err)
	}

	return nil
}

// download downloads a snapshot of either the metastore or a shard from a host to a given path.
func (cmd *Command) download(req *snapshotter.Request, path string) error {
	// Create local file to write to.
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("open temp file: %s", err)
	}
	defer f.Close()

	// Connect to snapshotter service.
	conn, err := tcp.Dial("tcp", cmd.host, snapshotter.MuxHeader)
	if err != nil {
		return err
	}
	defer conn.Close()

	// Write the request
	if err := json.NewEncoder(conn).Encode(req); err != nil {
		return fmt.Errorf("encode snapshot request: %s", err)
	}

	// Read snapshot from the connection
	if _, err := io.Copy(f, conn); err != nil {
		return fmt.Errorf("copy backup to file: %s", err)
	}

	return nil
}

// requestInfo will request the database or retention policy information from the host
func (cmd *Command) requestInfo(request *snapshotter.Request) (*snapshotter.Response, error) {
	// Connect to snapshotter service.
	conn, err := tcp.Dial("tcp", cmd.host, snapshotter.MuxHeader)
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	// Write the request
	if err := json.NewEncoder(conn).Encode(request); err != nil {
		return nil, fmt.Errorf("encode snapshot request: %s", err)
	}

	// Read the response
	var r snapshotter.Response
	if err := json.NewDecoder(conn).Decode(&r); err != nil {
		return nil, err
	}

	return &r, nil
}

// printUsage prints the usage message to STDERR.
func (cmd *Command) printUsage() {
	fmt.Fprintf(cmd.Stdout, `Downloads a snapshot of a data node and saves it to disk.

Usage: influxd backup [flags] PATH

    -host <host:port>
            The host to connect to snapshot. Defaults to 127.0.0.1:8088.
    -database <name>
            The database to backup.
    -retention <name>
            Optional. The retention policy to backup.
    -shard <id>
            Optional. The shard id to backup. If specified, retention is required.
    -since <2015-12-24T08:12:23>
            Optional. Do an incremental backup since the passed in RFC3339
            formatted time.

`)
}

// retentionAndShardFromPath will take the shard relative path and split it into the
// retention policy name and shard ID. The first part of the path should be the database name.
func retentionAndShardFromPath(path string) (retention, shard string, err error) {
	a := strings.Split(path, string(filepath.Separator))
	if len(a) != 3 {
		return "", "", fmt.Errorf("expected database, retention policy, and shard id in path: %s", path)
	}

	return a[1], a[2], nil
}
