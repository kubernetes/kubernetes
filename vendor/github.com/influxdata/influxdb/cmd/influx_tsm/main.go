package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"net/http"
	_ "net/http/pprof"

	"github.com/influxdata/influxdb/cmd/influx_tsm/b1"
	"github.com/influxdata/influxdb/cmd/influx_tsm/bz1"
	"github.com/influxdata/influxdb/cmd/influx_tsm/tsdb"
)

// ShardReader reads b* shards and converts to tsm shards
type ShardReader interface {
	KeyIterator
	Open() error
	Close() error
}

const (
	tsmExt = "tsm"
)

var description = `
Convert a database from b1 or bz1 format to tsm1 format.

This tool will backup the directories before conversion (if not disabled).
The backed-up files must be removed manually, generally after starting up the
node again to make sure all of data has been converted correctly.

To restore a backup:
  Shut down the node, remove the converted directory, and
  copy the backed-up directory to the original location.`

type options struct {
	DataPath       string
	BackupPath     string
	DBs            []string
	DebugAddr      string
	TSMSize        uint64
	Parallel       bool
	SkipBackup     bool
	UpdateInterval time.Duration
	Yes            bool
	CPUFile        string
}

func (o *options) Parse() error {
	fs := flag.NewFlagSet(os.Args[0], flag.ExitOnError)

	var dbs string

	fs.StringVar(&dbs, "dbs", "", "Comma-delimited list of databases to convert. Default is to convert all databases.")
	fs.Uint64Var(&opts.TSMSize, "sz", maxTSMSz, "Maximum size of individual TSM files.")
	fs.BoolVar(&opts.Parallel, "parallel", false, "Perform parallel conversion. (up to GOMAXPROCS shards at once)")
	fs.BoolVar(&opts.SkipBackup, "nobackup", false, "Disable database backups. Not recommended.")
	fs.StringVar(&opts.BackupPath, "backup", "", "The location to backup up the current databases. Must not be within the data directory.")
	fs.StringVar(&opts.DebugAddr, "debug", "", "If set, http debugging endpoints will be enabled on the given address")
	fs.DurationVar(&opts.UpdateInterval, "interval", 5*time.Second, "How often status updates are printed.")
	fs.BoolVar(&opts.Yes, "y", false, "Don't ask, just convert")
	fs.StringVar(&opts.CPUFile, "profile", "", "CPU Profile location")
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %v [options] <data-path> \n", os.Args[0])
		fmt.Fprintf(os.Stderr, "%v\n\nOptions:\n", description)
		fs.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\n")
	}

	if err := fs.Parse(os.Args[1:]); err != nil {
		return err
	}

	if len(fs.Args()) < 1 {
		return errors.New("no data directory specified")
	}
	var err error
	if o.DataPath, err = filepath.Abs(fs.Args()[0]); err != nil {
		return err
	}
	if o.DataPath, err = filepath.EvalSymlinks(filepath.Clean(o.DataPath)); err != nil {
		return err
	}

	if o.TSMSize > maxTSMSz {
		return fmt.Errorf("bad TSM file size, maximum TSM file size is %d", maxTSMSz)
	}

	// Check if specific databases were requested.
	o.DBs = strings.Split(dbs, ",")
	if len(o.DBs) == 1 && o.DBs[0] == "" {
		o.DBs = nil
	}

	if !o.SkipBackup {
		if o.BackupPath == "" {
			return errors.New("either -nobackup or -backup DIR must be set")
		}
		if o.BackupPath, err = filepath.Abs(o.BackupPath); err != nil {
			return err
		}
		if o.BackupPath, err = filepath.EvalSymlinks(filepath.Clean(o.BackupPath)); err != nil {
			if os.IsNotExist(err) {
				return errors.New("backup directory must already exist")
			}
			return err
		}

		if strings.HasPrefix(o.BackupPath, o.DataPath) {
			fmt.Println(o.BackupPath, o.DataPath)
			return errors.New("backup directory cannot be contained within data directory")
		}
	}

	if o.DebugAddr != "" {
		log.Printf("Starting debugging server on http://%v", o.DebugAddr)
		go func() {
			log.Fatal(http.ListenAndServe(o.DebugAddr, nil))
		}()
	}

	return nil
}

var opts options

const maxTSMSz uint64 = 2 * 1024 * 1024 * 1024

func init() {
	log.SetOutput(os.Stderr)
	log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds)
}

func main() {
	if err := opts.Parse(); err != nil {
		log.Fatal(err)
	}

	// Determine the list of databases
	dbs, err := ioutil.ReadDir(opts.DataPath)
	if err != nil {
		log.Fatalf("failed to access data directory at %v: %v\n", opts.DataPath, err)
	}
	fmt.Println() // Cleanly separate output from start of program.

	if opts.Parallel {
		if !isEnvSet("GOMAXPROCS") {
			// Only modify GOMAXPROCS if it wasn't set in the environment
			// This means 'GOMAXPROCS=1 influx_tsm -parallel' will not actually
			// run in parallel
			runtime.GOMAXPROCS(runtime.NumCPU())
		}
	}

	var badUser string
	if opts.SkipBackup {
		badUser = "(NOT RECOMMENDED)"
	}

	// Dump summary of what is about to happen.
	fmt.Println("b1 and bz1 shard conversion.")
	fmt.Println("-----------------------------------")
	fmt.Println("Data directory is:                 ", opts.DataPath)
	if !opts.SkipBackup {
		fmt.Println("Backup directory is:               ", opts.BackupPath)
	}
	fmt.Println("Databases specified:               ", allDBs(opts.DBs))
	fmt.Println("Database backups enabled:          ", yesno(!opts.SkipBackup), badUser)
	fmt.Printf("Parallel mode enabled (GOMAXPROCS): %s (%d)\n", yesno(opts.Parallel), runtime.GOMAXPROCS(0))
	fmt.Println()

	shards := collectShards(dbs)

	// Anything to convert?
	fmt.Printf("\nFound %d shards that will be converted.\n", len(shards))
	if len(shards) == 0 {
		fmt.Println("Nothing to do.")
		return
	}

	// Display list of convertible shards.
	fmt.Println()
	w := new(tabwriter.Writer)
	w.Init(os.Stdout, 0, 8, 1, '\t', 0)
	fmt.Fprintln(w, "Database\tRetention\tPath\tEngine\tSize")
	for _, si := range shards {
		fmt.Fprintf(w, "%v\t%v\t%v\t%v\t%d\n", si.Database, si.RetentionPolicy, si.FullPath(opts.DataPath), si.FormatAsString(), si.Size)
	}
	w.Flush()

	if !opts.Yes {
		// Get confirmation from user.
		fmt.Printf("\nThese shards will be converted. Proceed? y/N: ")
		liner := bufio.NewReader(os.Stdin)
		yn, err := liner.ReadString('\n')
		if err != nil {
			log.Fatalf("failed to read response: %v", err)
		}
		yn = strings.TrimRight(strings.ToLower(yn), "\n")
		if yn != "y" {
			log.Fatal("Conversion aborted.")
		}
	}
	fmt.Println("Conversion starting....")

	if opts.CPUFile != "" {
		f, err := os.Create(opts.CPUFile)
		if err != nil {
			log.Fatal(err)
		}
		if err = pprof.StartCPUProfile(f); err != nil {
			log.Fatal(err)
		}
		defer pprof.StopCPUProfile()
	}

	tr := newTracker(shards, opts)

	if err := tr.Run(); err != nil {
		log.Fatalf("Error occurred preventing completion: %v\n", err)
	}

	tr.PrintStats()
}

func collectShards(dbs []os.FileInfo) tsdb.ShardInfos {
	// Get the list of shards for conversion.
	var shards tsdb.ShardInfos
	for _, db := range dbs {
		d := tsdb.NewDatabase(filepath.Join(opts.DataPath, db.Name()))
		shs, err := d.Shards()
		if err != nil {
			log.Fatalf("Failed to access shards for database %v: %v\n", d.Name(), err)
		}
		shards = append(shards, shs...)
	}

	sort.Sort(shards)
	shards = shards.FilterFormat(tsdb.TSM1)
	if len(dbs) > 0 {
		shards = shards.ExclusiveDatabases(opts.DBs)
	}

	return shards
}

// backupDatabase backs up the database named db
func backupDatabase(db string) error {
	copyFile := func(path string, info os.FileInfo, err error) error {
		// Strip the DataPath from the path and replace with BackupPath.
		toPath := strings.Replace(path, opts.DataPath, opts.BackupPath, 1)

		if info.IsDir() {
			return os.MkdirAll(toPath, info.Mode())
		}

		in, err := os.Open(path)
		if err != nil {
			return err
		}
		defer in.Close()

		srcInfo, err := os.Stat(path)
		if err != nil {
			return err
		}

		out, err := os.OpenFile(toPath, os.O_CREATE|os.O_WRONLY, info.Mode())
		if err != nil {
			return err
		}
		defer out.Close()

		dstInfo, err := os.Stat(toPath)
		if err != nil {
			return err
		}

		if dstInfo.Size() == srcInfo.Size() {
			log.Printf("Backup file already found for %v with correct size, skipping.", path)
			return nil
		}

		if dstInfo.Size() > srcInfo.Size() {
			log.Printf("Invalid backup file found for %v, replacing with good copy.", path)
			if err := out.Truncate(0); err != nil {
				return err
			}
			if _, err := out.Seek(0, os.SEEK_SET); err != nil {
				return err
			}
		}

		if dstInfo.Size() > 0 {
			log.Printf("Resuming backup of file %v, starting at %v bytes", path, dstInfo.Size())
		}

		off, err := out.Seek(0, os.SEEK_END)
		if err != nil {
			return err
		}
		if _, err := in.Seek(off, os.SEEK_SET); err != nil {
			return err
		}

		log.Printf("Backing up file %v", path)

		_, err = io.Copy(out, in)

		return err
	}

	return filepath.Walk(filepath.Join(opts.DataPath, db), copyFile)
}

// convertShard converts the shard in-place.
func convertShard(si *tsdb.ShardInfo, tr *tracker) error {
	src := si.FullPath(opts.DataPath)
	dst := fmt.Sprintf("%v.%v", src, tsmExt)

	var reader ShardReader
	switch si.Format {
	case tsdb.BZ1:
		reader = bz1.NewReader(src, &tr.Stats, 0)
	case tsdb.B1:
		reader = b1.NewReader(src, &tr.Stats, 0)
	default:
		return fmt.Errorf("Unsupported shard format: %v", si.FormatAsString())
	}

	// Open the shard, and create a converter.
	if err := reader.Open(); err != nil {
		return fmt.Errorf("Failed to open %v for conversion: %v", src, err)
	}
	defer reader.Close()
	converter := NewConverter(dst, uint32(opts.TSMSize), &tr.Stats)

	// Perform the conversion.
	if err := converter.Process(reader); err != nil {
		return fmt.Errorf("Conversion of %v failed: %v", src, err)
	}

	// Delete source shard, and rename new tsm1 shard.
	if err := reader.Close(); err != nil {
		return fmt.Errorf("Conversion of %v failed due to close: %v", src, err)
	}

	if err := os.RemoveAll(si.FullPath(opts.DataPath)); err != nil {
		return fmt.Errorf("Deletion of %v failed: %v", src, err)
	}
	if err := os.Rename(dst, src); err != nil {
		return fmt.Errorf("Rename of %v to %v failed: %v", dst, src, err)
	}

	return nil
}

// ParallelGroup allows the maximum parrallelism of a set of operations to be controlled.
type ParallelGroup chan struct{}

// NewParallelGroup returns a group which allows n operations to run in parallel. A value of 0
// means no operations will ever run.
func NewParallelGroup(n int) ParallelGroup {
	return make(chan struct{}, n)
}

// Do executes one operation of the ParallelGroup
func (p ParallelGroup) Do(f func()) {
	p <- struct{}{} // acquire working slot
	defer func() { <-p }()

	f()
}

// yesno returns "yes" for true, "no" for false.
func yesno(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

// allDBs returns "all" if all databases are requested for conversion.
func allDBs(dbs []string) string {
	if dbs == nil {
		return "all"
	}
	return fmt.Sprintf("%v", dbs)
}

// isEnvSet checks to see if a variable was set in the environment
func isEnvSet(name string) bool {
	for _, s := range os.Environ() {
		if strings.SplitN(s, "=", 2)[0] == name {
			return true
		}
	}
	return false
}
