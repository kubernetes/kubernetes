package cli // import "github.com/influxdata/influxdb/cmd/influx/cli"

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"text/tabwriter"

	"golang.org/x/crypto/ssh/terminal"

	"github.com/influxdata/influxdb/client"
	"github.com/influxdata/influxdb/importer/v8"
	"github.com/influxdata/influxdb/models"
	"github.com/peterh/liner"
)

const (
	noTokenMsg = "Visit https://enterprise.influxdata.com to register for updates, InfluxDB server management, and monitoring.\n"
)

// ErrBlankCommand is returned when a parsed command is empty.
var ErrBlankCommand = errors.New("empty input")

// CommandLine holds CLI configuration and state
type CommandLine struct {
	Client           *client.Client
	Line             *liner.State
	Host             string
	Port             int
	Username         string
	Password         string
	Database         string
	Ssl              bool
	UnsafeSsl        bool
	RetentionPolicy  string
	ClientVersion    string
	ServerVersion    string
	Pretty           bool   // controls pretty print for json
	Format           string // controls the output format.  Valid values are json, csv, or column
	Precision        string
	WriteConsistency string
	Execute          string
	ShowVersion      bool
	Import           bool
	PPS              int // Controls how many points per second the import will allow via throttling
	Path             string
	Compressed       bool
	Chunked          bool
	Quit             chan struct{}
	IgnoreSignals    bool // Ignore signals normally caught by this process (used primarily for testing)
	ForceTTY         bool // Force the CLI to act as if it were connected to a TTY
	osSignals        chan os.Signal
	historyFilePath  string
}

// New returns an instance of CommandLine
func New(version string) *CommandLine {
	return &CommandLine{
		ClientVersion: version,
		Quit:          make(chan struct{}, 1),
		osSignals:     make(chan os.Signal, 1),
	}
}

// Run executes the CLI
func (c *CommandLine) Run() error {
	hasTTY := c.ForceTTY || terminal.IsTerminal(int(os.Stdin.Fd()))

	var promptForPassword bool
	// determine if they set the password flag but provided no value
	for _, v := range os.Args {
		v = strings.ToLower(v)
		if (strings.HasPrefix(v, "-password") || strings.HasPrefix(v, "--password")) && c.Password == "" {
			promptForPassword = true
			break
		}
	}

	// Check if we will be able to prompt for the password later.
	if promptForPassword && !hasTTY {
		return errors.New("Unable to prompt for a password with no TTY.")
	}

	// Read environment variables for username/password.
	if c.Username == "" {
		c.Username = os.Getenv("INFLUX_USERNAME")
	}
	// If we are going to be prompted for a password, always use the entered password.
	if promptForPassword {
		// Open the liner (temporarily) and prompt for the password.
		p, e := func() (string, error) {
			l := liner.NewLiner()
			defer l.Close()
			return l.PasswordPrompt("password: ")
		}()
		if e != nil {
			return errors.New("Unable to parse password")
		}
		c.Password = p
	} else if c.Password == "" {
		c.Password = os.Getenv("INFLUX_PASSWORD")
	}

	if err := c.Connect(""); err != nil {
		return fmt.Errorf(
			"Failed to connect to %s\nPlease check your connection settings and ensure 'influxd' is running.",
			c.Client.Addr())
	}

	// Modify precision.
	c.SetPrecision(c.Precision)

	if c.Execute != "" {
		// Make the non-interactive mode send everything through the CLI's parser
		// the same way the interactive mode works
		lines := strings.Split(c.Execute, "\n")
		for _, line := range lines {
			if err := c.ParseCommand(line); err != nil {
				return err
			}
		}
		return nil
	}

	if c.Import {
		path := net.JoinHostPort(c.Host, strconv.Itoa(c.Port))
		u, e := client.ParseConnectionString(path, c.Ssl)
		if e != nil {
			return e
		}

		config := v8.NewConfig()
		config.Username = c.Username
		config.Password = c.Password
		config.Precision = "ns"
		config.WriteConsistency = "any"
		config.Path = c.Path
		config.Version = c.ClientVersion
		config.URL = u
		config.Compressed = c.Compressed
		config.PPS = c.PPS
		config.Precision = c.Precision

		i := v8.NewImporter(config)
		if err := i.Import(); err != nil {
			err = fmt.Errorf("ERROR: %s\n", err)
			return err
		}
		return nil
	}

	if !hasTTY {
		cmd, err := ioutil.ReadAll(os.Stdin)
		if err != nil {
			return err
		}
		return c.ExecuteQuery(string(cmd))
	}

	if !c.IgnoreSignals {
		// register OS signals for graceful termination
		signal.Notify(c.osSignals, syscall.SIGINT, syscall.SIGTERM)
	}

	c.Line = liner.NewLiner()
	defer c.Line.Close()

	c.Line.SetMultiLineMode(true)

	token, err := c.DatabaseToken()
	if err != nil {
		return fmt.Errorf("Failed to check token: %s", err.Error())
	}
	if token == "" {
		fmt.Printf(noTokenMsg)
	}
	fmt.Printf("Connected to %s version %s\n", c.Client.Addr(), c.ServerVersion)

	c.Version()

	// Only load/write history if HOME environment variable is set.
	if homeDir := os.Getenv("HOME"); homeDir != "" {
		// Attempt to load the history file.
		c.historyFilePath = filepath.Join(homeDir, ".influx_history")
		if historyFile, err := os.Open(c.historyFilePath); err == nil {
			c.Line.ReadHistory(historyFile)
			historyFile.Close()
		}
	}

	// read from prompt until exit is run
	return c.mainLoop()
}

// mainLoop runs the main prompt loop for the CLI.
func (c *CommandLine) mainLoop() error {
	for {
		select {
		case <-c.osSignals:
			c.exit()
			return nil
		case <-c.Quit:
			c.exit()
			return nil
		default:
			l, e := c.Line.Prompt("> ")
			if e == io.EOF {
				// Instead of die, register that someone exited the program gracefully
				l = "exit"
			} else if e != nil {
				c.exit()
				return e
			}
			if err := c.ParseCommand(l); err != ErrBlankCommand && !strings.HasPrefix(strings.TrimSpace(l), "auth") {
				c.Line.AppendHistory(l)
				c.saveHistory()
			}
		}
	}
}

// ParseCommand parses an instruction and calls related method, if any
func (c *CommandLine) ParseCommand(cmd string) error {
	lcmd := strings.TrimSpace(strings.ToLower(cmd))
	tokens := strings.Fields(lcmd)

	if len(tokens) > 0 {
		switch tokens[0] {
		case "exit", "quit":
			close(c.Quit)
		case "gopher":
			c.gopher()
		case "connect":
			return c.Connect(cmd)
		case "auth":
			c.SetAuth(cmd)
		case "help":
			c.help()
		case "history":
			c.history()
		case "format":
			c.SetFormat(cmd)
		case "precision":
			c.SetPrecision(cmd)
		case "consistency":
			c.SetWriteConsistency(cmd)
		case "settings":
			c.Settings()
		case "pretty":
			c.Pretty = !c.Pretty
			if c.Pretty {
				fmt.Println("Pretty print enabled")
			} else {
				fmt.Println("Pretty print disabled")
			}
		case "use":
			c.use(cmd)
		case "insert":
			return c.Insert(cmd)
		default:
			return c.ExecuteQuery(cmd)
		}

		return nil
	}
	return ErrBlankCommand
}

// Connect connects client to a server
func (c *CommandLine) Connect(cmd string) error {
	var cl *client.Client
	var u url.URL

	// Remove the "connect" keyword if it exists
	path := strings.TrimSpace(strings.Replace(cmd, "connect", "", -1))

	// If they didn't provide a connection string, use the current settings
	if path == "" {
		path = net.JoinHostPort(c.Host, strconv.Itoa(c.Port))
	}

	var e error
	u, e = client.ParseConnectionString(path, c.Ssl)
	if e != nil {
		return e
	}

	config := client.NewConfig()
	config.URL = u
	config.Username = c.Username
	config.Password = c.Password
	config.UserAgent = "InfluxDBShell/" + c.ClientVersion
	config.Precision = c.Precision
	config.UnsafeSsl = c.UnsafeSsl
	cl, err := client.NewClient(config)
	if err != nil {
		return fmt.Errorf("Could not create client %s", err)
	}
	c.Client = cl

	var v string
	if _, v, e = c.Client.Ping(); e != nil {
		return fmt.Errorf("Failed to connect to %s\n", c.Client.Addr())
	}
	c.ServerVersion = v
	// Update the command with the current connection information
	if h, p, err := net.SplitHostPort(config.URL.Host); err == nil {
		c.Host = h
		if i, err := strconv.Atoi(p); err == nil {
			c.Port = i
		}
	}

	return nil
}

// SetAuth sets client authentication credentials
func (c *CommandLine) SetAuth(cmd string) {
	// If they pass in the entire command, we should parse it
	// auth <username> <password>
	args := strings.Fields(cmd)
	if len(args) == 3 {
		args = args[1:]
	} else {
		args = []string{}
	}

	if len(args) == 2 {
		c.Username = args[0]
		c.Password = args[1]
	} else {
		u, e := c.Line.Prompt("username: ")
		if e != nil {
			fmt.Printf("Unable to process input: %s", e)
			return
		}
		c.Username = strings.TrimSpace(u)
		p, e := c.Line.PasswordPrompt("password: ")
		if e != nil {
			fmt.Printf("Unable to process input: %s", e)
			return
		}
		c.Password = p
	}

	// Update the client as well
	c.Client.SetAuth(c.Username, c.Password)
}

func (c *CommandLine) use(cmd string) {
	args := strings.Split(strings.TrimSuffix(strings.TrimSpace(cmd), ";"), " ")
	if len(args) != 2 {
		fmt.Printf("Could not parse database name from %q.\n", cmd)
		return
	}
	d := args[1]

	// Validate if specified database exists
	response, err := c.Client.Query(client.Query{Command: "SHOW DATABASES"})
	if err != nil {
		fmt.Printf("ERR: %s\n", err)
		return
	} else if err := response.Error(); err != nil {
		if c.Username == "" {
			fmt.Printf("ERR: %s\n", err)
			return
		}
		// TODO(jsternberg): Fix SHOW DATABASES to be user-aware #6397.
		// If we are unable to run SHOW DATABASES, display a warning and use the
		// database anyway in case the person doesn't have permission to run the
		// command, but does have permission to use the database.
		fmt.Printf("WARN: %s\n", err)
	} else {
		// Verify the provided database exists
		if databaseExists := func() bool {
			for _, result := range response.Results {
				for _, row := range result.Series {
					if row.Name == "databases" {
						for _, values := range row.Values {
							for _, database := range values {
								if database == d {
									return true
								}
							}
						}
					}
				}
			}
			return false
		}(); !databaseExists {
			fmt.Printf("ERR: Database %s doesn't exist. Run SHOW DATABASES for a list of existing databases.\n", d)
			return
		}
	}

	c.Database = d
	fmt.Printf("Using database %s\n", d)
}

// SetPrecision sets client precision
func (c *CommandLine) SetPrecision(cmd string) {
	// Remove the "precision" keyword if it exists
	cmd = strings.TrimSpace(strings.Replace(cmd, "precision", "", -1))
	// normalize cmd
	cmd = strings.ToLower(cmd)

	switch cmd {
	case "h", "m", "s", "ms", "u", "ns":
		c.Precision = cmd
		c.Client.SetPrecision(c.Precision)
	case "rfc3339":
		c.Precision = ""
		c.Client.SetPrecision(c.Precision)
	default:
		fmt.Printf("Unknown precision %q. Please use rfc3339, h, m, s, ms, u or ns.\n", cmd)
	}
}

// SetFormat sets output format
func (c *CommandLine) SetFormat(cmd string) {
	// Remove the "format" keyword if it exists
	cmd = strings.TrimSpace(strings.Replace(cmd, "format", "", -1))
	// normalize cmd
	cmd = strings.ToLower(cmd)

	switch cmd {
	case "json", "csv", "column":
		c.Format = cmd
	default:
		fmt.Printf("Unknown format %q. Please use json, csv, or column.\n", cmd)
	}
}

// SetWriteConsistency sets cluster consistency level
func (c *CommandLine) SetWriteConsistency(cmd string) {
	// Remove the "consistency" keyword if it exists
	cmd = strings.TrimSpace(strings.Replace(cmd, "consistency", "", -1))
	// normalize cmd
	cmd = strings.ToLower(cmd)

	_, err := models.ParseConsistencyLevel(cmd)
	if err != nil {
		fmt.Printf("Unknown consistency level %q. Please use any, one, quorum, or all.\n", cmd)
		return
	}
	c.WriteConsistency = cmd
}

// isWhitespace returns true if the rune is a space, tab, or newline.
func isWhitespace(ch rune) bool { return ch == ' ' || ch == '\t' || ch == '\n' }

// isLetter returns true if the rune is a letter.
func isLetter(ch rune) bool { return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') }

// isDigit returns true if the rune is a digit.
func isDigit(ch rune) bool { return (ch >= '0' && ch <= '9') }

// isIdentFirstChar returns true if the rune can be used as the first char in an unquoted identifer.
func isIdentFirstChar(ch rune) bool { return isLetter(ch) || ch == '_' }

// isIdentChar returns true if the rune can be used in an unquoted identifier.
func isNotIdentChar(ch rune) bool { return !(isLetter(ch) || isDigit(ch) || ch == '_') }

func parseUnquotedIdentifier(stmt string) (string, string) {
	if fields := strings.FieldsFunc(stmt, isNotIdentChar); len(fields) > 0 {
		return fields[0], strings.TrimPrefix(stmt, fields[0])
	}
	return "", stmt
}

func parseDoubleQuotedIdentifier(stmt string) (string, string) {
	escapeNext := false
	fields := strings.FieldsFunc(stmt, func(ch rune) bool {
		if ch == '\\' {
			escapeNext = true
		} else if ch == '"' {
			if !escapeNext {
				return true
			}
			escapeNext = false
		}
		return false
	})
	if len(fields) > 0 {
		return fields[0], strings.TrimPrefix(stmt, "\""+fields[0]+"\"")
	}
	return "", stmt
}

func parseNextIdentifier(stmt string) (ident, remainder string) {
	if len(stmt) > 0 {
		switch {
		case isWhitespace(rune(stmt[0])):
			return parseNextIdentifier(stmt[1:])
		case isIdentFirstChar(rune(stmt[0])):
			return parseUnquotedIdentifier(stmt)
		case stmt[0] == '"':
			return parseDoubleQuotedIdentifier(stmt)
		}
	}
	return "", stmt
}

func (c *CommandLine) parseInto(stmt string) string {
	ident, stmt := parseNextIdentifier(stmt)
	if strings.HasPrefix(stmt, ".") {
		c.Database = ident
		fmt.Printf("Using database %s\n", c.Database)
		ident, stmt = parseNextIdentifier(stmt[1:])
	}
	if strings.HasPrefix(stmt, " ") {
		c.RetentionPolicy = ident
		fmt.Printf("Using retention policy %s\n", c.RetentionPolicy)
		return stmt[1:]
	}
	return stmt
}

// Insert runs an INSERT statement
func (c *CommandLine) Insert(stmt string) error {
	i, point := parseNextIdentifier(stmt)
	if !strings.EqualFold(i, "insert") {
		fmt.Printf("ERR: found %s, expected INSERT\n", i)
		return nil
	}
	if i, r := parseNextIdentifier(point); strings.EqualFold(i, "into") {
		point = c.parseInto(r)
	}
	_, err := c.Client.Write(client.BatchPoints{
		Points: []client.Point{
			client.Point{Raw: point},
		},
		Database:         c.Database,
		RetentionPolicy:  c.RetentionPolicy,
		Precision:        c.Precision,
		WriteConsistency: c.WriteConsistency,
	})
	if err != nil {
		fmt.Printf("ERR: %s\n", err)
		if c.Database == "" {
			fmt.Println("Note: error may be due to not setting a database or retention policy.")
			fmt.Println(`Please set a database with the command "use <database>" or`)
			fmt.Println("INSERT INTO <database>.<retention-policy> <point>")
		}
		return err
	}
	return nil
}

// query creates a query struct to be used with the client.
func (c *CommandLine) query(query string, database string) client.Query {
	return client.Query{
		Command:  query,
		Database: database,
		Chunked:  true,
	}
}

// ExecuteQuery runs any query statement
func (c *CommandLine) ExecuteQuery(query string) error {
	response, err := c.Client.Query(c.query(query, c.Database))
	if err != nil {
		fmt.Printf("ERR: %s\n", err)
		return err
	}
	c.FormatResponse(response, os.Stdout)
	if err := response.Error(); err != nil {
		fmt.Printf("ERR: %s\n", response.Error())
		if c.Database == "" {
			fmt.Println("Warning: It is possible this error is due to not setting a database.")
			fmt.Println(`Please set a database with the command "use <database>".`)
		}
		return err
	}
	return nil
}

// DatabaseToken retrieves database token
func (c *CommandLine) DatabaseToken() (string, error) {
	response, err := c.Client.Query(c.query("SHOW DIAGNOSTICS for 'registration'", ""))
	if err != nil {
		return "", err
	}

	if response.Error() != nil || len(response.Results) == 0 || len(response.Results[0].Series) == 0 {
		return "", nil
	}

	// Look for position of "token" column.
	for i, s := range (*response).Results[0].Series[0].Columns {
		if s == "token" {
			return (*response).Results[0].Series[0].Values[0][i].(string), nil
		}
	}
	return "", nil
}

// FormatResponse formats output to previsouly chosen format
func (c *CommandLine) FormatResponse(response *client.Response, w io.Writer) {
	switch c.Format {
	case "json":
		c.writeJSON(response, w)
	case "csv":
		c.writeCSV(response, w)
	case "column":
		c.writeColumns(response, w)
	default:
		fmt.Fprintf(w, "Unknown output format %q.\n", c.Format)
	}
}

func (c *CommandLine) writeJSON(response *client.Response, w io.Writer) {
	var data []byte
	var err error
	if c.Pretty {
		data, err = json.MarshalIndent(response, "", "    ")
	} else {
		data, err = json.Marshal(response)
	}
	if err != nil {
		fmt.Fprintf(w, "Unable to parse json: %s\n", err)
		return
	}
	fmt.Fprintln(w, string(data))
}

func (c *CommandLine) writeCSV(response *client.Response, w io.Writer) {
	csvw := csv.NewWriter(w)
	for _, result := range response.Results {
		// Create a tabbed writer for each result as they won't always line up
		rows := c.formatResults(result, "\t")
		for _, r := range rows {
			csvw.Write(strings.Split(r, "\t"))
		}
		csvw.Flush()
	}
}

func (c *CommandLine) writeColumns(response *client.Response, w io.Writer) {
	// Create a tabbed writer for each result as they won't always line up
	writer := new(tabwriter.Writer)
	writer.Init(w, 0, 8, 1, '\t', 0)

	for _, result := range response.Results {
		// Print out all messages first
		for _, m := range result.Messages {
			fmt.Fprintf(w, "%s: %s.\n", m.Level, m.Text)
		}
		csv := c.formatResults(result, "\t")
		for _, r := range csv {
			fmt.Fprintln(writer, r)
		}
		writer.Flush()
	}
}

// formatResults will behave differently if you are formatting for columns or csv
func (c *CommandLine) formatResults(result client.Result, separator string) []string {
	rows := []string{}
	// Create a tabbed writer for each result as they won't always line up
	for i, row := range result.Series {
		// gather tags
		tags := []string{}
		for k, v := range row.Tags {
			tags = append(tags, fmt.Sprintf("%s=%s", k, v))
			sort.Strings(tags)
		}

		columnNames := []string{}

		// Only put name/tags in a column if format is csv
		if c.Format == "csv" {
			if len(tags) > 0 {
				columnNames = append([]string{"tags"}, columnNames...)
			}

			if row.Name != "" {
				columnNames = append([]string{"name"}, columnNames...)
			}
		}

		for _, column := range row.Columns {
			columnNames = append(columnNames, column)
		}

		// Output a line separator if we have more than one set or results and format is column
		if i > 0 && c.Format == "column" {
			rows = append(rows, "")
		}

		// If we are column format, we break out the name/tag to separate lines
		if c.Format == "column" {
			if row.Name != "" {
				n := fmt.Sprintf("name: %s", row.Name)
				rows = append(rows, n)
			}
			if len(tags) > 0 {
				t := fmt.Sprintf("tags: %s", (strings.Join(tags, ", ")))
				rows = append(rows, t)
			}
		}

		rows = append(rows, strings.Join(columnNames, separator))

		// if format is column, write dashes under each column
		if c.Format == "column" {
			lines := []string{}
			for _, columnName := range columnNames {
				lines = append(lines, strings.Repeat("-", len(columnName)))
			}
			rows = append(rows, strings.Join(lines, separator))
		}

		for _, v := range row.Values {
			var values []string
			if c.Format == "csv" {
				if row.Name != "" {
					values = append(values, row.Name)
				}
				if len(tags) > 0 {
					values = append(values, strings.Join(tags, ","))
				}
			}

			for _, vv := range v {
				values = append(values, interfaceToString(vv))
			}
			rows = append(rows, strings.Join(values, separator))
		}
		// Output a line separator if in column format
		if c.Format == "column" {
			rows = append(rows, "")
		}
	}
	return rows
}

func interfaceToString(v interface{}) string {
	switch t := v.(type) {
	case nil:
		return ""
	case bool:
		return fmt.Sprintf("%v", v)
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, uintptr:
		return fmt.Sprintf("%d", t)
	case float32, float64:
		return fmt.Sprintf("%v", t)
	default:
		return fmt.Sprintf("%v", t)
	}
}

// Settings prints current settings
func (c *CommandLine) Settings() {
	w := new(tabwriter.Writer)
	w.Init(os.Stdout, 0, 1, 1, '\t', 0)
	if c.Port > 0 {
		fmt.Fprintf(w, "Host\t%s:%d\n", c.Host, c.Port)
	} else {
		fmt.Fprintf(w, "Host\t%s\n", c.Host)
	}
	fmt.Fprintf(w, "Username\t%s\n", c.Username)
	fmt.Fprintf(w, "Database\t%s\n", c.Database)
	fmt.Fprintf(w, "Pretty\t%v\n", c.Pretty)
	fmt.Fprintf(w, "Format\t%s\n", c.Format)
	fmt.Fprintf(w, "Write Consistency\t%s\n", c.WriteConsistency)
	fmt.Fprintln(w)
	w.Flush()
}

func (c *CommandLine) help() {
	fmt.Println(`Usage:
        connect <host:port>   connects to another node specified by host:port
        auth                  prompts for username and password
        pretty                toggles pretty print for the json format
        use <db_name>         sets current database
        format <format>       specifies the format of the server responses: json, csv, or column
        precision <format>    specifies the format of the timestamp: rfc3339, h, m, s, ms, u or ns
        consistency <level>   sets write consistency level: any, one, quorum, or all
        history               displays command history
        settings              outputs the current settings for the shell
        exit/quit/ctrl+d      quits the influx shell

        show databases        show database names
        show series           show series information
        show measurements     show measurement information
        show tag keys         show tag key information
        show field keys       show field key information

        A full list of influxql commands can be found at:
        https://docs.influxdata.com/influxdb/latest/query_language/spec/
`)
}

func (c *CommandLine) history() {
	var buf bytes.Buffer
	c.Line.WriteHistory(&buf)
	fmt.Print(buf.String())
}

func (c *CommandLine) saveHistory() {
	if historyFile, err := os.Create(c.historyFilePath); err != nil {
		fmt.Printf("There was an error writing history file: %s\n", err)
	} else {
		c.Line.WriteHistory(historyFile)
		historyFile.Close()
	}
}

func (c *CommandLine) gopher() {
	fmt.Println(`
                                          .-::-::://:-::-    .:/++/'
                                     '://:-''/oo+//++o+/.://o-    ./+:
                                  .:-.    '++-         .o/ '+yydhy'  o-
                               .:/.      .h:         :osoys  .smMN-  :/
                            -/:.'        s-         /MMMymh.   '/y/  s'
                         -+s:''''        d          -mMMms//     '-/o:
                       -/++/++/////:.    o:          '... s-        :s.
                     :+-+s-'       ':/'  's-             /+          'o:
                   '+-'o:        /ydhsh.  '//.        '-o-             o-
                  .y. o:        .MMMdm+y    ':+++:::/+:.'               s:
                .-h/  y-        'sdmds'h -+ydds:::-.'                   'h.
             .//-.d'  o:          '.' 'dsNMMMNh:.:++'                    :y
            +y.  'd   's.            .s:mddds:     ++                     o/
           'N-  odd    'o/.       './o-s-'   .---+++'                      o-
           'N'  yNd      .://:/:::::. -s   -+/s/./s'                       'o/'
            so'  .h         ''''       ////s: '+. .s                         +y'
             os/-.y'                       's' 'y::+                          +d'
               '.:o/                        -+:-:.'                            so.---.'
                   o'                                                          'd-.''/s'
                   .s'                                                          :y.''.y
                    -s                                                           mo:::'
                     ::                                                          yh
                      //                                      ''''               /M'
                       o+                                    .s///:/.            'N:
                        :+                                   /:    -s'            ho
                         's-                               -/s/:+/.+h'            +h
                           ys'                            ':'    '-.              -d
                            oh                                                    .h
                             /o                                                   .s
                              s.                                                  .h
                              -y                                                  .d
                               m/                                                 -h
                               +d                                                 /o
                               'N-                                                y:
                                h:                                                m.
                                s-                                               -d
                                o-                                               s+
                                +-                                              'm'
                                s/                                              oo--.
                                y-                                             /s  ':+'
                                s'                                           'od--' .d:
                                -+                                         ':o: ':+-/+
                                 y-                                      .:+-      '
                                //o-                                 '.:+/.
                                .-:+/'                           ''-/+/.
                                    ./:'                    ''.:o+/-'
                                      .+o:/:/+-'      ''.-+ooo/-'
                                         o:   -h///++////-.
                                        /:   .o/
                                       //+  'y
                                       ./sooy.

`)
}

// Version prints CLI version
func (c *CommandLine) Version() {
	fmt.Println("InfluxDB shell version:", c.ClientVersion)
}

func (c *CommandLine) exit() {
	// write to history file
	c.saveHistory()
	// release line resources
	c.Line.Close()
	c.Line = nil
}
