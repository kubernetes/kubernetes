// The misspell command corrects commonly misspelled English words in source files.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"text/template"
	"time"

	"github.com/client9/misspell"
)

var (
	defaultWrite *template.Template
	defaultRead  *template.Template

	stdout *log.Logger
	debug  *log.Logger

	version = "dev"
)

const (
	// Note for gometalinter it must be "File:Line:Column: Msg"
	//  note space beteen ": Msg"
	defaultWriteTmpl = `{{ .Filename }}:{{ .Line }}:{{ .Column }}: corrected "{{ .Original }}" to "{{ .Corrected }}"`
	defaultReadTmpl  = `{{ .Filename }}:{{ .Line }}:{{ .Column }}: "{{ .Original }}" is a misspelling of "{{ .Corrected }}"`
	csvTmpl          = `{{ printf "%q" .Filename }},{{ .Line }},{{ .Column }},{{ .Original }},{{ .Corrected }}`
	csvHeader        = `file,line,column,typo,corrected`
	sqliteTmpl       = `INSERT INTO misspell VALUES({{ printf "%q" .Filename }},{{ .Line }},{{ .Column }},{{ printf "%q" .Original }},{{ printf "%q" .Corrected }});`
	sqliteHeader     = `PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE misspell(
	"file" TEXT, "line" INTEGER, "column" INTEGER, "typo" TEXT, "corrected" TEXT
);`
	sqliteFooter = "COMMIT;"
)

func worker(writeit bool, r *misspell.Replacer, mode string, files <-chan string, results chan<- int) {
	count := 0
	for filename := range files {
		orig, err := misspell.ReadTextFile(filename)
		if err != nil {
			log.Println(err)
			continue
		}
		if len(orig) == 0 {
			continue
		}

		debug.Printf("Processing %s", filename)

		var updated string
		var changes []misspell.Diff

		if mode == "go" {
			updated, changes = r.ReplaceGo(orig)
		} else {
			updated, changes = r.Replace(orig)
		}

		if len(changes) == 0 {
			continue
		}
		count += len(changes)
		for _, diff := range changes {
			// add in filename
			diff.Filename = filename

			// output can be done by doing multiple goroutines
			// and can clobber os.Stdout.
			//
			// the log package can be used simultaneously from multiple goroutines
			var output bytes.Buffer
			if writeit {
				defaultWrite.Execute(&output, diff)
			} else {
				defaultRead.Execute(&output, diff)
			}

			// goroutine-safe print to os.Stdout
			stdout.Println(output.String())
		}

		if writeit {
			ioutil.WriteFile(filename, []byte(updated), 0)
		}
	}
	results <- count
}

func main() {
	t := time.Now()
	var (
		workers     = flag.Int("j", 0, "Number of workers, 0 = number of CPUs")
		writeit     = flag.Bool("w", false, "Overwrite file with corrections (default is just to display)")
		quietFlag   = flag.Bool("q", false, "Do not emit misspelling output")
		outFlag     = flag.String("o", "stdout", "output file or [stderr|stdout|]")
		format      = flag.String("f", "", "'csv', 'sqlite3' or custom Golang template for output")
		ignores     = flag.String("i", "", "ignore the following corrections, comma separated")
		locale      = flag.String("locale", "", "Correct spellings using locale perferances for US or UK.  Default is to use a neutral variety of English.  Setting locale to US will correct the British spelling of 'colour' to 'color'")
		mode        = flag.String("source", "auto", "Source mode: auto=guess, go=golang source, text=plain or markdown-like text")
		debugFlag   = flag.Bool("debug", false, "Debug matching, very slow")
		exitError   = flag.Bool("error", false, "Exit with 2 if misspelling found")
		showVersion = flag.Bool("v", false, "Show version and exit")

		showLegal = flag.Bool("legal", false, "Show legal information and exit")
	)
	flag.Parse()

	if *showVersion {
		fmt.Println(version)
		return
	}
	if *showLegal {
		fmt.Println(misspell.Legal)
		return
	}
	if *debugFlag {
		debug = log.New(os.Stderr, "DEBUG ", 0)
	} else {
		debug = log.New(ioutil.Discard, "", 0)
	}

	r := misspell.Replacer{
		Replacements: misspell.DictMain,
		Debug:        *debugFlag,
	}
	//
	// Figure out regional variations
	//
	switch strings.ToUpper(*locale) {
	case "":
		// nothing
	case "US":
		r.AddRuleList(misspell.DictAmerican)
	case "UK", "GB":
		r.AddRuleList(misspell.DictBritish)
	case "NZ", "AU", "CA":
		log.Fatalf("Help wanted.  https://github.com/client9/misspell/issues/6")
	default:
		log.Fatalf("Unknown locale: %q", *locale)
	}

	//
	// Stuff to ignore
	//
	if len(*ignores) > 0 {
		r.RemoveRule(strings.Split(*ignores, ","))
	}

	//
	// Source input mode
	//
	switch *mode {
	case "auto":
	case "go":
	case "text":
	default:
		log.Fatalf("Mode must be one of auto=guess, go=golang source, text=plain or markdown-like text")
	}

	//
	// Custom output
	//
	switch {
	case *format == "csv":
		tmpl := template.Must(template.New("csv").Parse(csvTmpl))
		defaultWrite = tmpl
		defaultRead = tmpl
		stdout.Println(csvHeader)
	case *format == "sqlite" || *format == "sqlite3":
		tmpl := template.Must(template.New("sqlite3").Parse(sqliteTmpl))
		defaultWrite = tmpl
		defaultRead = tmpl
		stdout.Println(sqliteHeader)
	case len(*format) > 0:
		t, err := template.New("custom").Parse(*format)
		if err != nil {
			log.Fatalf("Unable to compile log format: %s", err)
		}
		defaultWrite = t
		defaultRead = t
	default: // format == ""
		defaultWrite = template.Must(template.New("defaultWrite").Parse(defaultWriteTmpl))
		defaultRead = template.Must(template.New("defaultRead").Parse(defaultReadTmpl))
	}

	// we cant't just write to os.Stdout directly since we have multiple goroutine
	// all writing at the same time causing broken output.  Log is routine safe.
	// we see it so it doesn't use a prefix or include a time stamp.
	switch {
	case *quietFlag || *outFlag == "/dev/null":
		stdout = log.New(ioutil.Discard, "", 0)
	case *outFlag == "/dev/stderr" || *outFlag == "stderr":
		stdout = log.New(os.Stderr, "", 0)
	case *outFlag == "/dev/stdout" || *outFlag == "stdout":
		stdout = log.New(os.Stdout, "", 0)
	case *outFlag == "" || *outFlag == "-":
		stdout = log.New(os.Stdout, "", 0)
	default:
		fo, err := os.Create(*outFlag)
		if err != nil {
			log.Fatalf("unable to create outfile %q: %s", *outFlag, err)
		}
		defer fo.Close()
		stdout = log.New(fo, "", 0)
	}

	//
	// Number of Workers / CPU to use
	//
	if *workers < 0 {
		log.Fatalf("-j must >= 0")
	}
	if *workers == 0 {
		*workers = runtime.NumCPU()
	}
	if *debugFlag {
		*workers = 1
	}

	//
	// Done with Flags.
	//  Compile the Replacer and process files
	//
	r.Compile()

	args := flag.Args()
	debug.Printf("initialization complete in %v", time.Since(t))

	// stdin/stdout
	if len(args) == 0 {
		// if we are working with pipes/stdin/stdout
		// there is no concurrency, so we can directly
		// send data to the writers
		var fileout io.Writer
		var errout io.Writer
		switch *writeit {
		case true:
			// if we ARE writing the corrected stream
			// the corrected stream goes to stdout
			// and the misspelling errors goes to stderr
			// so we can do something like this:
			// curl something | misspell -w | gzip > afile.gz
			fileout = os.Stdout
			errout = os.Stderr
		case false:
			// if we are not writing out the corrected stream
			// then work just like files.  Misspelling errors
			// are sent to stdout
			fileout = ioutil.Discard
			errout = os.Stdout
		}
		count := 0
		next := func(diff misspell.Diff) {
			count++

			// don't even evaluate the output templates
			if *quietFlag {
				return
			}
			diff.Filename = "stdin"
			if *writeit {
				defaultWrite.Execute(errout, diff)
			} else {
				defaultRead.Execute(errout, diff)
			}
			errout.Write([]byte{'\n'})

		}
		err := r.ReplaceReader(os.Stdin, fileout, next)
		if err != nil {
			os.Exit(1)
		}
		switch *format {
		case "sqlite", "sqlite3":
			fileout.Write([]byte(sqliteFooter))
		}
		if count != 0 && *exitError {
			// error
			os.Exit(2)
		}
		return
	}

	c := make(chan string, 64)
	results := make(chan int, *workers)

	for i := 0; i < *workers; i++ {
		go worker(*writeit, &r, *mode, c, results)
	}

	for _, filename := range args {
		filepath.Walk(filename, func(path string, info os.FileInfo, err error) error {
			if err == nil && !info.IsDir() {
				c <- path
			}
			return nil
		})
	}
	close(c)

	count := 0
	for i := 0; i < *workers; i++ {
		changed := <-results
		count += changed
	}

	switch *format {
	case "sqlite", "sqlite3":
		stdout.Println(sqliteFooter)
	}

	if count != 0 && *exitError {
		os.Exit(2)
	}
}
