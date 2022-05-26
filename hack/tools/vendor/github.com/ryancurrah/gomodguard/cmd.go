package gomodguard

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-xmlfmt/xmlfmt"
	"github.com/mitchellh/go-homedir"
	"github.com/phayes/checkstyle"
	"gopkg.in/yaml.v2"
)

const (
	errFindingHomedir    = "unable to find home directory, %w"
	errReadingConfigFile = "could not read config file: %w"
	errParsingConfigFile = "could not parse config file: %w"
)

var (
	configFile           = ".gomodguard.yaml"
	logger               = log.New(os.Stderr, "", 0)
	errFindingConfigFile = fmt.Errorf("could not find config file")
)

// Run the gomodguard linter. Returns the exit code to use.
//nolint:funlen
func Run() int {
	var (
		args           []string
		help           bool
		noTest         bool
		report         string
		reportFile     string
		issuesExitCode int
		cwd, _         = os.Getwd()
	)

	flag.BoolVar(&help, "h", false, "Show this help text")
	flag.BoolVar(&help, "help", false, "")
	flag.BoolVar(&noTest, "n", false, "Don't lint test files")
	flag.BoolVar(&noTest, "no-test", false, "")
	flag.StringVar(&report, "r", "", "Report results to one of the following formats: checkstyle. "+
		"A report file destination must also be specified")
	flag.StringVar(&report, "report", "", "")
	flag.StringVar(&reportFile, "f", "", "Report results to the specified file. A report type must also be specified")
	flag.StringVar(&reportFile, "file", "", "")
	flag.IntVar(&issuesExitCode, "i", 2, "Exit code when issues were found")
	flag.IntVar(&issuesExitCode, "issues-exit-code", 2, "")
	flag.Parse()

	report = strings.TrimSpace(strings.ToLower(report))

	if help {
		showHelp()
		return 0
	}

	if report != "" && report != "checkstyle" {
		logger.Fatalf("error: invalid report type '%s'", report)
	}

	if report != "" && reportFile == "" {
		logger.Fatalf("error: a report file must be specified when a report is enabled")
	}

	if report == "" && reportFile != "" {
		logger.Fatalf("error: a report type must be specified when a report file is enabled")
	}

	args = flag.Args()
	if len(args) == 0 {
		args = []string{"./..."}
	}

	config, err := GetConfig(configFile)
	if err != nil {
		logger.Fatalf("error: %s", err)
	}

	filteredFiles := GetFilteredFiles(cwd, noTest, args)

	processor, err := NewProcessor(config)
	if err != nil {
		logger.Fatalf("error: %s", err)
	}

	logger.Printf("info: allowed modules, %+v", config.Allowed.Modules)
	logger.Printf("info: allowed module domains, %+v", config.Allowed.Domains)
	logger.Printf("info: blocked modules, %+v", config.Blocked.Modules.Get())
	logger.Printf("info: blocked modules with version constraints, %+v", config.Blocked.Versions.Get())

	results := processor.ProcessFiles(filteredFiles)

	if report == "checkstyle" {
		err := WriteCheckstyle(reportFile, results)
		if err != nil {
			logger.Fatalf("error: %s", err)
		}
	}

	for _, r := range results {
		fmt.Println(r.String())
	}

	if len(results) > 0 {
		return issuesExitCode
	}

	return 0
}

// GetConfig from YAML file.
func GetConfig(configFile string) (*Configuration, error) {
	config := Configuration{}

	home, err := homedir.Dir()
	if err != nil {
		return nil, fmt.Errorf(errFindingHomedir, err)
	}

	cfgFile := ""
	homeDirCfgFile := filepath.Join(home, configFile)

	switch {
	case fileExists(configFile):
		cfgFile = configFile
	case fileExists(homeDirCfgFile):
		cfgFile = homeDirCfgFile
	default:
		return nil, fmt.Errorf("%w: %s %s", errFindingConfigFile, configFile, homeDirCfgFile)
	}

	data, err := ioutil.ReadFile(cfgFile)
	if err != nil {
		return nil, fmt.Errorf(errReadingConfigFile, err)
	}

	err = yaml.Unmarshal(data, &config)
	if err != nil {
		return nil, fmt.Errorf(errParsingConfigFile, err)
	}

	return &config, nil
}

// GetFilteredFiles returns files based on search string arguments and filters.
func GetFilteredFiles(cwd string, skipTests bool, args []string) []string {
	var (
		foundFiles    = []string{}
		filteredFiles = []string{}
	)

	for _, f := range args {
		if strings.HasSuffix(f, "/...") {
			dir, _ := filepath.Split(f)

			foundFiles = append(foundFiles, expandGoWildcard(dir)...)

			continue
		}

		if _, err := os.Stat(f); err == nil {
			foundFiles = append(foundFiles, f)
		}
	}

	// Use relative path to print shorter names, sort out test foundFiles if chosen.
	for _, f := range foundFiles {
		if skipTests {
			if strings.HasSuffix(f, "_test.go") {
				continue
			}
		}

		if relativePath, err := filepath.Rel(cwd, f); err == nil {
			filteredFiles = append(filteredFiles, relativePath)

			continue
		}

		filteredFiles = append(filteredFiles, f)
	}

	return filteredFiles
}

// showHelp text for command line.
func showHelp() {
	helpText := `Usage: gomodguard <file> [files...]
Also supports package syntax but will use it in relative path, i.e. ./pkg/...
Flags:`
	fmt.Println(helpText)
	flag.PrintDefaults()
}

// WriteCheckstyle takes the results and writes them to a checkstyle formated file.
func WriteCheckstyle(checkstyleFilePath string, results []Issue) error {
	check := checkstyle.New()

	for i := range results {
		file := check.EnsureFile(results[i].FileName)
		file.AddError(checkstyle.NewError(results[i].LineNumber, 1, checkstyle.SeverityError, results[i].Reason,
			"gomodguard"))
	}

	checkstyleXML := fmt.Sprintf("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n%s", check.String())

	err := ioutil.WriteFile(checkstyleFilePath, []byte(xmlfmt.FormatXML(checkstyleXML, "", "  ")), 0644) // nolint:gosec
	if err != nil {
		return err
	}

	return nil
}

// fileExists returns true if the file path provided exists.
func fileExists(filename string) bool {
	info, err := os.Stat(filename)
	if os.IsNotExist(err) {
		return false
	}

	return !info.IsDir()
}

// expandGoWildcard path provided.
func expandGoWildcard(root string) []string {
	foundFiles := []string{}

	_ = filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		// Only append go foundFiles.
		if !strings.HasSuffix(info.Name(), ".go") {
			return nil
		}

		foundFiles = append(foundFiles, path)

		return nil
	})

	return foundFiles
}
