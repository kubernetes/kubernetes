/*
Copyright 2022 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kubelet

import (
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"reflect"
	"regexp"
	"regexp/syntax"
	"runtime"
	"strconv"
	"strings"
	"time"

	securejoin "github.com/cyphar/filepath-securejoin"
	"k8s.io/apimachinery/pkg/util/sets"
	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

const (
	dateLayout       = "2006-1-2 15:4:5"
	maxTailLines     = 100000
	maxServiceLength = 256
	maxServices      = 4
	nodeLogDir       = "/var/log/"
)

var (
	journal = journalServer{}
	// The set of known safe characters to pass to journalctl / GetWinEvent flags - only add to this list if the
	// character cannot be used to create invalid sequences. This is intended as a broad defense against malformed
	// input that could cause an escape.
	reServiceNameUnsafeCharacters = regexp.MustCompile(`[^a-zA-Z\-_.:0-9@]+`)
	reRelativeDate                = regexp.MustCompile(`^(\+|\-)?[\d]+(s|m|h|d)$`)
)

// journalServer returns text output from the OS specific service logger to view
// from the client. It runs with the privileges of the calling  process
// (the kubelet) and should only be allowed to be invoked by a root user.
type journalServer struct{}

// ServeHTTP translates HTTP query parameters into arguments to be passed
// to journalctl on the current system. It supports content-encoding of
// gzip to reduce total content size.
func (journalServer) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	var out io.Writer = w

	nlq, errs := newNodeLogQuery(req.URL.Query())
	if len(errs) > 0 {
		http.Error(w, errs.ToAggregate().Error(), http.StatusBadRequest)
		return
	}

	// TODO: Also set a response header that indicates how the request's query was resolved,
	// e.g. "kube-log-source: journal://foobar?arg1=value" or "kube-log-source: file:///var/log/foobar.log"
	w.Header().Set("Content-Type", "text/plain;charset=UTF-8")
	if req.Header.Get("Accept-Encoding") == "gzip" {
		w.Header().Set("Content-Encoding", "gzip")

		gz, err := gzip.NewWriterLevel(out, gzip.BestSpeed)
		if err != nil {
			fmt.Fprintf(w, "\nfailed to get gzip writer: %v\n", err)
			return
		}
		defer gz.Close()
		out = gz
	}
	nlq.Copy(out)
}

// nodeLogQuery encapsulates the log query request
type nodeLogQuery struct {
	// Services are the list of services to be queried
	Services []string
	// Files are the list of files
	Files []string
	options
}

// options encapsulates the query options for services
type options struct {
	// SinceTime is an RFC3339 timestamp from which to show logs.
	SinceTime *time.Time
	// UntilTime is an RFC3339 timestamp until which to show logs.
	UntilTime *time.Time
	// TailLines is used to retrieve the specified number of lines (not more than 100k) from the end of the log.
	// Support for this is implementation specific and only available for service logs.
	TailLines *int
	// Boot show messages from a specific boot. Allowed values are less than 1. Passing an invalid boot offset will fail
	// retrieving logs and return an error. Support for this is implementation specific
	Boot *int
	// Pattern filters log entries by the provided regex pattern. On Linux nodes, this pattern will be read as a
	// PCRE2 regex, on Windows nodes it will be read as a PowerShell regex. Support for this is implementation specific.
	Pattern string
	ocAdm
}

// ocAdm encapsulates the oc adm node-logs specific options
type ocAdm struct {
	// Since is an ISO timestamp or relative date from which to show logs
	Since string
	// Until is an ISO timestamp or relative date until which to show logs
	Until string
	// Format is the alternate format (short, cat, json, short-unix) to display journal logs
	Format string
	// CaseSensitive controls the case sensitivity of pattern searches
	CaseSensitive bool
}

// newNodeLogQuery parses query values and converts all known options into nodeLogQuery
func newNodeLogQuery(query url.Values) (*nodeLogQuery, field.ErrorList) {
	allErrs := field.ErrorList{}
	var nlq nodeLogQuery
	var err error

	queries, okQuery := query["query"]
	if len(queries) > 0 {
		for _, q := range queries {
			// The presence of / or \ is a hint that the query is for a log file. If the query is for foo.log without a
			// slash prefix, the heuristics will still return the file contents.
			if strings.ContainsAny(q, `/\`) {
				nlq.Files = append(nlq.Files, q)
			} else if strings.TrimSpace(q) != "" { // Prevent queries with just spaces
				nlq.Services = append(nlq.Services, q)
			}
		}
	}
	units, okUnit := query["unit"]
	if len(units) > 0 {
		for _, u := range units {
			// We don't check for files as the heuristics do not apply to unit
			if strings.TrimSpace(u) != "" { // Prevent queries with just spaces
				nlq.Services = append(nlq.Services, u)
			}
		}
	}

	// Prevent specifying  an empty or blank space query.
	// Example: kubectl get --raw /api/v1/nodes/$node/proxy/logs?query="   "
	if (okQuery || okUnit) && (len(nlq.Files) == 0 && len(nlq.Services) == 0) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("unit"), queries, "unit cannot be empty"))
	}

	var sinceTime time.Time
	sinceTimeValue := query.Get("sinceTime")
	if len(sinceTimeValue) > 0 {
		sinceTime, err = time.Parse(time.RFC3339, sinceTimeValue)
		if err != nil {
			allErrs = append(allErrs, field.Invalid(field.NewPath("sinceTime"), sinceTimeValue, "invalid time format"))
		} else {
			nlq.SinceTime = &sinceTime
		}
	}

	var untilTime time.Time
	untilTimeValue := query.Get("untilTime")
	if len(untilTimeValue) > 0 {
		untilTime, err = time.Parse(time.RFC3339, untilTimeValue)
		if err != nil {
			allErrs = append(allErrs, field.Invalid(field.NewPath("untilTime"), untilTimeValue, "invalid time format"))
		} else {
			nlq.UntilTime = &untilTime
		}
	}

	var boot int
	bootValue := query.Get("boot")
	if len(bootValue) > 0 {
		boot, err = strconv.Atoi(bootValue)
		if err != nil {
			allErrs = append(allErrs, field.Invalid(field.NewPath("boot"), bootValue, err.Error()))
		} else {
			nlq.Boot = &boot
		}
	}

	var tailLines int
	tailLinesValue := query.Get("tailLines")
	if len(tailLinesValue) == 0 {
		tailLinesValue = query.Get("tail")
	}
	if len(tailLinesValue) > 0 {
		tailLines, err = strconv.Atoi(tailLinesValue)
		if err != nil {
			allErrs = append(allErrs, field.Invalid(field.NewPath("tailLines"), tailLinesValue, err.Error()))
		} else {
			nlq.TailLines = &tailLines
		}
	}

	pattern := query.Get("pattern")
	if len(pattern) == 0 {
		pattern = query.Get("grep")
	}
	if len(pattern) > 0 {
		nlq.Pattern = pattern
		caseSensitiveValue := query.Get("case-sensitive")
		if len(caseSensitiveValue) > 0 {
			caseSensitive, err := strconv.ParseBool(query.Get("case-sensitive"))
			if err != nil {
				allErrs = append(allErrs, field.Invalid(field.NewPath("case-sensitive"), query.Get("case-sensitive"),
					err.Error()))
			} else {
				nlq.CaseSensitive = caseSensitive
			}
		}
	}

	nlq.Since = query.Get("since")
	nlq.Until = query.Get("until")
	nlq.Format = query.Get("output")

	if len(allErrs) > 0 {
		return nil, allErrs
	}

	return &nlq, allErrs
}

func validateServices(services []string) field.ErrorList {
	allErrs := field.ErrorList{}

	for _, s := range services {
		if err := safeServiceName(s); err != nil {
			allErrs = append(allErrs, field.Invalid(field.NewPath("query"), s, err.Error()))
		}
	}

	if len(services) > maxServices {
		allErrs = append(allErrs, field.TooMany(field.NewPath("query"), len(services), maxServices))
	}
	return allErrs
}

func (n *nodeLogQuery) validate() field.ErrorList {
	allErrs := validateServices(n.Services)
	switch {
	// OCP: Allow len(n.Files) == 0 && len(n.Services) == 0 as we want to be able to return all journal / WinEvent logs
	case len(n.Files) > 0 && len(n.Services) > 0:
		allErrs = append(allErrs, field.Invalid(field.NewPath("query"), fmt.Sprintf("%v, %v", n.Files, n.Services),
			"cannot specify a file and service"))
	case len(n.Files) > 1:
		allErrs = append(allErrs, field.Invalid(field.NewPath("query"), n.Files, "cannot specify more than one file"))
	case len(n.Files) == 1 && !reflect.DeepEqual(n.options, options{}):
		allErrs = append(allErrs, field.Invalid(field.NewPath("query"), n.Files, "cannot specify file with options"))
	case len(n.Files) == 1:
		if fullLogFilename, err := securejoin.SecureJoin(nodeLogDir, n.Files[0]); err != nil {
			allErrs = append(allErrs, field.Invalid(field.NewPath("query"), n.Files, err.Error()))
		} else if _, err := os.Stat(fullLogFilename); err != nil {
			allErrs = append(allErrs, field.Invalid(field.NewPath("query"), n.Files, err.Error()))
		}
	}

	if n.SinceTime != nil && n.UntilTime != nil && (n.SinceTime.After(*n.UntilTime)) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("untilTime"), n.UntilTime, "must be after `sinceTime`"))
	}

	if n.Boot != nil && runtime.GOOS == "windows" {
		allErrs = append(allErrs, field.Invalid(field.NewPath("boot"), *n.Boot, "boot is not supported on Windows"))
	}

	if n.Boot != nil && *n.Boot > 0 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("boot"), *n.Boot, "must be less than 1"))
	}

	if n.TailLines != nil {
		if err := utilvalidation.IsInRange((int)(*n.TailLines), 0, maxTailLines); err != nil {
			allErrs = append(allErrs, field.Invalid(field.NewPath("tailLines"), *n.TailLines, err[0]))
		}
	}

	if _, err := syntax.Parse(n.Pattern, syntax.Perl); err != nil {
		allErrs = append(allErrs, field.Invalid(field.NewPath("pattern"), n.Pattern, err.Error()))
	}

	// "oc adm node-logs" specific validation

	if n.SinceTime != nil && (len(n.Since) > 0 || len(n.Until) > 0) {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("sinceTime"),
			"`since or until` and `sinceTime` cannot be specified"))
	}

	if n.UntilTime != nil && (len(n.Since) > 0 || len(n.Until) > 0) {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("untilTime"),
			"`since or until` and `untilTime` cannot be specified"))
	}

	if err := validateDate(n.Since); err != nil {
		allErrs = append(allErrs, field.Invalid(field.NewPath("since"), n.Since, err.Error()))
	}

	if err := validateDate(n.Until); err != nil {
		allErrs = append(allErrs, field.Invalid(field.NewPath("until"), n.Until, err.Error()))
	}

	allowedFormats := sets.New[string]("short-precise", "json", "short", "short-unix", "short-iso",
		"short-iso-precise", "cat", "")
	if len(n.Format) > 0 && runtime.GOOS == "windows" {
		allErrs = append(allErrs, field.Invalid(field.NewPath("output"), n.Format,
			"output is not supported on Windows"))
	} else if !allowedFormats.Has(n.Format) {
		allErrs = append(allErrs, field.NotSupported(field.NewPath("output"), n.Format, allowedFormats.UnsortedList()))
	}

	return allErrs
}

// Copy streams the contents of the OS specific logging command executed  with the current args to the provided
// writer. If an error occurs a line is written to the output.
func (n *nodeLogQuery) Copy(w io.Writer) {
	// set the deadline to the maximum across both runs
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(30*time.Second))
	defer cancel()
	boot := 0
	if n.Boot != nil {
		boot = *n.Boot
	}
	n.copyForBoot(ctx, w, boot)
}

// copyForBoot invokes the OS specific logging command with the  provided args
func (n *nodeLogQuery) copyForBoot(ctx context.Context, w io.Writer, previousBoot int) {
	if ctx.Err() != nil {
		return
	}
	nativeLoggers, fileLoggers := n.splitNativeVsFileLoggers(ctx)

	if len(fileLoggers) > 0 && !reflect.DeepEqual(n.options, options{}) {
		fmt.Fprintf(w, "\noptions present and query resolved to log files for %v\ntry without specifying options\n",
			fileLoggers)
		return
	}

	if len(fileLoggers) > 0 {
		copyFileLogs(ctx, w, fileLoggers)
		return
	}
	// OCP: Return all logs in the case where nativeLoggers == ""
	n.copyServiceLogs(ctx, w, nativeLoggers, previousBoot)

}

// splitNativeVsFileLoggers checks if each service logs to native OS logs or to a file and returns a list of services
// that log natively vs maybe to a file
func (n *nodeLogQuery) splitNativeVsFileLoggers(ctx context.Context) ([]string, []string) {
	var nativeLoggers []string
	var fileLoggers []string

	for _, service := range n.Services {
		// Check the journalctl output to figure if the service is using journald or not. This is not needed in the
		// Get-WinEvent case as the command returns an error if a service is not logging to the Application provider.
		if checkForNativeLogger(ctx, service) {
			nativeLoggers = append(nativeLoggers, service)
		} else {
			fileLoggers = append(fileLoggers, service)
		}
	}
	return nativeLoggers, fileLoggers
}

// copyServiceLogs invokes journalctl or Get-WinEvent with the provided args. Note that
// services are explicitly passed here to account for the heuristics.
func (n *nodeLogQuery) copyServiceLogs(ctx context.Context, w io.Writer, services []string, previousBoot int) {
	cmdStr, args, err := getLoggingCmd(n, services)
	if err != nil {
		fmt.Fprintf(w, "\nfailed to get logging cmd: %v\n", err)
		return
	}
	cmd := exec.CommandContext(ctx, cmdStr, args...)
	cmd.Stdout = w
	cmd.Stderr = w

	if err := cmd.Run(); err != nil {
		if _, ok := err.(*exec.ExitError); ok {
			return
		}
		if previousBoot == 0 {
			fmt.Fprintf(w, "\nerror: journal output not available\n")
		}
	}
}

// copyFileLogs loops over all the services and attempts to collect the file logs of each service
func copyFileLogs(ctx context.Context, w io.Writer, services []string) {
	if ctx.Err() != nil {
		fmt.Fprintf(w, "\ncontext error: %v\n", ctx.Err())
		return
	}

	for _, service := range services {
		heuristicsCopyFileLogs(ctx, w, service)
	}
}

// heuristicsCopyFileLogs attempts to collect logs from either
// /var/log/service
// /var/log/service.log or
// /var/log/service/service.log or
// in that order stopping on first success.
func heuristicsCopyFileLogs(ctx context.Context, w io.Writer, service string) {
	logFileNames := [3]string{
		service,
		fmt.Sprintf("%s.log", service),
		fmt.Sprintf("%s/%s.log", service, service),
	}

	var err error
	for _, logFileName := range logFileNames {
		var logFile string
		logFile, err = securejoin.SecureJoin(nodeLogDir, logFileName)
		if err != nil {
			break
		}
		err = heuristicsCopyFileLog(ctx, w, logFile)
		if err == nil {
			break
		} else if errors.Is(err, os.ErrNotExist) {
			continue
		} else {
			break
		}
	}

	if err != nil {
		// If the last error was file not found it implies that no log file was found for the service
		if errors.Is(err, os.ErrNotExist) {
			fmt.Fprintf(w, "\nlog not found for %s\n", service)
			return
		}
		fmt.Fprintf(w, "\nerror getting log for %s: %v\n", service, err)
	}
}

// readerCtx is the interface that wraps io.Reader with a context
type readerCtx struct {
	ctx context.Context
	io.Reader
}

func (r *readerCtx) Read(p []byte) (n int, err error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.Reader.Read(p)
}

// newReaderCtx gets a context-aware io.Reader
func newReaderCtx(ctx context.Context, r io.Reader) io.Reader {
	return &readerCtx{
		ctx:    ctx,
		Reader: r,
	}
}

// heuristicsCopyFileLog returns the contents of the given logFile
func heuristicsCopyFileLog(ctx context.Context, w io.Writer, logFile string) error {
	fInfo, err := os.Stat(logFile)
	if err != nil {
		return err
	}
	// This is to account for the heuristics where logs for service foo
	// could be in /var/log/foo/
	if fInfo.IsDir() {
		return os.ErrNotExist
	}

	f, err := os.Open(logFile)
	if err != nil {
		return err
	}
	defer f.Close()

	if _, err := io.Copy(w, newReaderCtx(ctx, f)); err != nil {
		return err
	}
	return nil
}

func safeServiceName(s string) error {
	// Max length of a service name is 256 across supported OSes
	if len(s) > maxServiceLength {
		return fmt.Errorf("length must be less than 100")
	}

	if reServiceNameUnsafeCharacters.MatchString(s) {
		return fmt.Errorf("input contains unsupported characters")
	}
	return nil
}

func validateDate(date string) error {
	if len(date) == 0 {
		return nil
	}
	if reRelativeDate.MatchString(date) {
		return nil
	}
	if _, err := time.Parse(dateLayout, date); err == nil {
		return nil
	}
	return fmt.Errorf("date must be a relative time of the form '(+|-)[0-9]+(s|m|h|d)' or a date in 'YYYY-MM-DD HH:MM:SS' form")
}
