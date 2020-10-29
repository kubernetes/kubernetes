package kubelet

import (
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
)

var journal = journalServer{}

// journalServer returns text output from the system journal to view from
// the client. It runs with the privileges of the calling process (the
// kubelet) and should only be allowed to be invoked by a root user.
type journalServer struct{}

// ServeHTTP translates HTTP query parameters into arguments to be passed
// to journalctl on the current system. It supports content-encoding of
// gzip to reduce total content size.
func (journalServer) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	var out io.Writer = w
	args, err := newJournalArgsFromURL(req.URL.Query())
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "text/plain;charset=UTF-8")
	if req.Header.Get("Accept-Encoding") == "gzip" {
		w.Header().Set("Content-Encoding", "gzip")

		gz := gzip.NewWriter(out)
		defer gz.Close()
		out = gz
	}
	args.Copy(out)
}

// journalArgs assists in invoking the journalctl command.
type journalArgs struct {
	Since         string
	Until         string
	Tail          int
	Timeout       int
	Format        string
	Boot          *int
	Units         []string
	Pattern       string
	CaseSensitive bool
}

// newJournalArgsFromURL returns bounds checked values based on query
// parameters. Parameter names are deliberately chosen to align with
// journalctl arguments. If journalctl ever creates incompatible arguments,
// this method should introduce new parameters that preserves compatibility.
func newJournalArgsFromURL(query url.Values) (*journalArgs, error) {
	since, err := validJournalDateRange(query.Get("since"))
	if err != nil {
		return nil, fmt.Errorf("parameter 'since' is invalid: %v", err)
	}
	until, err := validJournalDateRange(query.Get("until"))
	if err != nil {
		return nil, fmt.Errorf("parameter 'until' is invalid: %v", err)
	}
	format, err := stringInSlice(query.Get("output"), "short-precise", "json", "short", "short-unix", "short-iso", "short-iso-precise", "cat", "")
	if err != nil {
		return nil, fmt.Errorf("parameter 'output' is invalid: %v", err)
	}
	if len(format) == 0 {
		format = "short-precise"
	}
	units, err := safeStrings(query["unit"])
	if err != nil {
		return nil, fmt.Errorf("parameter 'unit' is invalid: %v", err)
	}
	var boot *int
	if bootStr := query.Get("boot"); len(bootStr) > 0 {
		boot, err = validIntRange(bootStr, -100, 0)
		if err != nil {
			return nil, fmt.Errorf("parameter 'boot' is invalid: %v", err)
		}
	}
	pattern, err := safeString(query.Get("grep"))
	if err != nil {
		return nil, fmt.Errorf("parameter 'grep' is invalid: %v", err)
	}

	// All parameters loaded from the query must be thoroughly sanitized - do
	// not pass query parameters directly to journalctl without limiting them
	// as demonstrated above.
	return &journalArgs{
		Units: units,

		Since: since,
		Until: until,
		Tail:  boundedIntegerOrDefault(query.Get("tail"), 0, 100000, 0),
		Boot:  boot,

		Timeout: boundedIntegerOrDefault(query.Get("timeout"), 1, 60, 30),

		Pattern:       pattern,
		CaseSensitive: boolean(query.Get("case-sensitive"), true),

		Format: format,
	}, nil
}

// Args returns the journalctl arguments for the given args.
func (a *journalArgs) Args() []string {
	args := []string{
		"--utc",
		"--no-pager",
	}
	if len(a.Since) > 0 {
		args = append(args, "--since="+a.Since)
	}
	if len(a.Until) > 0 {
		args = append(args, "--until="+a.Until)
	}
	if a.Tail > 0 {
		args = append(args, "--pager-end", fmt.Sprintf("--lines=%d", a.Tail))
	}
	if len(a.Format) > 0 {
		args = append(args, "--output="+a.Format)
	}
	for _, unit := range a.Units {
		if len(unit) > 0 {
			args = append(args, "--unit="+unit)
		}
	}
	if len(a.Pattern) > 0 {
		args = append(args, "--grep="+a.Pattern)
		args = append(args, fmt.Sprintf("--case-sensitive=%t", a.CaseSensitive))
	}
	return args
}

// Copy streams the contents of the journalctl command executed with the current
// args to the provided writer, timing out at a.Timeout. If an error occurs a line
// is written to the output.
func (a *journalArgs) Copy(w io.Writer) {
	// set the deadline to the maximum across both runs
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(time.Duration(a.Timeout)*time.Second))
	defer cancel()
	if a.Boot != nil {
		a.copyForBoot(ctx, w, *a.Boot)
	} else {
		// show the previous boot if possible, eating errors
		a.copyForBoot(ctx, w, -1)
		// show the current boot
		a.copyForBoot(ctx, w, 0)
	}
}

// copyForBoot invokes the provided args for a named boot record. If previousBoot is != 0, then
// errors are silently ignored.
func (a *journalArgs) copyForBoot(ctx context.Context, w io.Writer, previousBoot int) {
	if ctx.Err() != nil {
		return
	}

	args := a.Args()
	args = append(args, "--boot", fmt.Sprintf("%d", previousBoot))
	cmd := exec.Command("journalctl", args...)
	cmd.Stdout = w
	cmd.Stderr = w

	// force termination
	go func() {
		<-ctx.Done()
		if p := cmd.Process; p != nil {
			p.Kill()
		}
	}()

	if err := cmd.Run(); err != nil {
		if _, ok := err.(*exec.ExitError); ok {
			return
		}
		if previousBoot == 0 {
			fmt.Fprintf(w, "error: journal output not available\n")
		}
	}
}

func stringInSlice(s string, allowed ...string) (string, error) {
	for _, allow := range allowed {
		if s == allow {
			return allow, nil
		}
	}
	return "", fmt.Errorf("only the following values are allowed: %s", strings.Join(allowed, ", "))
}

func boolean(s string, defaultValue bool) bool {
	if len(s) == 0 {
		return defaultValue
	}
	if s == "1" || s == "true" {
		return true
	}
	return false
}

func validIntRange(s string, min, max int) (*int, error) {
	i, err := strconv.Atoi(s)
	if err != nil {
		return nil, err
	}
	if i < min || i > max {
		return nil, fmt.Errorf("integer must be in range [%d, %d]", min, max)
	}
	return &i, nil
}

func boundedIntegerOrDefault(s string, min, max, defaultValue int) int {
	i, err := strconv.Atoi(s)
	if err != nil {
		i = defaultValue
	}
	if i < min {
		i = min
	}
	if i > max {
		i = max
	}
	return i
}

var (
	reRelativeDate = regexp.MustCompile(`^(\+|\-)?[\d]+(s|m|h|d)$`)
	// The set of known safe characters to pass to journalctl flags - only
	// add to this list if the character cannot be used to create invalid
	// sequences. This is intended as a broad defense against malformed
	// input that could cause a journalctl escape.
	reUnsafeCharacters = regexp.MustCompile(`[^a-zA-Z\-_.0-9\s@]+`)
)

const (
	dateFormat         = `2006-01-02 15:04:05.999999`
	maxParameterLength = 100
	maxTotalLength     = 1000
)

func validJournalDateRange(s string) (string, error) {
	if len(s) == 0 {
		return "", nil
	}
	if reRelativeDate.MatchString(s) {
		return s, nil
	}
	if _, err := time.Parse(dateFormat, s); err == nil {
		return s, nil
	}
	return "", fmt.Errorf("date must be a relative time of the form '(+|-)[0-9]+(s|m|h|d)' or a date in 'YYYY-MM-DD HH:MM:SS' form")
}

func safeString(s string) (string, error) {
	if len(s) > maxParameterLength {
		return "", fmt.Errorf("input is too long, max length is %d", maxParameterLength)
	}
	if reUnsafeCharacters.MatchString(s) {
		return "", fmt.Errorf("input contains unsupported characters")
	}
	return s, nil
}

func safeStrings(arr []string) ([]string, error) {
	var out []string
	var total int
	for _, s := range arr {
		s, err := safeString(s)
		if err != nil {
			return nil, err
		}
		total += len(s)
		if total > maxTotalLength {
			return nil, fmt.Errorf("total input length across all values must be less than %d", maxTotalLength)
		}
		out = append(out, s)
	}
	return out, nil
}
