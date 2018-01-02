package build

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"runtime"
	"strconv"
	"strings"
	"sync"

	apierrors "github.com/docker/docker/api/errors"
	"github.com/docker/docker/api/server/httputils"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/backend"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/versions"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/progress"
	"github.com/docker/docker/pkg/streamformatter"
	units "github.com/docker/go-units"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

func newImageBuildOptions(ctx context.Context, r *http.Request) (*types.ImageBuildOptions, error) {
	version := httputils.VersionFromContext(ctx)
	options := &types.ImageBuildOptions{}
	if httputils.BoolValue(r, "forcerm") && versions.GreaterThanOrEqualTo(version, "1.12") {
		options.Remove = true
	} else if r.FormValue("rm") == "" && versions.GreaterThanOrEqualTo(version, "1.12") {
		options.Remove = true
	} else {
		options.Remove = httputils.BoolValue(r, "rm")
	}
	if httputils.BoolValue(r, "pull") && versions.GreaterThanOrEqualTo(version, "1.16") {
		options.PullParent = true
	}

	options.Dockerfile = r.FormValue("dockerfile")
	options.SuppressOutput = httputils.BoolValue(r, "q")
	options.NoCache = httputils.BoolValue(r, "nocache")
	options.ForceRemove = httputils.BoolValue(r, "forcerm")
	options.MemorySwap = httputils.Int64ValueOrZero(r, "memswap")
	options.Memory = httputils.Int64ValueOrZero(r, "memory")
	options.CPUShares = httputils.Int64ValueOrZero(r, "cpushares")
	options.CPUPeriod = httputils.Int64ValueOrZero(r, "cpuperiod")
	options.CPUQuota = httputils.Int64ValueOrZero(r, "cpuquota")
	options.CPUSetCPUs = r.FormValue("cpusetcpus")
	options.CPUSetMems = r.FormValue("cpusetmems")
	options.CgroupParent = r.FormValue("cgroupparent")
	options.NetworkMode = r.FormValue("networkmode")
	options.Tags = r.Form["t"]
	options.ExtraHosts = r.Form["extrahosts"]
	options.SecurityOpt = r.Form["securityopt"]
	options.Squash = httputils.BoolValue(r, "squash")
	options.Target = r.FormValue("target")
	options.RemoteContext = r.FormValue("remote")

	if r.Form.Get("shmsize") != "" {
		shmSize, err := strconv.ParseInt(r.Form.Get("shmsize"), 10, 64)
		if err != nil {
			return nil, err
		}
		options.ShmSize = shmSize
	}

	if i := container.Isolation(r.FormValue("isolation")); i != "" {
		if !container.Isolation.IsValid(i) {
			return nil, fmt.Errorf("Unsupported isolation: %q", i)
		}
		options.Isolation = i
	}

	if runtime.GOOS != "windows" && options.SecurityOpt != nil {
		return nil, fmt.Errorf("The daemon on this platform does not support setting security options on build")
	}

	var buildUlimits = []*units.Ulimit{}
	ulimitsJSON := r.FormValue("ulimits")
	if ulimitsJSON != "" {
		if err := json.Unmarshal([]byte(ulimitsJSON), &buildUlimits); err != nil {
			return nil, err
		}
		options.Ulimits = buildUlimits
	}

	// Note that there are two ways a --build-arg might appear in the
	// json of the query param:
	//     "foo":"bar"
	// and "foo":nil
	// The first is the normal case, ie. --build-arg foo=bar
	// or  --build-arg foo
	// where foo's value was picked up from an env var.
	// The second ("foo":nil) is where they put --build-arg foo
	// but "foo" isn't set as an env var. In that case we can't just drop
	// the fact they mentioned it, we need to pass that along to the builder
	// so that it can print a warning about "foo" being unused if there is
	// no "ARG foo" in the Dockerfile.
	buildArgsJSON := r.FormValue("buildargs")
	if buildArgsJSON != "" {
		var buildArgs = map[string]*string{}
		if err := json.Unmarshal([]byte(buildArgsJSON), &buildArgs); err != nil {
			return nil, err
		}
		options.BuildArgs = buildArgs
	}

	labelsJSON := r.FormValue("labels")
	if labelsJSON != "" {
		var labels = map[string]string{}
		if err := json.Unmarshal([]byte(labelsJSON), &labels); err != nil {
			return nil, err
		}
		options.Labels = labels
	}

	cacheFromJSON := r.FormValue("cachefrom")
	if cacheFromJSON != "" {
		var cacheFrom = []string{}
		if err := json.Unmarshal([]byte(cacheFromJSON), &cacheFrom); err != nil {
			return nil, err
		}
		options.CacheFrom = cacheFrom
	}
	options.SessionID = r.FormValue("session")

	return options, nil
}

func (br *buildRouter) postPrune(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	report, err := br.backend.PruneCache(ctx)
	if err != nil {
		return err
	}
	return httputils.WriteJSON(w, http.StatusOK, report)
}

func (br *buildRouter) postBuild(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	var (
		notVerboseBuffer = bytes.NewBuffer(nil)
		version          = httputils.VersionFromContext(ctx)
	)

	w.Header().Set("Content-Type", "application/json")

	output := ioutils.NewWriteFlusher(w)
	defer output.Close()
	errf := func(err error) error {
		if httputils.BoolValue(r, "q") && notVerboseBuffer.Len() > 0 {
			output.Write(notVerboseBuffer.Bytes())
		}
		// Do not write the error in the http output if it's still empty.
		// This prevents from writing a 200(OK) when there is an internal error.
		if !output.Flushed() {
			return err
		}
		_, err = w.Write(streamformatter.FormatError(err))
		if err != nil {
			logrus.Warnf("could not write error response: %v", err)
		}
		return nil
	}

	buildOptions, err := newImageBuildOptions(ctx, r)
	if err != nil {
		return errf(err)
	}
	buildOptions.AuthConfigs = getAuthConfigs(r.Header)

	if buildOptions.Squash && !br.daemon.HasExperimental() {
		return apierrors.NewBadRequestError(
			errors.New("squash is only supported with experimental mode"))
	}

	out := io.Writer(output)
	if buildOptions.SuppressOutput {
		out = notVerboseBuffer
	}

	// Currently, only used if context is from a remote url.
	// Look at code in DetectContextFromRemoteURL for more information.
	createProgressReader := func(in io.ReadCloser) io.ReadCloser {
		progressOutput := streamformatter.NewJSONProgressOutput(out, true)
		return progress.NewProgressReader(in, progressOutput, r.ContentLength, "Downloading context", buildOptions.RemoteContext)
	}

	wantAux := versions.GreaterThanOrEqualTo(version, "1.30")

	imgID, err := br.backend.Build(ctx, backend.BuildConfig{
		Source:         r.Body,
		Options:        buildOptions,
		ProgressWriter: buildProgressWriter(out, wantAux, createProgressReader),
	})
	if err != nil {
		return errf(err)
	}

	// Everything worked so if -q was provided the output from the daemon
	// should be just the image ID and we'll print that to stdout.
	if buildOptions.SuppressOutput {
		fmt.Fprintln(streamformatter.NewStdoutWriter(output), imgID)
	}
	return nil
}

func getAuthConfigs(header http.Header) map[string]types.AuthConfig {
	authConfigs := map[string]types.AuthConfig{}
	authConfigsEncoded := header.Get("X-Registry-Config")

	if authConfigsEncoded == "" {
		return authConfigs
	}

	authConfigsJSON := base64.NewDecoder(base64.URLEncoding, strings.NewReader(authConfigsEncoded))
	// Pulling an image does not error when no auth is provided so to remain
	// consistent with the existing api decode errors are ignored
	json.NewDecoder(authConfigsJSON).Decode(&authConfigs)
	return authConfigs
}

type syncWriter struct {
	w  io.Writer
	mu sync.Mutex
}

func (s *syncWriter) Write(b []byte) (count int, err error) {
	s.mu.Lock()
	count, err = s.w.Write(b)
	s.mu.Unlock()
	return
}

func buildProgressWriter(out io.Writer, wantAux bool, createProgressReader func(io.ReadCloser) io.ReadCloser) backend.ProgressWriter {
	out = &syncWriter{w: out}

	var aux *streamformatter.AuxFormatter
	if wantAux {
		aux = &streamformatter.AuxFormatter{Writer: out}
	}

	return backend.ProgressWriter{
		Output:             out,
		StdoutFormatter:    streamformatter.NewStdoutWriter(out),
		StderrFormatter:    streamformatter.NewStderrWriter(out),
		AuxFormatter:       aux,
		ProgressReaderFunc: createProgressReader,
	}
}
