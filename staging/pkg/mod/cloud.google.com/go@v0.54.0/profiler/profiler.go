// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package profiler is a client for the Stackdriver Profiler service.
//
// This package is still experimental and subject to change.
//
// Usage example:
//
//   import "cloud.google.com/go/profiler"
//   ...
//   if err := profiler.Start(profiler.Config{Service: "my-service"}); err != nil {
//       // TODO: Handle error.
//   }
//
// Calling Start will start a goroutine to collect profiles and upload to
// the profiler server, at the rhythm specified by the server.
//
// The caller must provide the service string in the config, and may provide
// other information as well. See Config for details.
//
// Profiler has CPU, heap and goroutine profiling enabled by default. Mutex
// profiling can be enabled in the config. Note that goroutine and mutex
// profiles are shown as "threads" and "contention" profiles in the profiler
// UI.
package profiler

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"regexp"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"

	gcemd "cloud.google.com/go/compute/metadata"
	"cloud.google.com/go/internal/version"
	"github.com/golang/protobuf/proto"
	"github.com/golang/protobuf/ptypes"
	"github.com/google/pprof/profile"
	gax "github.com/googleapis/gax-go/v2"
	"google.golang.org/api/option"
	gtransport "google.golang.org/api/transport/grpc"
	pb "google.golang.org/genproto/googleapis/devtools/cloudprofiler/v2"
	edpb "google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	grpcmd "google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

var (
	config       Config
	startOnce    allowUntilSuccess
	mutexEnabled bool
	// The functions below are stubbed to be overrideable for testing.
	getProjectID     = gcemd.ProjectID
	getInstanceName  = gcemd.InstanceName
	getZone          = gcemd.Zone
	startCPUProfile  = pprof.StartCPUProfile
	stopCPUProfile   = pprof.StopCPUProfile
	writeHeapProfile = pprof.WriteHeapProfile
	sleep            = gax.Sleep
	dialGRPC         = gtransport.DialPool
	onGCE            = gcemd.OnGCE
	serviceRegexp    = regexp.MustCompile(`^[a-z]([-a-z0-9_.]{0,253}[a-z0-9])?$`)

	// For testing only.
	// When the profiling loop has exited without error and this channel is
	// non-nil, "true" will be sent to this channel.
	profilingDone chan bool
)

const (
	apiAddress       = "cloudprofiler.googleapis.com:443"
	xGoogAPIMetadata = "x-goog-api-client"
	zoneNameLabel    = "zone"
	versionLabel     = "version"
	languageLabel    = "language"
	instanceLabel    = "instance"
	scope            = "https://www.googleapis.com/auth/monitoring.write"

	initialBackoff = time.Minute
	// Ensure the agent will recover within 1 hour.
	maxBackoff        = time.Hour
	backoffMultiplier = 1.3 // Backoff envelope increases by this factor on each retry.
	retryInfoMetadata = "google.rpc.retryinfo-bin"
)

// Config is the profiler configuration.
type Config struct {
	// Service must be provided to start the profiler. It specifies the name of
	// the service under which the profiled data will be recorded and exposed at
	// the Profiler UI for the project. You can specify an arbitrary string, but
	// see Deployment.target at
	// https://github.com/googleapis/googleapis/blob/master/google/devtools/cloudprofiler/v2/profiler.proto
	// for restrictions. If the parameter is not set, the agent will probe
	// GAE_SERVICE environment variable which is present in Google App Engine
	// environment.
	// NOTE: The string should be the same across different replicas of
	// your service so that the globally constant profiling rate is
	// maintained. Do not put things like PID or unique pod ID in the name.
	Service string

	// ServiceVersion is an optional field specifying the version of the
	// service. It can be an arbitrary string. Profiler profiles
	// once per minute for each version of each service in each zone.
	// ServiceVersion defaults to GAE_VERSION environment variable if that is
	// set, or to empty string otherwise.
	ServiceVersion string

	// DebugLogging enables detailed debug logging from profiler. It
	// defaults to false.
	DebugLogging bool

	// MutexProfiling enables mutex profiling. It defaults to false.
	// Note that mutex profiling is not supported by Go versions older
	// than Go 1.8.
	MutexProfiling bool

	// When true, collecting the CPU profiles is disabled.
	NoCPUProfiling bool

	// When true, collecting the allocation profiles is disabled.
	NoAllocProfiling bool

	// AllocForceGC forces garbage collection before the collection of each heap
	// profile collected to produce the allocation profile. This increases the
	// accuracy of allocation profiling. It defaults to false.
	AllocForceGC bool

	// When true, collecting the heap profiles is disabled.
	NoHeapProfiling bool

	// When true, collecting the goroutine profiles is disabled.
	NoGoroutineProfiling bool

	// ProjectID is the Cloud Console project ID to use instead of the one set by
	// GOOGLE_CLOUD_PROJECT environment variable or read from the VM metadata
	// server.
	//
	// Set this if you are running the agent in your local environment
	// or anywhere else outside of Google Cloud Platform.
	ProjectID string

	// APIAddr is the HTTP endpoint to use to connect to the profiler
	// agent API. Defaults to the production environment, overridable
	// for testing.
	APIAddr string

	// Instance is the name of Compute Engine instance the profiler agent runs
	// on. This is normally determined from the Compute Engine metadata server
	// and doesn't need to be initialized. It needs to be set in rare cases where
	// the metadata server is present but is flaky or otherwise misbehave.
	Instance string

	// Zone is the zone of Compute Engine instance the profiler agent runs
	// on. This is normally determined from the Compute Engine metadata server
	// and doesn't need to be initialized. It needs to be set in rare cases where
	// the metadata server is present but is flaky or otherwise misbehave.
	Zone string

	// numProfiles is the number of profiles which should be collected before
	// the profile collection loop exits.When numProfiles is 0, profiles will
	// be collected for the duration of the program. For testing only.
	numProfiles int
}

// allowUntilSuccess is an object that will perform action till
// it succeeds once.
// This is a modified form of Go's sync.Once
type allowUntilSuccess struct {
	m    sync.Mutex
	done uint32
}

// do calls function f only if it hasnt returned nil previously.
// Once f returns nil, do will not call function f any more.
// This is a modified form of Go's sync.Once.Do
func (o *allowUntilSuccess) do(f func() error) (err error) {
	o.m.Lock()
	defer o.m.Unlock()
	if o.done == 0 {
		if err = f(); err == nil {
			o.done = 1
		}
	} else {
		debugLog("profiler.Start() called again after it was previously called")
		err = nil
	}
	return err
}

// Start starts a goroutine to collect and upload profiles. The
// caller must provide the service string in the config. See
// Config for details. Start should only be called once. Any
// additional calls will be ignored.
func Start(cfg Config, options ...option.ClientOption) error {
	startError := startOnce.do(func() error {
		return start(cfg, options...)
	})
	return startError
}

func start(cfg Config, options ...option.ClientOption) error {
	if err := initializeConfig(cfg); err != nil {
		debugLog("failed to initialize config: %v", err)
		return err
	}
	if config.MutexProfiling {
		if mutexEnabled = enableMutexProfiling(); !mutexEnabled {
			return fmt.Errorf("mutex profiling is not supported by %s, requires Go 1.8 or later", runtime.Version())
		}
	}

	ctx := context.Background()

	opts := []option.ClientOption{
		option.WithEndpoint(config.APIAddr),
		option.WithScopes(scope),
		option.WithUserAgent(fmt.Sprintf("gcloud-go-profiler/%s", version.Repo)),
	}
	opts = append(opts, options...)

	connPool, err := dialGRPC(ctx, opts...)
	if err != nil {
		debugLog("failed to dial GRPC: %v", err)
		return err
	}

	a, err := initializeAgent(pb.NewProfilerServiceClient(connPool))
	if err != nil {
		debugLog("failed to start the profiling agent: %v", err)
		return err
	}
	go pollProfilerService(withXGoogHeader(ctx), a)
	return nil
}

func debugLog(format string, e ...interface{}) {
	if config.DebugLogging {
		log.Printf(format, e...)
	}
}

// agent polls the profiler server for instructions on behalf of a task,
// and collects and uploads profiles as requested.
type agent struct {
	client        pb.ProfilerServiceClient
	deployment    *pb.Deployment
	profileLabels map[string]string
	profileTypes  []pb.ProfileType
}

// abortedBackoffDuration retrieves the retry duration from gRPC trailing
// metadata, which is set by the profiler server.
func abortedBackoffDuration(md grpcmd.MD) (time.Duration, error) {
	elem := md[retryInfoMetadata]
	if len(elem) <= 0 {
		return 0, errors.New("no retry info")
	}

	var retryInfo edpb.RetryInfo
	if err := proto.Unmarshal([]byte(elem[0]), &retryInfo); err != nil {
		return 0, err
	} else if time, err := ptypes.Duration(retryInfo.RetryDelay); err != nil {
		return 0, err
	} else {
		if time < 0 {
			return 0, errors.New("negative retry duration")
		}
		return time, nil
	}
}

type retryer struct {
	backoff gax.Backoff
	md      grpcmd.MD
}

func (r *retryer) Retry(err error) (time.Duration, bool) {
	st, _ := status.FromError(err)
	if st != nil && st.Code() == codes.Aborted {
		dur, err := abortedBackoffDuration(r.md)
		if err == nil {
			return dur, true
		}
		debugLog("failed to get backoff duration: %v", err)
	}
	return r.backoff.Pause(), true
}

// createProfile talks to the profiler server to create profile. In
// case of error, the goroutine will sleep and retry. Sleep duration may
// be specified by the server. Otherwise it will be an exponentially
// increasing value, bounded by maxBackoff.
func (a *agent) createProfile(ctx context.Context) *pb.Profile {
	req := pb.CreateProfileRequest{
		Parent:      "projects/" + a.deployment.ProjectId,
		Deployment:  a.deployment,
		ProfileType: a.profileTypes,
	}

	var p *pb.Profile
	md := grpcmd.New(map[string]string{})

	gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		debugLog("creating a new profile via profiler service")
		var err error
		p, err = a.client.CreateProfile(ctx, &req, grpc.Trailer(&md))
		if err != nil {
			debugLog("failed to create profile, will retry: %v", err)
		}
		return err
	}, gax.WithRetry(func() gax.Retryer {
		return &retryer{
			backoff: gax.Backoff{
				Initial:    initialBackoff,
				Max:        maxBackoff,
				Multiplier: backoffMultiplier,
			},
			md: md,
		}
	}))

	debugLog("successfully created profile %v", p.GetProfileType())
	return p
}

func (a *agent) profileAndUpload(ctx context.Context, p *pb.Profile) {
	var prof bytes.Buffer
	pt := p.GetProfileType()

	switch pt {
	case pb.ProfileType_CPU:
		duration, err := ptypes.Duration(p.Duration)
		if err != nil {
			debugLog("failed to get profile duration for CPU profile: %v", err)
			return
		}
		if err := startCPUProfile(&prof); err != nil {
			debugLog("failed to start CPU profile: %v", err)
			return
		}
		sleep(ctx, duration)
		stopCPUProfile()
	case pb.ProfileType_HEAP:
		if err := heapProfile(&prof); err != nil {
			debugLog("failed to write heap profile: %v", err)
			return
		}
	case pb.ProfileType_HEAP_ALLOC:
		duration, err := ptypes.Duration(p.Duration)
		if err != nil {
			debugLog("failed to get profile duration for allocation profile: %v", err)
			return
		}
		if err := deltaAllocProfile(ctx, duration, config.AllocForceGC, &prof); err != nil {
			debugLog("failed to collect allocation profile: %v", err)
			return
		}
	case pb.ProfileType_THREADS:
		if err := pprof.Lookup("goroutine").WriteTo(&prof, 0); err != nil {
			debugLog("failed to collect goroutine profile: %v", err)
			return
		}
	case pb.ProfileType_CONTENTION:
		duration, err := ptypes.Duration(p.Duration)
		if err != nil {
			debugLog("failed to get profile duration: %v", err)
			return
		}
		if err := deltaMutexProfile(ctx, duration, &prof); err != nil {
			debugLog("failed to collect mutex profile: %v", err)
			return
		}
	default:
		debugLog("unexpected profile type: %v", pt)
		return
	}

	p.ProfileBytes = prof.Bytes()
	p.Labels = a.profileLabels
	req := pb.UpdateProfileRequest{Profile: p}

	// Upload profile, discard profile in case of error.
	debugLog("start uploading profile")
	if _, err := a.client.UpdateProfile(ctx, &req); err != nil {
		debugLog("failed to upload profile: %v", err)
	}
}

// deltaMutexProfile writes mutex profile changes over a time period specified
// with 'duration' to 'prof'.
func deltaMutexProfile(ctx context.Context, duration time.Duration, prof *bytes.Buffer) error {
	if !mutexEnabled {
		return errors.New("mutex profiling is not enabled")
	}
	p0, err := mutexProfile()
	if err != nil {
		return err
	}
	sleep(ctx, duration)
	p, err := mutexProfile()
	if err != nil {
		return err
	}

	p0.Scale(-1)
	p, err = profile.Merge([]*profile.Profile{p0, p})
	if err != nil {
		return err
	}

	return p.Write(prof)
}

func mutexProfile() (*profile.Profile, error) {
	p := pprof.Lookup("mutex")
	if p == nil {
		return nil, errors.New("mutex profiling is not supported")
	}
	var buf bytes.Buffer
	if err := p.WriteTo(&buf, 0); err != nil {
		return nil, err
	}
	return profile.Parse(&buf)
}

// withXGoogHeader sets the name and version of the application in
// the `x-goog-api-client` header passed on each request. Intended for
// use by Google-written clients.
func withXGoogHeader(ctx context.Context, keyval ...string) context.Context {
	kv := append([]string{"gl-go", version.Go(), "gccl", version.Repo}, keyval...)
	kv = append(kv, "gax", gax.Version, "grpc", grpc.Version)

	md, _ := grpcmd.FromOutgoingContext(ctx)
	md = md.Copy()
	md[xGoogAPIMetadata] = []string{gax.XGoogHeader(kv...)}
	return grpcmd.NewOutgoingContext(ctx, md)
}

// initializeAgent initializes the profiling agent. It returns an error if
// profile collection should not be started because collection is disabled
// for all profile types.
func initializeAgent(c pb.ProfilerServiceClient) (*agent, error) {
	labels := map[string]string{languageLabel: "go"}
	if config.Zone != "" {
		labels[zoneNameLabel] = config.Zone
	}
	if config.ServiceVersion != "" {
		labels[versionLabel] = config.ServiceVersion
	}
	d := &pb.Deployment{
		ProjectId: config.ProjectID,
		Target:    config.Service,
		Labels:    labels,
	}

	profileLabels := map[string]string{}

	if config.Instance != "" {
		profileLabels[instanceLabel] = config.Instance
	}

	var profileTypes []pb.ProfileType
	if !config.NoCPUProfiling {
		profileTypes = append(profileTypes, pb.ProfileType_CPU)
	}
	if !config.NoHeapProfiling {
		profileTypes = append(profileTypes, pb.ProfileType_HEAP)
	}
	if !config.NoGoroutineProfiling {
		profileTypes = append(profileTypes, pb.ProfileType_THREADS)
	}
	if !config.NoAllocProfiling {
		profileTypes = append(profileTypes, pb.ProfileType_HEAP_ALLOC)
	}
	if mutexEnabled {
		profileTypes = append(profileTypes, pb.ProfileType_CONTENTION)
	}

	if len(profileTypes) == 0 {
		return nil, fmt.Errorf("collection is not enabled for any profile types")
	}

	return &agent{
		client:        c,
		deployment:    d,
		profileLabels: profileLabels,
		profileTypes:  profileTypes,
	}, nil
}

func initializeConfig(cfg Config) error {
	config = cfg

	if config.Service == "" {
		for _, ev := range []string{"GAE_SERVICE", "K_SERVICE"} {
			if val := os.Getenv(ev); val != "" {
				config.Service = val
				break
			}
		}
	}
	if config.Service == "" {
		return errors.New("service name must be configured")
	}
	if !serviceRegexp.MatchString(config.Service) {
		return fmt.Errorf("service name %q does not match regular expression %v", config.Service, serviceRegexp)
	}

	if config.ServiceVersion == "" {
		for _, ev := range []string{"GAE_VERSION", "K_REVISION"} {
			if val := os.Getenv(ev); val != "" {
				config.ServiceVersion = val
				break
			}
		}
	}

	if projectID := os.Getenv("GOOGLE_CLOUD_PROJECT"); config.ProjectID == "" && projectID != "" {
		// Cloud Shell and App Engine set this environment variable to the project
		// ID, so use it if present. In case of App Engine the project ID is also
		// available from the GCE metadata server, but by using the environment
		// variable saves one request to the metadata server. The environment
		// project ID is only used if no project ID is provided in the
		// configuration.
		config.ProjectID = projectID
	}
	if onGCE() {
		var err error
		if config.ProjectID == "" {
			if config.ProjectID, err = getProjectID(); err != nil {
				return fmt.Errorf("failed to get the project ID from Compute Engine metadata: %v", err)
			}
		}

		if config.Zone == "" {
			if config.Zone, err = getZone(); err != nil {
				return fmt.Errorf("failed to get zone from Compute Engine metadata: %v", err)
			}
		}

		if config.Instance == "" {
			if config.Instance, err = getInstanceName(); err != nil {
				if _, ok := err.(gcemd.NotDefinedError); !ok {
					return fmt.Errorf("failed to get instance name from Compute Engine metadata: %v", err)
				}
				debugLog("failed to get instance name from Compute Engine metadata, will use empty name: %v", err)
			}
		}
	} else {
		if config.ProjectID == "" {
			return fmt.Errorf("project ID must be specified in the configuration if running outside of GCP")
		}
	}

	if config.APIAddr == "" {
		config.APIAddr = apiAddress
	}
	return nil
}

// pollProfilerService starts an endless loop to poll the profiler
// server for instructions, and collects and uploads profiles as
// requested.
func pollProfilerService(ctx context.Context, a *agent) {
	debugLog("Stackdriver Profiler Go Agent version: %s", version.Repo)
	debugLog("profiler has started")
	for i := 0; config.numProfiles == 0 || i < config.numProfiles; i++ {
		p := a.createProfile(ctx)
		a.profileAndUpload(ctx, p)
	}

	if profilingDone != nil {
		profilingDone <- true
	}
}
