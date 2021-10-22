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

package profiler

import (
	"bytes"
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	gcemd "cloud.google.com/go/compute/metadata"
	"cloud.google.com/go/internal/testutil"
	"cloud.google.com/go/profiler/mocks"
	"cloud.google.com/go/profiler/testdata"
	"github.com/golang/mock/gomock"
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

const (
	testProjectID                = "test-project-ID"
	testInstance                 = "test-instance"
	testZone                     = "test-zone"
	testService                  = "test-service"
	testSvcVersion               = "test-service-version"
	testProfileDuration          = time.Second * 10
	testProfileCollectionTimeout = time.Second * 15
)

func createTestDeployment() *pb.Deployment {
	labels := map[string]string{
		zoneNameLabel: testZone,
		versionLabel:  testSvcVersion,
	}
	return &pb.Deployment{
		ProjectId: testProjectID,
		Target:    testService,
		Labels:    labels,
	}
}

func createTestAgent(psc pb.ProfilerServiceClient) *agent {
	return &agent{
		client:        psc,
		deployment:    createTestDeployment(),
		profileLabels: map[string]string{instanceLabel: testInstance},
		profileTypes:  []pb.ProfileType{pb.ProfileType_CPU, pb.ProfileType_HEAP, pb.ProfileType_THREADS},
	}
}

func createTrailers(dur time.Duration) map[string]string {
	b, _ := proto.Marshal(&edpb.RetryInfo{
		RetryDelay: ptypes.DurationProto(dur),
	})
	return map[string]string{
		retryInfoMetadata: string(b),
	}
}

func TestCreateProfile(t *testing.T) {
	ctx := context.Background()
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	mpc := mocks.NewMockProfilerServiceClient(ctrl)
	a := createTestAgent(mpc)
	p := &pb.Profile{Name: "test_profile"}
	wantRequest := pb.CreateProfileRequest{
		Parent:      "projects/" + a.deployment.ProjectId,
		Deployment:  a.deployment,
		ProfileType: a.profileTypes,
	}

	mpc.EXPECT().CreateProfile(ctx, gomock.Eq(&wantRequest), gomock.Any()).Times(1).Return(p, nil)

	gotP := a.createProfile(ctx)

	if !testutil.Equal(gotP, p) {
		t.Errorf("CreateProfile() got wrong profile, got %v, want %v", gotP, p)
	}
}

func TestProfileAndUpload(t *testing.T) {
	oldStartCPUProfile, oldStopCPUProfile, oldWriteHeapProfile, oldSleep := startCPUProfile, stopCPUProfile, writeHeapProfile, sleep
	defer func() {
		startCPUProfile, stopCPUProfile, writeHeapProfile, sleep = oldStartCPUProfile, oldStopCPUProfile, oldWriteHeapProfile, oldSleep
	}()

	ctx := context.Background()
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	var heapCollected1, heapCollected2, heapUploaded, allocUploaded bytes.Buffer
	testdata.HeapProfileCollected1.Write(&heapCollected1)
	testdata.HeapProfileCollected2.Write(&heapCollected2)
	testdata.HeapProfileUploaded.Write(&heapUploaded)
	testdata.AllocProfileUploaded.Write(&allocUploaded)
	callCount := 0
	writeTwoHeapFunc := func(w io.Writer) error {
		callCount++
		if callCount%2 == 1 {
			w.Write(heapCollected1.Bytes())
			return nil
		}
		w.Write(heapCollected2.Bytes())
		return nil
	}

	errFunc := func(io.Writer) error { return errors.New("") }
	testDuration := time.Second * 5
	tests := []struct {
		profileType          pb.ProfileType
		duration             *time.Duration
		startCPUProfileFunc  func(io.Writer) error
		writeHeapProfileFunc func(io.Writer) error
		wantBytes            []byte
	}{
		{
			profileType: pb.ProfileType_CPU,
			duration:    &testDuration,
			startCPUProfileFunc: func(w io.Writer) error {
				w.Write([]byte{1})
				return nil
			},
			writeHeapProfileFunc: errFunc,
			wantBytes:            []byte{1},
		},
		{
			profileType:          pb.ProfileType_CPU,
			startCPUProfileFunc:  errFunc,
			writeHeapProfileFunc: errFunc,
		},
		{
			profileType: pb.ProfileType_CPU,
			duration:    &testDuration,
			startCPUProfileFunc: func(w io.Writer) error {
				w.Write([]byte{2})
				return nil
			},
			writeHeapProfileFunc: func(w io.Writer) error {
				w.Write([]byte{3})
				return nil
			},
			wantBytes: []byte{2},
		},
		{
			profileType:         pb.ProfileType_HEAP,
			startCPUProfileFunc: errFunc,
			writeHeapProfileFunc: func(w io.Writer) error {
				w.Write(heapCollected1.Bytes())
				return nil
			},
			wantBytes: heapUploaded.Bytes(),
		},
		{
			profileType:          pb.ProfileType_HEAP_ALLOC,
			startCPUProfileFunc:  errFunc,
			writeHeapProfileFunc: writeTwoHeapFunc,
			duration:             &testDuration,
			wantBytes:            allocUploaded.Bytes(),
		},
		{
			profileType:          pb.ProfileType_HEAP,
			startCPUProfileFunc:  errFunc,
			writeHeapProfileFunc: errFunc,
		},
		{
			profileType: pb.ProfileType_HEAP,
			startCPUProfileFunc: func(w io.Writer) error {
				w.Write([]byte{5})
				return nil
			},
			writeHeapProfileFunc: func(w io.Writer) error {
				w.Write(heapCollected1.Bytes())
				return nil
			},
			wantBytes: heapUploaded.Bytes(),
		},
		{
			profileType: pb.ProfileType_PROFILE_TYPE_UNSPECIFIED,
			startCPUProfileFunc: func(w io.Writer) error {
				w.Write([]byte{7})
				return nil
			},
			writeHeapProfileFunc: func(w io.Writer) error {
				w.Write(heapCollected1.Bytes())
				return nil
			},
		},
	}

	for _, tt := range tests {
		mpc := mocks.NewMockProfilerServiceClient(ctrl)
		a := createTestAgent(mpc)
		startCPUProfile = tt.startCPUProfileFunc
		stopCPUProfile = func() {}
		writeHeapProfile = tt.writeHeapProfileFunc
		var gotSleep *time.Duration
		sleep = func(ctx context.Context, d time.Duration) error {
			gotSleep = &d
			return nil
		}
		p := &pb.Profile{ProfileType: tt.profileType}
		if tt.duration != nil {
			p.Duration = ptypes.DurationProto(*tt.duration)
		}
		if tt.wantBytes != nil {
			wantProfile := &pb.Profile{
				ProfileType:  p.ProfileType,
				Duration:     p.Duration,
				ProfileBytes: tt.wantBytes,
				Labels:       a.profileLabels,
			}
			wantRequest := pb.UpdateProfileRequest{
				Profile: wantProfile,
			}
			mpc.EXPECT().UpdateProfile(ctx, gomock.Eq(&wantRequest)).Times(1)
		} else {
			mpc.EXPECT().UpdateProfile(gomock.Any(), gomock.Any()).MaxTimes(0)
		}

		a.profileAndUpload(ctx, p)

		if tt.duration == nil {
			if gotSleep != nil {
				t.Errorf("profileAndUpload(%v) slept for: %v, want no sleep", p, gotSleep)
			}
		} else {
			if gotSleep == nil {
				t.Errorf("profileAndUpload(%v) didn't sleep, want sleep for: %v", p, tt.duration)
			} else if *gotSleep != *tt.duration {
				t.Errorf("profileAndUpload(%v) slept for wrong duration, got: %v, want: %v", p, gotSleep, tt.duration)
			}
		}
	}
}

func TestRetry(t *testing.T) {
	normalDuration := time.Second * 3
	negativeDuration := time.Second * -3

	tests := []struct {
		trailers  map[string]string
		wantPause *time.Duration
	}{
		{
			createTrailers(normalDuration),
			&normalDuration,
		},
		{
			createTrailers(negativeDuration),
			nil,
		},
		{
			map[string]string{retryInfoMetadata: "wrong format"},
			nil,
		},
		{
			map[string]string{},
			nil,
		},
	}

	for _, tt := range tests {
		md := grpcmd.New(tt.trailers)
		r := &retryer{
			backoff: gax.Backoff{
				Initial:    initialBackoff,
				Max:        maxBackoff,
				Multiplier: backoffMultiplier,
			},
			md: md,
		}

		pause, shouldRetry := r.Retry(status.Error(codes.Aborted, ""))

		if !shouldRetry {
			t.Error("retryer.Retry() returned shouldRetry false, want true")
		}

		if tt.wantPause != nil {
			if pause != *tt.wantPause {
				t.Errorf("retryer.Retry() returned wrong pause, got: %v, want: %v", pause, tt.wantPause)
			}
		} else {
			if pause > initialBackoff {
				t.Errorf("retryer.Retry() returned wrong pause, got: %v, want: < %v", pause, initialBackoff)
			}
		}
	}

	md := grpcmd.New(map[string]string{})

	r := &retryer{
		backoff: gax.Backoff{
			Initial:    initialBackoff,
			Max:        maxBackoff,
			Multiplier: backoffMultiplier,
		},
		md: md,
	}
	for i := 0; i < 100; i++ {
		pause, shouldRetry := r.Retry(errors.New(""))
		if !shouldRetry {
			t.Errorf("retryer.Retry() called %v times, returned shouldRetry false, want true", i)
		}
		if pause > maxBackoff {
			t.Errorf("retryer.Retry() called %v times, returned wrong pause, got: %v, want: < %v", i, pause, maxBackoff)
		}
	}
}

func TestWithXGoogHeader(t *testing.T) {
	ctx := withXGoogHeader(context.Background())
	md, _ := grpcmd.FromOutgoingContext(ctx)

	if xg := md[xGoogAPIMetadata]; len(xg) == 0 {
		t.Errorf("withXGoogHeader() sets empty xGoogHeader")
	} else {
		if !strings.Contains(xg[0], "gl-go/") {
			t.Errorf("withXGoogHeader() got: %v, want gl-go key", xg[0])
		}
		if !strings.Contains(xg[0], "gccl/") {
			t.Errorf("withXGoogHeader() got: %v, want gccl key", xg[0])
		}
		if !strings.Contains(xg[0], "gax/") {
			t.Errorf("withXGoogHeader() got: %v, want gax key", xg[0])
		}
		if !strings.Contains(xg[0], "grpc/") {
			t.Errorf("withXGoogHeader() got: %v, want grpc key", xg[0])
		}
	}
}

func TestInitializeAgent(t *testing.T) {
	oldConfig, oldMutexEnabled := config, mutexEnabled
	defer func() {
		config, mutexEnabled = oldConfig, oldMutexEnabled
	}()

	for _, tt := range []struct {
		config               Config
		enableMutex          bool
		wantErr              bool
		wantProfileTypes     []pb.ProfileType
		wantDeploymentLabels map[string]string
		wantProfileLabels    map[string]string
	}{
		{
			config:               Config{ServiceVersion: testSvcVersion, Zone: testZone},
			wantProfileTypes:     []pb.ProfileType{pb.ProfileType_CPU, pb.ProfileType_HEAP, pb.ProfileType_THREADS, pb.ProfileType_HEAP_ALLOC},
			wantDeploymentLabels: map[string]string{zoneNameLabel: testZone, versionLabel: testSvcVersion, languageLabel: "go"},
			wantProfileLabels:    map[string]string{},
		},
		{
			config:               Config{Zone: testZone},
			wantProfileTypes:     []pb.ProfileType{pb.ProfileType_CPU, pb.ProfileType_HEAP, pb.ProfileType_THREADS, pb.ProfileType_HEAP_ALLOC},
			wantDeploymentLabels: map[string]string{zoneNameLabel: testZone, languageLabel: "go"},
			wantProfileLabels:    map[string]string{},
		},
		{
			config:               Config{ServiceVersion: testSvcVersion},
			wantProfileTypes:     []pb.ProfileType{pb.ProfileType_CPU, pb.ProfileType_HEAP, pb.ProfileType_THREADS, pb.ProfileType_HEAP_ALLOC},
			wantDeploymentLabels: map[string]string{versionLabel: testSvcVersion, languageLabel: "go"},
			wantProfileLabels:    map[string]string{},
		},
		{
			config:               Config{Instance: testInstance},
			wantProfileTypes:     []pb.ProfileType{pb.ProfileType_CPU, pb.ProfileType_HEAP, pb.ProfileType_THREADS, pb.ProfileType_HEAP_ALLOC},
			wantDeploymentLabels: map[string]string{languageLabel: "go"},
			wantProfileLabels:    map[string]string{instanceLabel: testInstance},
		},
		{
			config:               Config{Instance: testInstance},
			enableMutex:          true,
			wantProfileTypes:     []pb.ProfileType{pb.ProfileType_CPU, pb.ProfileType_HEAP, pb.ProfileType_THREADS, pb.ProfileType_HEAP_ALLOC, pb.ProfileType_CONTENTION},
			wantDeploymentLabels: map[string]string{languageLabel: "go"},
			wantProfileLabels:    map[string]string{instanceLabel: testInstance},
		},
		{
			config:               Config{NoHeapProfiling: true},
			wantProfileTypes:     []pb.ProfileType{pb.ProfileType_CPU, pb.ProfileType_THREADS, pb.ProfileType_HEAP_ALLOC},
			wantDeploymentLabels: map[string]string{languageLabel: "go"},
			wantProfileLabels:    map[string]string{},
		},
		{
			config:               Config{NoHeapProfiling: true, NoGoroutineProfiling: true, NoAllocProfiling: true},
			wantProfileTypes:     []pb.ProfileType{pb.ProfileType_CPU},
			wantDeploymentLabels: map[string]string{languageLabel: "go"},
			wantProfileLabels:    map[string]string{},
		},
		{
			config:               Config{NoCPUProfiling: true},
			wantProfileTypes:     []pb.ProfileType{pb.ProfileType_HEAP, pb.ProfileType_THREADS, pb.ProfileType_HEAP_ALLOC},
			wantDeploymentLabels: map[string]string{languageLabel: "go"},
			wantProfileLabels:    map[string]string{},
		},
		{
			config:  Config{NoCPUProfiling: true, NoHeapProfiling: true, NoGoroutineProfiling: true, NoAllocProfiling: true},
			wantErr: true,
		},
	} {

		config = tt.config
		config.ProjectID = testProjectID
		config.Service = testService
		mutexEnabled = tt.enableMutex
		a, err := initializeAgent(nil)
		if err != nil {
			if !tt.wantErr {
				t.Fatalf("initializeAgent() got error: %v, want no error", err)
			}
			continue
		}

		wantDeployment := &pb.Deployment{
			ProjectId: testProjectID,
			Target:    testService,
			Labels:    tt.wantDeploymentLabels,
		}
		if !testutil.Equal(a.deployment, wantDeployment) {
			t.Errorf("initializeAgent() got deployment: %v, want %v", a.deployment, wantDeployment)
		}
		if !testutil.Equal(a.profileLabels, tt.wantProfileLabels) {
			t.Errorf("initializeAgent() got profile labels: %v, want %v", a.profileLabels, tt.wantProfileLabels)
		}
		if !testutil.Equal(a.profileTypes, tt.wantProfileTypes) {
			t.Errorf("initializeAgent() got profile types: %v, want %v", a.profileTypes, tt.wantProfileTypes)
		}
	}
}

func TestInitializeConfig(t *testing.T) {
	oldConfig, oldGAEService, oldGAEVersion, oldKnativeService, oldKnativeVersion, oldEnvProjectID, oldGetProjectID, oldGetInstanceName, oldGetZone, oldOnGCE := config, os.Getenv("GAE_SERVICE"), os.Getenv("GAE_VERSION"), os.Getenv("K_SERVICE"), os.Getenv("K_REVISION"), os.Getenv("GOOGLE_CLOUD_PROJECT"), getProjectID, getInstanceName, getZone, onGCE
	defer func() {
		config, getProjectID, getInstanceName, getZone, onGCE = oldConfig, oldGetProjectID, oldGetInstanceName, oldGetZone, oldOnGCE
		if err := os.Setenv("GAE_SERVICE", oldGAEService); err != nil {
			t.Fatal(err)
		}
		if err := os.Setenv("GAE_VERSION", oldGAEVersion); err != nil {
			t.Fatal(err)
		}
		if err := os.Setenv("K_SERVICE", oldKnativeService); err != nil {
			t.Fatal(err)
		}
		if err := os.Setenv("K_REVISION", oldKnativeVersion); err != nil {
			t.Fatal(err)
		}
		if err := os.Setenv("GOOGLE_CLOUD_PROJECT", oldEnvProjectID); err != nil {
			t.Fatal(err)
		}
	}()
	const (
		testGAEService     = "test-gae-service"
		testGAEVersion     = "test-gae-version"
		testKnativeService = "test-knative-service"
		testKnativeVersion = "test-knative-version"
		testGCEProjectID   = "test-gce-project-id"
		testEnvProjectID   = "test-env-project-id"
	)
	for _, tt := range []struct {
		desc            string
		config          Config
		wantConfig      Config
		wantErrorString string
		onGAE           bool
		onKnative       bool
		onGCE           bool
		envProjectID    bool
	}{
		{
			"accepts service name",
			Config{Service: testService},
			Config{Service: testService, ProjectID: testGCEProjectID, Zone: testZone, Instance: testInstance},
			"",
			false,
			false,
			true,
			false,
		},
		{
			"env project overrides GCE project",
			Config{Service: testService},
			Config{Service: testService, ProjectID: testEnvProjectID, Zone: testZone, Instance: testInstance},
			"",
			false,
			false,
			true,
			true,
		},
		{
			"requires service name",
			Config{},
			Config{},
			"service name must be configured",
			false,
			false,
			true,
			false,
		},
		{
			"requires valid service name",
			Config{Service: "Service"},
			Config{Service: "Service"},
			"service name \"Service\" does not match regular expression ^[a-z]([-a-z0-9_.]{0,253}[a-z0-9])?$",
			false,
			false,
			true,
			false,
		},
		{
			"accepts service name from config and service version from GAE",
			Config{Service: testService},
			Config{Service: testService, ServiceVersion: testGAEVersion, ProjectID: testGCEProjectID, Zone: testZone, Instance: testInstance},
			"",
			true,
			false,
			true,
			false,
		},
		{
			"reads both service name and version from GAE env vars",
			Config{},
			Config{Service: testGAEService, ServiceVersion: testGAEVersion, ProjectID: testGCEProjectID, Zone: testZone, Instance: testInstance},
			"",
			true,
			false,
			true,
			false,
		},
		{
			"reads both service name and version from Knative env vars",
			Config{},
			Config{Service: testKnativeService, ServiceVersion: testKnativeVersion, ProjectID: testGCEProjectID, Zone: testZone, Instance: testInstance},
			"",
			false,
			true,
			true,
			false,
		},
		{
			"accepts service version from config",
			Config{Service: testService, ServiceVersion: testSvcVersion},
			Config{Service: testService, ServiceVersion: testSvcVersion, ProjectID: testGCEProjectID, Zone: testZone, Instance: testInstance},
			"",
			false,
			false,
			true,
			false,
		},
		{
			"configured version has priority over GAE-provided version",
			Config{Service: testService, ServiceVersion: testSvcVersion},
			Config{Service: testService, ServiceVersion: testSvcVersion, ProjectID: testGCEProjectID, Zone: testZone, Instance: testInstance},
			"",
			true,
			false,
			true,
			false,
		},
		{
			"configured version has priority over Knative-provided version",
			Config{Service: testService, ServiceVersion: testSvcVersion},
			Config{Service: testService, ServiceVersion: testSvcVersion, ProjectID: testGCEProjectID, Zone: testZone, Instance: testInstance},
			"",
			false,
			true,
			true,
			false,
		},
		{
			"GAE version has priority over Knative-provided version",
			Config{},
			Config{Service: testGAEService, ServiceVersion: testGAEVersion, ProjectID: testGCEProjectID, Zone: testZone, Instance: testInstance},
			"",
			true,
			true,
			true,
			false,
		},
		{
			"configured project ID has priority over metadata-provided project ID",
			Config{Service: testService, ProjectID: testProjectID},
			Config{Service: testService, ProjectID: testProjectID, Zone: testZone, Instance: testInstance},
			"",
			false,
			false,
			true,
			false,
		},
		{
			"configured project ID has priority over environment project ID",
			Config{Service: testService, ProjectID: testProjectID},
			Config{Service: testService, ProjectID: testProjectID},
			"",
			false,
			false,
			false,
			true,
		},
		{
			"requires project ID if not on GCE",
			Config{Service: testService},
			Config{Service: testService},
			"project ID must be specified in the configuration if running outside of GCP",
			false,
			false,
			false,
			false,
		},
		{
			"configured zone has priority over metadata-provided zone",
			Config{Service: testService, ProjectID: testProjectID, Zone: testZone + "-override"},
			Config{Service: testService, ProjectID: testProjectID, Zone: testZone + "-override", Instance: testInstance},
			"",
			false,
			false,
			true,
			false,
		},
		{
			"configured instance has priority over metadata-provided instance",
			Config{Service: testService, ProjectID: testProjectID, Instance: testInstance + "-override"},
			Config{Service: testService, ProjectID: testProjectID, Zone: testZone, Instance: testInstance + "-override"},
			"",
			false,
			false,
			true,
			false,
		},
	} {
		t.Logf("Running test: %s", tt.desc)
		gaeEnvService, gaeEnvVersion := "", ""
		if tt.onGAE {
			gaeEnvService, gaeEnvVersion = testGAEService, testGAEVersion
		}
		if err := os.Setenv("GAE_SERVICE", gaeEnvService); err != nil {
			t.Fatal(err)
		}
		if err := os.Setenv("GAE_VERSION", gaeEnvVersion); err != nil {
			t.Fatal(err)
		}
		knEnvService, knEnvVersion := "", ""
		if tt.onKnative {
			knEnvService, knEnvVersion = testKnativeService, testKnativeVersion
		}
		if err := os.Setenv("K_SERVICE", knEnvService); err != nil {
			t.Fatal(err)
		}
		if err := os.Setenv("K_REVISION", knEnvVersion); err != nil {
			t.Fatal(err)
		}
		if tt.onGCE {
			onGCE = func() bool { return true }
			getProjectID = func() (string, error) { return testGCEProjectID, nil }
			getZone = func() (string, error) { return testZone, nil }
			getInstanceName = func() (string, error) { return testInstance, nil }
		} else {
			onGCE = func() bool { return false }
			getProjectID = func() (string, error) { return "", fmt.Errorf("test get project id error") }
			getZone = func() (string, error) { return "", fmt.Errorf("test get zone error") }
			getInstanceName = func() (string, error) { return "", fmt.Errorf("test get instance error") }
		}
		envProjectID := ""
		if tt.envProjectID {
			envProjectID = testEnvProjectID
		}
		if err := os.Setenv("GOOGLE_CLOUD_PROJECT", envProjectID); err != nil {
			t.Fatal(err)
		}

		errorString := ""
		if err := initializeConfig(tt.config); err != nil {
			errorString = err.Error()
		}

		if !strings.Contains(errorString, tt.wantErrorString) {
			t.Errorf("initializeConfig(%v) got error: %v, want contain %v", tt.config, errorString, tt.wantErrorString)
		}
		if tt.wantErrorString == "" {
			tt.wantConfig.APIAddr = apiAddress
		}
		if config != tt.wantConfig {
			t.Errorf("initializeConfig(%v) got: %v, want %v", tt.config, config, tt.wantConfig)
		}
	}

	for _, tt := range []struct {
		desc              string
		wantErr           bool
		getProjectIDError error
		getZoneError      error
		getInstanceError  error
	}{
		{
			desc:              "metadata returns error for project ID",
			wantErr:           true,
			getProjectIDError: errors.New("fake get project ID error"),
		},
		{
			desc:         "metadata returns error for zone",
			wantErr:      true,
			getZoneError: errors.New("fake get zone error"),
		},
		{
			desc:             "metadata returns error for instance",
			wantErr:          true,
			getInstanceError: errors.New("fake get instance error"),
		},
		{
			desc:             "metadata returns NotDefinedError for instance",
			getInstanceError: gcemd.NotDefinedError("fake GCE metadata NotDefinedError error"),
		},
	} {
		onGCE = func() bool { return true }
		getProjectID = func() (string, error) { return testGCEProjectID, tt.getProjectIDError }
		getZone = func() (string, error) { return testZone, tt.getZoneError }
		getInstanceName = func() (string, error) { return testInstance, tt.getInstanceError }

		if err := initializeConfig(Config{Service: testService}); (err != nil) != tt.wantErr {
			t.Errorf("%s: initializeConfig() got error: %v, want error %t", tt.desc, err, tt.wantErr)
		}
	}
}

type fakeProfilerServer struct {
	count       int
	gotProfiles map[string][]byte
}

func (fs *fakeProfilerServer) CreateProfile(ctx context.Context, in *pb.CreateProfileRequest) (*pb.Profile, error) {
	fs.count++
	switch fs.count % 2 {
	case 1:
		return &pb.Profile{Name: "testCPU", ProfileType: pb.ProfileType_CPU, Duration: ptypes.DurationProto(testProfileDuration)}, nil
	default:
		return &pb.Profile{Name: "testHeap", ProfileType: pb.ProfileType_HEAP}, nil
	}
}

func (fs *fakeProfilerServer) UpdateProfile(ctx context.Context, in *pb.UpdateProfileRequest) (*pb.Profile, error) {
	switch in.Profile.ProfileType {
	case pb.ProfileType_CPU:
		fs.gotProfiles["CPU"] = in.Profile.ProfileBytes
	case pb.ProfileType_HEAP:
		fs.gotProfiles["HEAP"] = in.Profile.ProfileBytes
	}
	return in.Profile, nil
}

func (fs *fakeProfilerServer) CreateOfflineProfile(_ context.Context, _ *pb.CreateOfflineProfileRequest) (*pb.Profile, error) {
	return nil, status.Error(codes.Unimplemented, "")
}

func profileeLoop(quit chan bool) {
	for {
		select {
		case <-quit:
			return
		default:
			profileeWork()
		}
	}
}

func profileeWork() {
	data := make([]byte, 10*1024*1024)
	rand.Read(data)

	var b bytes.Buffer
	gz := gzip.NewWriter(&b)
	if _, err := gz.Write(data); err != nil {
		log.Println("failed to write to gzip stream", err)
		return
	}
	if err := gz.Flush(); err != nil {
		log.Println("failed to flush to gzip stream", err)
		return
	}
	if err := gz.Close(); err != nil {
		log.Println("failed to close gzip stream", err)
	}
}

func validateProfile(rawData []byte, wantFunctionName string) error {
	p, err := profile.ParseData(rawData)
	if err != nil {
		return fmt.Errorf("ParseData failed: %v", err)
	}

	if len(p.Sample) == 0 {
		return fmt.Errorf("profile contains zero samples: %v", p)
	}

	if len(p.Location) == 0 {
		return fmt.Errorf("profile contains zero locations: %v", p)
	}

	if len(p.Function) == 0 {
		return fmt.Errorf("profile contains zero functions: %v", p)
	}

	for _, l := range p.Location {
		if len(l.Line) > 0 && l.Line[0].Function != nil && strings.Contains(l.Line[0].Function.Name, wantFunctionName) {
			return nil
		}
	}
	return fmt.Errorf("wanted function name %s not found in the profile", wantFunctionName)
}

func TestDeltaMutexProfile(t *testing.T) {
	oldMutexEnabled, oldMaxProcs := mutexEnabled, runtime.GOMAXPROCS(10)
	defer func() {
		mutexEnabled = oldMutexEnabled
		runtime.GOMAXPROCS(oldMaxProcs)
	}()
	if mutexEnabled = enableMutexProfiling(); !mutexEnabled {
		t.Skip("Go too old - mutex profiling not supported.")
	}

	hog(time.Second, mutexHog)
	go func() {
		hog(2*time.Second, backgroundHog)
	}()

	var prof bytes.Buffer
	if err := deltaMutexProfile(context.Background(), time.Second, &prof); err != nil {
		t.Fatalf("deltaMutexProfile() got error: %v", err)
	}
	p, err := profile.Parse(&prof)
	if err != nil {
		t.Fatalf("profile.Parse() got error: %v", err)
	}

	if s := sum(p, "mutexHog"); s != 0 {
		t.Errorf("mutexHog found in the delta mutex profile (sum=%d):\n%s", s, p)
	}
	if s := sum(p, "backgroundHog"); s <= 0 {
		t.Errorf("backgroundHog not in the delta mutex profile (sum=%d):\n%s", s, p)
	}
}

// sum returns the sum of all mutex counts from the samples whose
// stacks include the specified function name.
func sum(p *profile.Profile, fname string) int64 {
	locIDs := map[*profile.Location]bool{}
	for _, loc := range p.Location {
		for _, l := range loc.Line {
			if strings.Contains(l.Function.Name, fname) {
				locIDs[loc] = true
				break
			}
		}
	}
	var s int64
	for _, sample := range p.Sample {
		for _, loc := range sample.Location {
			if locIDs[loc] {
				s += sample.Value[0]
				break
			}
		}
	}
	return s
}

func mutexHog(mu1, mu2 *sync.Mutex, start time.Time, dt time.Duration) {
	for time.Since(start) < dt {
		mu1.Lock()
		runtime.Gosched()
		mu2.Lock()
		mu1.Unlock()
		mu2.Unlock()
	}
}

// backgroundHog is identical to mutexHog. We keep them separate
// in order to distinguish them with function names in the stack trace.
func backgroundHog(mu1, mu2 *sync.Mutex, start time.Time, dt time.Duration) {
	for time.Since(start) < dt {
		mu1.Lock()
		runtime.Gosched()
		mu2.Lock()
		mu1.Unlock()
		mu2.Unlock()
	}
}

func hog(dt time.Duration, hogger func(mu1, mu2 *sync.Mutex, start time.Time, dt time.Duration)) {
	start := time.Now()
	mu1 := new(sync.Mutex)
	mu2 := new(sync.Mutex)
	var wg sync.WaitGroup
	wg.Add(10)
	for i := 0; i < 10; i++ {
		go func() {
			defer wg.Done()
			hogger(mu1, mu2, start, dt)
		}()
	}
	wg.Wait()
}

func TestAgentWithServer(t *testing.T) {
	oldDialGRPC, oldConfig, oldProfilingDone := dialGRPC, config, profilingDone
	defer func() {
		dialGRPC, config, profilingDone = oldDialGRPC, oldConfig, oldProfilingDone
	}()

	profilingDone = make(chan bool)

	srv, err := testutil.NewServer()
	if err != nil {
		t.Fatalf("testutil.NewServer(): %v", err)
	}
	fakeServer := &fakeProfilerServer{gotProfiles: map[string][]byte{}}
	pb.RegisterProfilerServiceServer(srv.Gsrv, fakeServer)
	srv.Start()

	dialGRPC = func(ctx context.Context, opts ...option.ClientOption) (gtransport.ConnPool, error) {
		conn, err := gtransport.DialInsecure(ctx, opts...)
		if err != nil {
			return nil, err
		}
		return testConnPool{conn}, nil
	}
	if err := Start(Config{
		Service:     testService,
		ProjectID:   testProjectID,
		APIAddr:     srv.Addr,
		Instance:    testInstance,
		Zone:        testZone,
		numProfiles: 2,
	}); err != nil {
		t.Fatalf("Start(): %v", err)
	}

	quitProfilee := make(chan bool)
	go profileeLoop(quitProfilee)

	select {
	case <-profilingDone:
	case <-time.After(testProfileCollectionTimeout):
		t.Errorf("got timeout after %v, want profile collection done", testProfileCollectionTimeout)
	}
	quitProfilee <- true

	for _, pType := range []string{"CPU", "HEAP"} {
		if profile, ok := fakeServer.gotProfiles[pType]; !ok {
			t.Errorf("fakeServer.gotProfiles[%s] got no profile, want profile", pType)
		} else if err := validateProfile(profile, "profilee"); err != nil {
			t.Errorf("validateProfile(%s) got error: %v", pType, err)
		}
	}
}

// testConnPool is a gtransport.ConnPool used for testing.
type testConnPool struct{ *grpc.ClientConn }

func (p testConnPool) Num() int               { return 1 }
func (p testConnPool) Conn() *grpc.ClientConn { return p.ClientConn }
