package framework

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/spf13/pflag"

	apiextensionsapiserver "k8s.io/apiextensions-apiserver/pkg/apiserver"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	genericapiserver "k8s.io/apiserver/pkg/server"
	genericapiserveroptions "k8s.io/apiserver/pkg/server/options"
	"k8s.io/apiserver/pkg/util/compatibility"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	client "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/cert"
	basecompatibility "k8s.io/component-base/compatibility"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	aggregatorscheme "k8s.io/kube-aggregator/pkg/apiserver/scheme"
	netutils "k8s.io/utils/net"

	"k8s.io/kubernetes/cmd/kube-apiserver/app"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/controlplane"
	controlplaneapiserver "k8s.io/kubernetes/pkg/controlplane/apiserver"
	generatedopenapi "k8s.io/kubernetes/pkg/generated/openapi"
	"k8s.io/kubernetes/test/e2e/invariants/metrics"
	"k8s.io/kubernetes/test/utils"
)

// This key is for testing purposes only and is not considered secure.
const ecdsaPrivateKey = `-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIEZmTmUhuanLjPA2CLquXivuwBDHTt5XYwgIr/kA1LtRoAoGCCqGSM49
AwEHoUQDQgAEH6cuzP8XuD5wal6wf9M6xDljTOPLX2i8uIp/C/ASqiIGUeeKQtX0
/IR3qCXyThP/dbCiHrF3v1cuhBOHY8CLVg==
-----END EC PRIVATE KEY-----`

// TestServerSetup holds configuration information for a kube-apiserver test server.
type TestServerSetup struct {
	ModifyServerRunOptions func(*options.ServerRunOptions)
	ModifyServerConfig     func(*controlplane.Config)
	// DisableInvariantChecks skips the invariant checks at the end of the test.
	DisableInvariantChecks bool
	// Flags holds command-line flags to apply to the server.
	Flags []string
}

var (
	buildBinaryOnce sync.Once
	apiserverBinary string
)

func buildAPIServer(t testing.TB) string {
	buildBinaryOnce.Do(func() {
		apiserverBinary = filepath.Join(os.TempDir(), "kube-apiserver-integration-"+uuid.New().String())
		cmd := exec.Command("go", "build", "-o", apiserverBinary, "k8s.io/kubernetes/cmd/kube-apiserver")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			t.Fatalf("Failed to build kube-apiserver: %v", err)
		}
	})
	return apiserverBinary
}

type TearDownFunc func()

type testServerEnvironment struct {
	certDir   string
	port      int
	listener  net.Listener
	finalArgs []string
}

func setupTestServerEnvironment(t testing.TB, setupFlags []string) testServerEnvironment {
	certDir, err := os.MkdirTemp("", "test-integration-"+strings.ReplaceAll(t.Name(), "/", "_"))
	if err != nil {
		t.Fatalf("Couldn't create temp dir: %v", err)
	}

	_, defaultServiceClusterIPRange, _ := netutils.ParseCIDRSloppy("10.0.0.0/24")
	proxySigningKey, err := utils.NewPrivateKey()
	if err != nil {
		t.Fatal(err)
	}
	proxySigningCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: "front-proxy-ca"}, proxySigningKey)
	if err != nil {
		t.Fatal(err)
	}
	proxyCACertFile, _ := os.CreateTemp(certDir, "proxy-ca.crt")
	if err := os.WriteFile(proxyCACertFile.Name(), utils.EncodeCertPEM(proxySigningCert), 0644); err != nil {
		t.Fatal(err)
	}
	defer proxyCACertFile.Close()
	
	clientSigningKey, err := utils.NewPrivateKey()
	if err != nil {
		t.Fatal(err)
	}
	clientSigningCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: "client-ca"}, clientSigningKey)
	if err != nil {
		t.Fatal(err)
	}
	clientCACertFile, _ := os.CreateTemp(certDir, "client-ca.crt")
	if err := os.WriteFile(clientCACertFile.Name(), utils.EncodeCertPEM(clientSigningCert), 0644); err != nil {
		t.Fatal(err)
	}
	defer clientCACertFile.Close()
	
	listener, _, err := genericapiserveroptions.CreateListener("tcp", "127.0.0.1:0", net.ListenConfig{})
	if err != nil {
		t.Fatal(err)
	}

	saSigningKeyFile, err := os.CreateTemp("/tmp", "insecure_test_key")
	if err != nil {
		t.Fatalf("create temp file failed: %v", err)
	}
	defer saSigningKeyFile.Close()
	if err = os.WriteFile(saSigningKeyFile.Name(), []byte(ecdsaPrivateKey), 0666); err != nil {
		t.Fatalf("write file %s failed: %v", saSigningKeyFile.Name(), err)
	}

	tokenFile := filepath.Join(certDir, "token.csv")
	if err := os.WriteFile(tokenFile, []byte("test-token,test-user,1,\"system:masters\"\n"), 0644); err != nil {
		t.Fatalf("Failed to write token.csv: %v", err)
	}

	port := listener.Addr().(*net.TCPAddr).Port

	defaultArgs := []string{
		fmt.Sprintf("--secure-port=%d", port),
		"--bind-address=127.0.0.1",
		"--cert-dir=" + certDir,
		"--service-account-signing-key-file=" + saSigningKeyFile.Name(),
		"--service-account-key-file=" + saSigningKeyFile.Name(),
		"--service-account-issuer=https://kubernetes.default.svc",
		"--authorization-mode=Node,RBAC",
		"--etcd-prefix=" + path.Join("/", uuid.New().String(), "registry"),
		"--etcd-servers=" + GetEtcdURL(),
		"--service-cluster-ip-range=" + defaultServiceClusterIPRange.String(),
		"--requestheader-username-headers=X-Remote-User",
		"--requestheader-group-headers=X-Remote-Group",
		"--requestheader-extra-headers-prefix=X-Remote-Extra-",
		"--requestheader-allowed-names=kube-aggregator",
		"--requestheader-client-ca-file=" + proxyCACertFile.Name(),
		"--client-ca-file=" + clientCACertFile.Name(),
		"--api-audiences=https://foo.bar.example.com",
		"--token-auth-file=" + tokenFile,
	}

	overrideFlags := make(map[string]bool)
	for _, f := range setupFlags {
		parts := strings.SplitN(f, "=", 2)
		name := strings.TrimPrefix(parts[0], "--")
		overrideFlags[name] = true
	}
	
	var finalArgs []string
	for _, f := range defaultArgs {
		parts := strings.SplitN(f, "=", 2)
		name := strings.TrimPrefix(parts[0], "--")
		if !overrideFlags[name] {
			finalArgs = append(finalArgs, f)
		}
	}
	finalArgs = append(finalArgs, setupFlags...)

	return testServerEnvironment{
		certDir:   certDir,
		port:      port,
		listener:  listener,
		finalArgs: finalArgs,
	}
}

// StartTestServer runs a kube-apiserver, optionally calling out to the setup.ModifyServerRunOptions and setup.ModifyServerConfig functions
// TODO (pohly): convert to ktesting contexts
func StartTestServer(ctx context.Context, t testing.TB, setup TestServerSetup) (client.Interface, *rest.Config, TearDownFunc) {
	ctx, cancel := context.WithCancel(ctx)
	env := setupTestServerEnvironment(t, setup.Flags)

	opts := options.NewServerRunOptions()

	// If EmulationVersion of DefaultFeatureGate is set during test, we need to propagate it to the apiserver ComponentGlobalsRegistry.
	featureGate := utilfeature.DefaultMutableFeatureGate.DeepCopy()
	effectiveVersion := compatibility.DefaultKubeEffectiveVersionForTest()
	effectiveVersion.SetEmulationVersion(featureGate.EmulationVersion())
	// set up new instance of ComponentGlobalsRegistry instead of using the DefaultComponentGlobalsRegistry to avoid contention in parallel tests.
	componentGlobalsRegistry := basecompatibility.NewComponentGlobalsRegistry()
	if err := componentGlobalsRegistry.Register(basecompatibility.DefaultKubeComponent, effectiveVersion, featureGate); err != nil {
		t.Fatal(err)
	}
	opts.GenericServerRunOptions.ComponentGlobalsRegistry = componentGlobalsRegistry

	// Parse default arguments and user-provided flags into opts.
	// This ensures both in-process and out-of-process paths share the exact same configuration logic.
	flagSet := pflag.NewFlagSet("test", pflag.ContinueOnError)
	for _, f := range opts.Flags().FlagSets {
		flagSet.AddFlagSet(f)
	}
	if err := flagSet.Parse(env.finalArgs); err != nil {
		t.Fatalf("Failed to parse flags: %v", err)
	}

	if setup.ModifyServerRunOptions != nil {
		setup.ModifyServerRunOptions(opts)
	}

	if err := opts.GenericServerRunOptions.ComponentGlobalsRegistry.Set(); err != nil {
		t.Fatalf("Failed to set ComponentGlobalsRegistry: %v", err)
	}

	// In-process execution must use the listener we bound
	opts.SecureServing.Listener = env.listener

	// If the local ComponentGlobalsRegistry is changed by ModifyServerRunOptions,
	// we need to copy the new feature values back to the DefaultFeatureGate because most feature checks still use the DefaultFeatureGate.
	if !featureGate.EmulationVersion().EqualTo(utilfeature.DefaultMutableFeatureGate.EmulationVersion()) || !featureGate.MinCompatibilityVersion().EqualTo(utilfeature.DefaultMutableFeatureGate.MinCompatibilityVersion()) {
		featuregatetesting.SetFeatureGateVersionsDuringTest(t, utilfeature.DefaultMutableFeatureGate, effectiveVersion.EmulationVersion(), effectiveVersion.MinCompatibilityVersion())
	}
	parsedFeatureGate := opts.GenericServerRunOptions.ComponentGlobalsRegistry.FeatureGateFor(basecompatibility.DefaultKubeComponent)
	featureOverrides := map[featuregate.Feature]bool{}
	for f := range utilfeature.DefaultMutableFeatureGate.GetAll() {
		if parsedFeatureGate.Enabled(f) != utilfeature.DefaultFeatureGate.Enabled(f) {
			featureOverrides[f] = parsedFeatureGate.Enabled(f)
		}
	}
	if len(featureOverrides) > 0 {
		featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featureOverrides)
	}
	utilfeature.DefaultMutableFeatureGate.AddMetrics()

	completedOptions, err := opts.Complete(ctx)
	if err != nil {
		t.Fatal(err)
	}

	if errs := completedOptions.Validate(); len(errs) != 0 {
		t.Fatalf("failed to validate ServerRunOptions: %v", utilerrors.NewAggregate(errs))
	}

	genericConfig, versionedInformers, storageFactory, err := controlplaneapiserver.BuildGenericConfig(
		completedOptions.CompletedOptions,
		[]*runtime.Scheme{legacyscheme.Scheme, apiextensionsapiserver.Scheme, aggregatorscheme.Scheme},
		controlplane.DefaultAPIResourceConfigSource(),
		generatedopenapi.GetOpenAPIDefinitions,
	)
	if err != nil {
		t.Fatal(err)
	}

	kubeAPIServerConfig, _, _, err := app.CreateKubeAPIServerConfig(completedOptions, genericConfig, versionedInformers, storageFactory)
	if err != nil {
		t.Fatal(err)
	}

	if setup.ModifyServerConfig != nil {
		setup.ModifyServerConfig(kubeAPIServerConfig)
	}
	kubeAPIServer, err := kubeAPIServerConfig.Complete().New(genericapiserver.NewEmptyDelegate())
	if err != nil {
		t.Fatal(err)
	}

	errCh := make(chan error)
	go func() {
		defer close(errCh)
		if err := kubeAPIServer.ControlPlane.GenericAPIServer.PrepareRun().RunWithContext(ctx); err != nil {
			errCh <- err
		}
	}()

	// Adjust the loopback config for external use (external server name and CA)
	kubeAPIServerClientConfig := rest.CopyConfig(kubeAPIServerConfig.ControlPlane.Generic.LoopbackClientConfig)
	kubeAPIServerClientConfig.CAFile = path.Join(env.certDir, "apiserver.crt")
	kubeAPIServerClientConfig.CAData = nil
	kubeAPIServerClientConfig.ServerName = ""

	return waitForServerReady(ctx, t, env, setup, errCh, kubeAPIServerClientConfig, cancel)
}

// StartTestServerProcess runs a kube-apiserver as a separate process.
func StartTestServerProcess(ctx context.Context, t testing.TB, setup TestServerSetup) (client.Interface, *rest.Config, TearDownFunc) {
	if setup.ModifyServerRunOptions != nil || setup.ModifyServerConfig != nil {
		t.Fatalf("ModifyServerRunOptions and ModifyServerConfig are not supported for StartTestServerProcess")
	}

	ctx, cancel := context.WithCancel(ctx)
	env := setupTestServerEnvironment(t, setup.Flags)
	
	binary := buildAPIServer(t)
	env.listener.Close() // Release port for the child process

	cmd := exec.CommandContext(ctx, binary, env.finalArgs...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatalf("Failed to start apiserver process: %v", err)
	}

	errCh := make(chan error)
	go func() {
		defer close(errCh)
		if err := cmd.Wait(); err != nil {
			if err.Error() != "signal: killed" {
				errCh <- err
			}
		}
	}()

	kubeAPIServerClientConfig := &rest.Config{
		Host:        fmt.Sprintf("https://127.0.0.1:%d", env.port),
		BearerToken: "test-token",
		TLSClientConfig: rest.TLSClientConfig{
			Insecure: true,
		},
	}

	return waitForServerReady(ctx, t, env, setup, errCh, kubeAPIServerClientConfig, cancel)
}

func waitForServerReady(ctx context.Context, t testing.TB, env testServerEnvironment, setup TestServerSetup, errCh chan error, kubeAPIServerClientConfig *rest.Config, cancel context.CancelFunc) (client.Interface, *rest.Config, TearDownFunc) {
	// wait for health
	err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (done bool, err error) {
		select {
		case err := <-errCh:
			return false, err
		default:
		}

		healthzConfig := rest.CopyConfig(kubeAPIServerClientConfig)
		healthzConfig.ContentType = ""
		healthzConfig.AcceptContentTypes = ""
		kubeClient, err := client.NewForConfig(healthzConfig)
		if err != nil {
			// this happens because we race the API server start
			t.Log(err)
			return false, nil
		}

		healthStatus := 0
		kubeClient.Discovery().RESTClient().Get().AbsPath("/healthz").Do(ctx).StatusCode(&healthStatus)
		if healthStatus != http.StatusOK {
			return false, nil
		}

		if _, err := kubeClient.CoreV1().Namespaces().Get(ctx, "default", metav1.GetOptions{}); err != nil {
			return false, nil
		}
		if _, err := kubeClient.CoreV1().Namespaces().Get(ctx, "kube-system", metav1.GetOptions{}); err != nil {
			return false, nil
		}

		return true, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	kubeAPIServerClient, err := client.NewForConfig(kubeAPIServerClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	tearDownFn := func() {
		// Scrape metrics before stopping
		if !setup.DisableInvariantChecks {
			if ctx.Err() != nil {
				t.Logf("Skipping metrics scrape because context is already canceled: %v", ctx.Err())
			} else {
				if err := metrics.CheckMetricInvariants(ctx, kubeAPIServerClient, false); err != nil {
					t.Errorf("Invariant check failed (if the test intentionally breaks metrics/auth, consider setting DisableInvariantChecks: true in TestServerSetup): %v", err)
				}
			}
		}
		// Calling cancel function is stopping apiserver and cleaning up
		// after itself, including shutting down its storage layer.
		cancel()

		// If the apiserver was started, let's wait for it to
		// shutdown clearly.
		if errCh != nil {
			err, ok := <-errCh
			if ok && err != nil {
				t.Error(err)
			}
		}
		if err := os.RemoveAll(env.certDir); err != nil {
			t.Log(err)
		}
	}

	return kubeAPIServerClient, kubeAPIServerClientConfig, tearDownFn
}
