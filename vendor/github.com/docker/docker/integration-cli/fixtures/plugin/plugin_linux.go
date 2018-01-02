package plugin

import (
	"encoding/json"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/libcontainerd"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/plugin"
	"github.com/docker/docker/registry"
	"github.com/pkg/errors"
	"golang.org/x/net/context"
)

// Create creates a new plugin with the specified name
func Create(ctx context.Context, c CreateClient, name string, opts ...CreateOpt) error {
	tmpDir, err := ioutil.TempDir("", "create-test-plugin")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)

	tar, err := makePluginBundle(tmpDir, opts...)
	if err != nil {
		return err
	}
	defer tar.Close()

	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	return c.PluginCreate(ctx, tar, types.PluginCreateOptions{RepoName: name})
}

// TODO(@cpuguy83): we really shouldn't have to do this...
// The manager panics on init when `Executor` is not set.
type dummyExecutor struct{}

func (dummyExecutor) Client(libcontainerd.Backend) (libcontainerd.Client, error) { return nil, nil }
func (dummyExecutor) Cleanup()                                                   {}
func (dummyExecutor) UpdateOptions(...libcontainerd.RemoteOption) error          { return nil }

// CreateInRegistry makes a plugin (locally) and pushes it to a registry.
// This does not use a dockerd instance to create or push the plugin.
// If you just want to create a plugin in some daemon, use `Create`.
//
// This can be useful when testing plugins on swarm where you don't really want
// the plugin to exist on any of the daemons (immediately) and there needs to be
// some way to distribute the plugin.
func CreateInRegistry(ctx context.Context, repo string, auth *types.AuthConfig, opts ...CreateOpt) error {
	tmpDir, err := ioutil.TempDir("", "create-test-plugin-local")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)

	inPath := filepath.Join(tmpDir, "plugin")
	if err := os.MkdirAll(inPath, 0755); err != nil {
		return errors.Wrap(err, "error creating plugin root")
	}

	tar, err := makePluginBundle(inPath, opts...)
	if err != nil {
		return err
	}
	defer tar.Close()

	managerConfig := plugin.ManagerConfig{
		Store:           plugin.NewStore(),
		RegistryService: registry.NewService(registry.ServiceOptions{V2Only: true}),
		Root:            filepath.Join(tmpDir, "root"),
		ExecRoot:        "/run/docker", // manager init fails if not set
		Executor:        dummyExecutor{},
		LogPluginEvent:  func(id, name, action string) {}, // panics when not set
	}
	manager, err := plugin.NewManager(managerConfig)
	if err != nil {
		return errors.Wrap(err, "error creating plugin manager")
	}

	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	if err := manager.CreateFromContext(ctx, tar, &types.PluginCreateOptions{RepoName: repo}); err != nil {
		return err
	}

	if auth == nil {
		auth = &types.AuthConfig{}
	}
	err = manager.Push(ctx, repo, nil, auth, ioutil.Discard)
	return errors.Wrap(err, "error pushing plugin")
}

func makePluginBundle(inPath string, opts ...CreateOpt) (io.ReadCloser, error) {
	p := &types.PluginConfig{
		Interface: types.PluginConfigInterface{
			Socket: "basic.sock",
			Types:  []types.PluginInterfaceType{{Capability: "docker.dummy/1.0"}},
		},
		Entrypoint: []string{"/basic"},
	}
	cfg := &Config{
		PluginConfig: p,
	}
	for _, o := range opts {
		o(cfg)
	}
	if cfg.binPath == "" {
		binPath, err := ensureBasicPluginBin()
		if err != nil {
			return nil, err
		}
		cfg.binPath = binPath
	}

	configJSON, err := json.Marshal(p)
	if err != nil {
		return nil, err
	}
	if err := ioutil.WriteFile(filepath.Join(inPath, "config.json"), configJSON, 0644); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Join(inPath, "rootfs", filepath.Dir(p.Entrypoint[0])), 0755); err != nil {
		return nil, errors.Wrap(err, "error creating plugin rootfs dir")
	}
	if err := archive.NewDefaultArchiver().CopyFileWithTar(cfg.binPath, filepath.Join(inPath, "rootfs", p.Entrypoint[0])); err != nil {
		return nil, errors.Wrap(err, "error copying plugin binary to rootfs path")
	}
	tar, err := archive.Tar(inPath, archive.Uncompressed)
	return tar, errors.Wrap(err, "error making plugin archive")
}

func ensureBasicPluginBin() (string, error) {
	name := "docker-basic-plugin"
	p, err := exec.LookPath(name)
	if err == nil {
		return p, nil
	}

	goBin, err := exec.LookPath("go")
	if err != nil {
		return "", err
	}
	installPath := filepath.Join(os.Getenv("GOPATH"), "bin", name)
	cmd := exec.Command(goBin, "build", "-o", installPath, "./"+filepath.Join("fixtures", "plugin", "basic"))
	cmd.Env = append(cmd.Env, "CGO_ENABLED=0")
	if out, err := cmd.CombinedOutput(); err != nil {
		return "", errors.Wrapf(err, "error building basic plugin bin: %s", string(out))
	}
	return installPath, nil
}
