package goutil

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/pkg/errors"

	"github.com/golangci/golangci-lint/pkg/logutils"
)

type EnvKey string

const (
	EnvGoCache EnvKey = "GOCACHE"
	EnvGoRoot  EnvKey = "GOROOT"
)

type Env struct {
	vars   map[string]string
	log    logutils.Log
	debugf logutils.DebugFunc
}

func NewEnv(log logutils.Log) *Env {
	return &Env{
		vars:   map[string]string{},
		log:    log,
		debugf: logutils.Debug("env"),
	}
}

func (e *Env) Discover(ctx context.Context) error {
	startedAt := time.Now()
	args := []string{"env", "-json"}
	args = append(args, string(EnvGoCache), string(EnvGoRoot))
	out, err := exec.CommandContext(ctx, "go", args...).Output()
	if err != nil {
		return errors.Wrap(err, "failed to run 'go env'")
	}

	if err = json.Unmarshal(out, &e.vars); err != nil {
		return errors.Wrapf(err, "failed to parse 'go %s' json", strings.Join(args, " "))
	}

	e.debugf("Read go env for %s: %#v", time.Since(startedAt), e.vars)
	return nil
}

func (e Env) Get(k EnvKey) string {
	envValue := os.Getenv(string(k))
	if envValue != "" {
		return envValue
	}

	return e.vars[string(k)]
}
