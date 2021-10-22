package sdktesting

import (
	"os"
	"runtime"
	"strings"
)

// StashEnv stashes the current environment variables except variables listed in envToKeepx
// Returns an function to pop out old environment
func StashEnv(envToKeep ...string) func() {
	if runtime.GOOS == "windows" {
		envToKeep = append(envToKeep, "ComSpec")
		envToKeep = append(envToKeep, "SYSTEM32")
		envToKeep = append(envToKeep, "SYSTEMROOT")
	}
	envToKeep = append(envToKeep, "PATH")
	extraEnv := getEnvs(envToKeep)
	originalEnv := os.Environ()
	os.Clearenv() // clear env
	for key, val := range extraEnv {
		os.Setenv(key, val)
	}
	return func() {
		popEnv(originalEnv)
	}
}

func getEnvs(envs []string) map[string]string {
	extraEnvs := make(map[string]string)
	for _, env := range envs {
		if val, ok := os.LookupEnv(env); ok && len(val) > 0 {
			extraEnvs[env] = val
		}
	}
	return extraEnvs
}

// PopEnv takes the list of the environment values and injects them into the
// process's environment variable data. Clears any existing environment values
// that may already exist.
func popEnv(env []string) {
	os.Clearenv()

	for _, e := range env {
		p := strings.SplitN(e, "=", 2)
		k, v := p[0], ""
		if len(p) > 1 {
			v = p[1]
		}
		os.Setenv(k, v)
	}
}
