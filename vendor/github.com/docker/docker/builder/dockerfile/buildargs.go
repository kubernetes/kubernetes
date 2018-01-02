package dockerfile

import (
	"fmt"
	"io"

	"github.com/docker/docker/runconfig/opts"
)

// builtinAllowedBuildArgs is list of built-in allowed build args
// these args are considered transparent and are excluded from the image history.
// Filtering from history is implemented in dispatchers.go
var builtinAllowedBuildArgs = map[string]bool{
	"HTTP_PROXY":  true,
	"http_proxy":  true,
	"HTTPS_PROXY": true,
	"https_proxy": true,
	"FTP_PROXY":   true,
	"ftp_proxy":   true,
	"NO_PROXY":    true,
	"no_proxy":    true,
}

// buildArgs manages arguments used by the builder
type buildArgs struct {
	// args that are allowed for expansion/substitution and passing to commands in 'run'.
	allowedBuildArgs map[string]*string
	// args defined before the first `FROM` in a Dockerfile
	allowedMetaArgs map[string]*string
	// args referenced by the Dockerfile
	referencedArgs map[string]struct{}
	// args provided by the user on the command line
	argsFromOptions map[string]*string
}

func newBuildArgs(argsFromOptions map[string]*string) *buildArgs {
	return &buildArgs{
		allowedBuildArgs: make(map[string]*string),
		allowedMetaArgs:  make(map[string]*string),
		referencedArgs:   make(map[string]struct{}),
		argsFromOptions:  argsFromOptions,
	}
}

// WarnOnUnusedBuildArgs checks if there are any leftover build-args that were
// passed but not consumed during build. Print a warning, if there are any.
func (b *buildArgs) WarnOnUnusedBuildArgs(out io.Writer) {
	leftoverArgs := []string{}
	for arg := range b.argsFromOptions {
		_, isReferenced := b.referencedArgs[arg]
		_, isBuiltin := builtinAllowedBuildArgs[arg]
		if !isBuiltin && !isReferenced {
			leftoverArgs = append(leftoverArgs, arg)
		}
	}
	if len(leftoverArgs) > 0 {
		fmt.Fprintf(out, "[Warning] One or more build-args %v were not consumed\n", leftoverArgs)
	}
}

// ResetAllowed clears the list of args that are allowed to be used by a
// directive
func (b *buildArgs) ResetAllowed() {
	b.allowedBuildArgs = make(map[string]*string)
}

// AddMetaArg adds a new meta arg that can be used by FROM directives
func (b *buildArgs) AddMetaArg(key string, value *string) {
	b.allowedMetaArgs[key] = value
}

// AddArg adds a new arg that can be used by directives
func (b *buildArgs) AddArg(key string, value *string) {
	b.allowedBuildArgs[key] = value
	b.referencedArgs[key] = struct{}{}
}

// IsReferencedOrNotBuiltin checks if the key is a built-in arg, or if it has been
// referenced by the Dockerfile. Returns true if the arg is not a builtin or
// if the builtin has been referenced in the Dockerfile.
func (b *buildArgs) IsReferencedOrNotBuiltin(key string) bool {
	_, isBuiltin := builtinAllowedBuildArgs[key]
	_, isAllowed := b.allowedBuildArgs[key]
	return isAllowed || !isBuiltin
}

// GetAllAllowed returns a mapping with all the allowed args
func (b *buildArgs) GetAllAllowed() map[string]string {
	return b.getAllFromMapping(b.allowedBuildArgs)
}

// GetAllMeta returns a mapping with all the meta meta args
func (b *buildArgs) GetAllMeta() map[string]string {
	return b.getAllFromMapping(b.allowedMetaArgs)
}

func (b *buildArgs) getAllFromMapping(source map[string]*string) map[string]string {
	m := make(map[string]string)

	keys := keysFromMaps(source, builtinAllowedBuildArgs)
	for _, key := range keys {
		v, ok := b.getBuildArg(key, source)
		if ok {
			m[key] = v
		}
	}
	return m
}

// FilterAllowed returns all allowed args without the filtered args
func (b *buildArgs) FilterAllowed(filter []string) []string {
	envs := []string{}
	configEnv := opts.ConvertKVStringsToMap(filter)

	for key, val := range b.GetAllAllowed() {
		if _, ok := configEnv[key]; !ok {
			envs = append(envs, fmt.Sprintf("%s=%s", key, val))
		}
	}
	return envs
}

func (b *buildArgs) getBuildArg(key string, mapping map[string]*string) (string, bool) {
	defaultValue, exists := mapping[key]
	// Return override from options if one is defined
	if v, ok := b.argsFromOptions[key]; ok && v != nil {
		return *v, ok
	}

	if defaultValue == nil {
		if v, ok := b.allowedMetaArgs[key]; ok && v != nil {
			return *v, ok
		}
		return "", false
	}
	return *defaultValue, exists
}

func keysFromMaps(source map[string]*string, builtin map[string]bool) []string {
	keys := []string{}
	for key := range source {
		keys = append(keys, key)
	}
	for key := range builtin {
		keys = append(keys, key)
	}
	return keys
}
