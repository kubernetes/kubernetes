package plugin

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	"github.com/docker/distribution/reference"
	"github.com/docker/docker/api/server/httputils"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/streamformatter"
	"github.com/pkg/errors"
	"golang.org/x/net/context"
)

func parseHeaders(headers http.Header) (map[string][]string, *types.AuthConfig) {

	metaHeaders := map[string][]string{}
	for k, v := range headers {
		if strings.HasPrefix(k, "X-Meta-") {
			metaHeaders[k] = v
		}
	}

	// Get X-Registry-Auth
	authEncoded := headers.Get("X-Registry-Auth")
	authConfig := &types.AuthConfig{}
	if authEncoded != "" {
		authJSON := base64.NewDecoder(base64.URLEncoding, strings.NewReader(authEncoded))
		if err := json.NewDecoder(authJSON).Decode(authConfig); err != nil {
			authConfig = &types.AuthConfig{}
		}
	}

	return metaHeaders, authConfig
}

// parseRemoteRef parses the remote reference into a reference.Named
// returning the tag associated with the reference. In the case the
// given reference string includes both digest and tag, the returned
// reference will have the digest without the tag, but the tag will
// be returned.
func parseRemoteRef(remote string) (reference.Named, string, error) {
	// Parse remote reference, supporting remotes with name and tag
	remoteRef, err := reference.ParseNormalizedNamed(remote)
	if err != nil {
		return nil, "", err
	}

	type canonicalWithTag interface {
		reference.Canonical
		Tag() string
	}

	if canonical, ok := remoteRef.(canonicalWithTag); ok {
		remoteRef, err = reference.WithDigest(reference.TrimNamed(remoteRef), canonical.Digest())
		if err != nil {
			return nil, "", err
		}
		return remoteRef, canonical.Tag(), nil
	}

	remoteRef = reference.TagNameOnly(remoteRef)

	return remoteRef, "", nil
}

func (pr *pluginRouter) getPrivileges(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	metaHeaders, authConfig := parseHeaders(r.Header)

	ref, _, err := parseRemoteRef(r.FormValue("remote"))
	if err != nil {
		return err
	}

	privileges, err := pr.backend.Privileges(ctx, ref, metaHeaders, authConfig)
	if err != nil {
		return err
	}
	return httputils.WriteJSON(w, http.StatusOK, privileges)
}

func (pr *pluginRouter) upgradePlugin(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return errors.Wrap(err, "failed to parse form")
	}

	var privileges types.PluginPrivileges
	dec := json.NewDecoder(r.Body)
	if err := dec.Decode(&privileges); err != nil {
		return errors.Wrap(err, "failed to parse privileges")
	}
	if dec.More() {
		return errors.New("invalid privileges")
	}

	metaHeaders, authConfig := parseHeaders(r.Header)
	ref, tag, err := parseRemoteRef(r.FormValue("remote"))
	if err != nil {
		return err
	}

	name, err := getName(ref, tag, vars["name"])
	if err != nil {
		return err
	}
	w.Header().Set("Docker-Plugin-Name", name)

	w.Header().Set("Content-Type", "application/json")
	output := ioutils.NewWriteFlusher(w)

	if err := pr.backend.Upgrade(ctx, ref, name, metaHeaders, authConfig, privileges, output); err != nil {
		if !output.Flushed() {
			return err
		}
		output.Write(streamformatter.FormatError(err))
	}

	return nil
}

func (pr *pluginRouter) pullPlugin(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return errors.Wrap(err, "failed to parse form")
	}

	var privileges types.PluginPrivileges
	dec := json.NewDecoder(r.Body)
	if err := dec.Decode(&privileges); err != nil {
		return errors.Wrap(err, "failed to parse privileges")
	}
	if dec.More() {
		return errors.New("invalid privileges")
	}

	metaHeaders, authConfig := parseHeaders(r.Header)
	ref, tag, err := parseRemoteRef(r.FormValue("remote"))
	if err != nil {
		return err
	}

	name, err := getName(ref, tag, r.FormValue("name"))
	if err != nil {
		return err
	}
	w.Header().Set("Docker-Plugin-Name", name)

	w.Header().Set("Content-Type", "application/json")
	output := ioutils.NewWriteFlusher(w)

	if err := pr.backend.Pull(ctx, ref, name, metaHeaders, authConfig, privileges, output); err != nil {
		if !output.Flushed() {
			return err
		}
		output.Write(streamformatter.FormatError(err))
	}

	return nil
}

func getName(ref reference.Named, tag, name string) (string, error) {
	if name == "" {
		if _, ok := ref.(reference.Canonical); ok {
			trimmed := reference.TrimNamed(ref)
			if tag != "" {
				nt, err := reference.WithTag(trimmed, tag)
				if err != nil {
					return "", err
				}
				name = reference.FamiliarString(nt)
			} else {
				name = reference.FamiliarString(reference.TagNameOnly(trimmed))
			}
		} else {
			name = reference.FamiliarString(ref)
		}
	} else {
		localRef, err := reference.ParseNormalizedNamed(name)
		if err != nil {
			return "", err
		}
		if _, ok := localRef.(reference.Canonical); ok {
			return "", errors.New("cannot use digest in plugin tag")
		}
		if reference.IsNameOnly(localRef) {
			// TODO: log change in name to out stream
			name = reference.FamiliarString(reference.TagNameOnly(localRef))
		}
	}
	return name, nil
}

func (pr *pluginRouter) createPlugin(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	options := &types.PluginCreateOptions{
		RepoName: r.FormValue("name")}

	if err := pr.backend.CreateFromContext(ctx, r.Body, options); err != nil {
		return err
	}
	//TODO: send progress bar
	w.WriteHeader(http.StatusNoContent)
	return nil
}

func (pr *pluginRouter) enablePlugin(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	name := vars["name"]
	timeout, err := strconv.Atoi(r.Form.Get("timeout"))
	if err != nil {
		return err
	}
	config := &types.PluginEnableConfig{Timeout: timeout}

	return pr.backend.Enable(name, config)
}

func (pr *pluginRouter) disablePlugin(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	name := vars["name"]
	config := &types.PluginDisableConfig{
		ForceDisable: httputils.BoolValue(r, "force"),
	}

	return pr.backend.Disable(name, config)
}

func (pr *pluginRouter) removePlugin(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	name := vars["name"]
	config := &types.PluginRmConfig{
		ForceRemove: httputils.BoolValue(r, "force"),
	}
	return pr.backend.Remove(name, config)
}

func (pr *pluginRouter) pushPlugin(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return errors.Wrap(err, "failed to parse form")
	}

	metaHeaders, authConfig := parseHeaders(r.Header)

	w.Header().Set("Content-Type", "application/json")
	output := ioutils.NewWriteFlusher(w)

	if err := pr.backend.Push(ctx, vars["name"], metaHeaders, authConfig, output); err != nil {
		if !output.Flushed() {
			return err
		}
		output.Write(streamformatter.FormatError(err))
	}
	return nil
}

func (pr *pluginRouter) setPlugin(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	var args []string
	if err := json.NewDecoder(r.Body).Decode(&args); err != nil {
		return err
	}
	if err := pr.backend.Set(vars["name"], args); err != nil {
		return err
	}
	w.WriteHeader(http.StatusNoContent)
	return nil
}

func (pr *pluginRouter) listPlugins(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	pluginFilters, err := filters.FromParam(r.Form.Get("filters"))
	if err != nil {
		return err
	}
	l, err := pr.backend.List(pluginFilters)
	if err != nil {
		return err
	}
	return httputils.WriteJSON(w, http.StatusOK, l)
}

func (pr *pluginRouter) inspectPlugin(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	result, err := pr.backend.Inspect(vars["name"])
	if err != nil {
		return err
	}
	return httputils.WriteJSON(w, http.StatusOK, result)
}
