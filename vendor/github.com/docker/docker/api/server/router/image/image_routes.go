package image

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/docker/docker/api/server/httputils"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/backend"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/api/types/versions"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/streamformatter"
	"github.com/docker/docker/pkg/system"
	"github.com/docker/docker/registry"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
	"golang.org/x/net/context"
)

func (s *imageRouter) postCommit(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	if err := httputils.CheckForJSON(r); err != nil {
		return err
	}

	cname := r.Form.Get("container")

	pause := httputils.BoolValue(r, "pause")
	version := httputils.VersionFromContext(ctx)
	if r.FormValue("pause") == "" && versions.GreaterThanOrEqualTo(version, "1.13") {
		pause = true
	}

	c, _, _, err := s.decoder.DecodeConfig(r.Body)
	if err != nil && err != io.EOF { //Do not fail if body is empty.
		return err
	}

	commitCfg := &backend.ContainerCommitConfig{
		ContainerCommitConfig: types.ContainerCommitConfig{
			Pause:        pause,
			Repo:         r.Form.Get("repo"),
			Tag:          r.Form.Get("tag"),
			Author:       r.Form.Get("author"),
			Comment:      r.Form.Get("comment"),
			Config:       c,
			MergeConfigs: true,
		},
		Changes: r.Form["changes"],
	}

	imgID, err := s.backend.Commit(cname, commitCfg)
	if err != nil {
		return err
	}

	return httputils.WriteJSON(w, http.StatusCreated, &types.IDResponse{ID: imgID})
}

// Creates an image from Pull or from Import
func (s *imageRouter) postImagesCreate(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {

	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	var (
		image    = r.Form.Get("fromImage")
		repo     = r.Form.Get("repo")
		tag      = r.Form.Get("tag")
		message  = r.Form.Get("message")
		err      error
		output   = ioutils.NewWriteFlusher(w)
		platform = &specs.Platform{}
	)
	defer output.Close()

	w.Header().Set("Content-Type", "application/json")

	version := httputils.VersionFromContext(ctx)
	if versions.GreaterThanOrEqualTo(version, "1.32") {
		// TODO @jhowardmsft. The following environment variable is an interim
		// measure to allow the daemon to have a default platform if omitted by
		// the client. This allows LCOW and WCOW to work with a down-level CLI
		// for a short period of time, as the CLI changes can't be merged
		// until after the daemon changes have been merged. Once the CLI is
		// updated, this can be removed. PR for CLI is currently in
		// https://github.com/docker/cli/pull/474.
		apiPlatform := r.FormValue("platform")
		if system.LCOWSupported() && apiPlatform == "" {
			apiPlatform = os.Getenv("LCOW_API_PLATFORM_IF_OMITTED")
		}
		platform = system.ParsePlatform(apiPlatform)
		if err = system.ValidatePlatform(platform); err != nil {
			err = fmt.Errorf("invalid platform: %s", err)
		}
	}

	if err == nil {
		if image != "" { //pull
			metaHeaders := map[string][]string{}
			for k, v := range r.Header {
				if strings.HasPrefix(k, "X-Meta-") {
					metaHeaders[k] = v
				}
			}

			authEncoded := r.Header.Get("X-Registry-Auth")
			authConfig := &types.AuthConfig{}
			if authEncoded != "" {
				authJSON := base64.NewDecoder(base64.URLEncoding, strings.NewReader(authEncoded))
				if err := json.NewDecoder(authJSON).Decode(authConfig); err != nil {
					// for a pull it is not an error if no auth was given
					// to increase compatibility with the existing api it is defaulting to be empty
					authConfig = &types.AuthConfig{}
				}
			}
			err = s.backend.PullImage(ctx, image, tag, platform.OS, metaHeaders, authConfig, output)
		} else { //import
			src := r.Form.Get("fromSrc")
			// 'err' MUST NOT be defined within this block, we need any error
			// generated from the download to be available to the output
			// stream processing below
			err = s.backend.ImportImage(src, repo, platform.OS, tag, message, r.Body, output, r.Form["changes"])
		}
	}
	if err != nil {
		if !output.Flushed() {
			return err
		}
		output.Write(streamformatter.FormatError(err))
	}

	return nil
}

type validationError struct {
	cause error
}

func (e validationError) Error() string {
	return e.cause.Error()
}

func (e validationError) Cause() error {
	return e.cause
}

func (validationError) InvalidParameter() {}

func (s *imageRouter) postImagesPush(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	metaHeaders := map[string][]string{}
	for k, v := range r.Header {
		if strings.HasPrefix(k, "X-Meta-") {
			metaHeaders[k] = v
		}
	}
	if err := httputils.ParseForm(r); err != nil {
		return err
	}
	authConfig := &types.AuthConfig{}

	authEncoded := r.Header.Get("X-Registry-Auth")
	if authEncoded != "" {
		// the new format is to handle the authConfig as a header
		authJSON := base64.NewDecoder(base64.URLEncoding, strings.NewReader(authEncoded))
		if err := json.NewDecoder(authJSON).Decode(authConfig); err != nil {
			// to increase compatibility to existing api it is defaulting to be empty
			authConfig = &types.AuthConfig{}
		}
	} else {
		// the old format is supported for compatibility if there was no authConfig header
		if err := json.NewDecoder(r.Body).Decode(authConfig); err != nil {
			return errors.Wrap(validationError{err}, "Bad parameters and missing X-Registry-Auth")
		}
	}

	image := vars["name"]
	tag := r.Form.Get("tag")

	output := ioutils.NewWriteFlusher(w)
	defer output.Close()

	w.Header().Set("Content-Type", "application/json")

	if err := s.backend.PushImage(ctx, image, tag, metaHeaders, authConfig, output); err != nil {
		if !output.Flushed() {
			return err
		}
		output.Write(streamformatter.FormatError(err))
	}
	return nil
}

func (s *imageRouter) getImagesGet(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	w.Header().Set("Content-Type", "application/x-tar")

	output := ioutils.NewWriteFlusher(w)
	defer output.Close()
	var names []string
	if name, ok := vars["name"]; ok {
		names = []string{name}
	} else {
		names = r.Form["names"]
	}

	if err := s.backend.ExportImage(names, output); err != nil {
		if !output.Flushed() {
			return err
		}
		output.Write(streamformatter.FormatError(err))
	}
	return nil
}

func (s *imageRouter) postImagesLoad(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}
	quiet := httputils.BoolValueOrDefault(r, "quiet", true)

	w.Header().Set("Content-Type", "application/json")

	output := ioutils.NewWriteFlusher(w)
	defer output.Close()
	if err := s.backend.LoadImage(r.Body, output, quiet); err != nil {
		output.Write(streamformatter.FormatError(err))
	}
	return nil
}

type missingImageError struct{}

func (missingImageError) Error() string {
	return "image name cannot be blank"
}

func (missingImageError) InvalidParameter() {}

func (s *imageRouter) deleteImages(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	name := vars["name"]

	if strings.TrimSpace(name) == "" {
		return missingImageError{}
	}

	force := httputils.BoolValue(r, "force")
	prune := !httputils.BoolValue(r, "noprune")

	list, err := s.backend.ImageDelete(name, force, prune)
	if err != nil {
		return err
	}

	return httputils.WriteJSON(w, http.StatusOK, list)
}

func (s *imageRouter) getImagesByName(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	imageInspect, err := s.backend.LookupImage(vars["name"])
	if err != nil {
		return err
	}

	return httputils.WriteJSON(w, http.StatusOK, imageInspect)
}

func (s *imageRouter) getImagesJSON(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	imageFilters, err := filters.FromJSON(r.Form.Get("filters"))
	if err != nil {
		return err
	}

	filterParam := r.Form.Get("filter")
	// FIXME(vdemeester) This has been deprecated in 1.13, and is target for removal for v17.12
	if filterParam != "" {
		imageFilters.Add("reference", filterParam)
	}

	images, err := s.backend.Images(imageFilters, httputils.BoolValue(r, "all"), false)
	if err != nil {
		return err
	}

	return httputils.WriteJSON(w, http.StatusOK, images)
}

func (s *imageRouter) getImagesHistory(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	name := vars["name"]
	history, err := s.backend.ImageHistory(name)
	if err != nil {
		return err
	}

	return httputils.WriteJSON(w, http.StatusOK, history)
}

func (s *imageRouter) postImagesTag(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}
	if err := s.backend.TagImage(vars["name"], r.Form.Get("repo"), r.Form.Get("tag")); err != nil {
		return err
	}
	w.WriteHeader(http.StatusCreated)
	return nil
}

func (s *imageRouter) getImagesSearch(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}
	var (
		config      *types.AuthConfig
		authEncoded = r.Header.Get("X-Registry-Auth")
		headers     = map[string][]string{}
	)

	if authEncoded != "" {
		authJSON := base64.NewDecoder(base64.URLEncoding, strings.NewReader(authEncoded))
		if err := json.NewDecoder(authJSON).Decode(&config); err != nil {
			// for a search it is not an error if no auth was given
			// to increase compatibility with the existing api it is defaulting to be empty
			config = &types.AuthConfig{}
		}
	}
	for k, v := range r.Header {
		if strings.HasPrefix(k, "X-Meta-") {
			headers[k] = v
		}
	}
	limit := registry.DefaultSearchLimit
	if r.Form.Get("limit") != "" {
		limitValue, err := strconv.Atoi(r.Form.Get("limit"))
		if err != nil {
			return err
		}
		limit = limitValue
	}
	query, err := s.backend.SearchRegistryForImages(ctx, r.Form.Get("filters"), r.Form.Get("term"), limit, config, headers)
	if err != nil {
		return err
	}
	return httputils.WriteJSON(w, http.StatusOK, query.Results)
}

func (s *imageRouter) postImagesPrune(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	pruneFilters, err := filters.FromJSON(r.Form.Get("filters"))
	if err != nil {
		return err
	}

	pruneReport, err := s.backend.ImagesPrune(ctx, pruneFilters)
	if err != nil {
		return err
	}
	return httputils.WriteJSON(w, http.StatusOK, pruneReport)
}
