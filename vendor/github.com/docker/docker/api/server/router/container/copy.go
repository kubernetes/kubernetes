package container

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/docker/docker/api/server/httputils"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/versions"
	"golang.org/x/net/context"
)

// postContainersCopy is deprecated in favor of getContainersArchive.
func (s *containerRouter) postContainersCopy(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	// Deprecated since 1.8, Errors out since 1.12
	version := httputils.VersionFromContext(ctx)
	if versions.GreaterThanOrEqualTo(version, "1.24") {
		w.WriteHeader(http.StatusNotFound)
		return nil
	}
	if err := httputils.CheckForJSON(r); err != nil {
		return err
	}

	cfg := types.CopyConfig{}
	if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
		return err
	}

	if cfg.Resource == "" {
		return fmt.Errorf("Path cannot be empty")
	}

	data, err := s.backend.ContainerCopy(vars["name"], cfg.Resource)
	if err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "no such container") {
			w.WriteHeader(http.StatusNotFound)
			return nil
		}
		if os.IsNotExist(err) {
			return fmt.Errorf("Could not find the file %s in container %s", cfg.Resource, vars["name"])
		}
		return err
	}
	defer data.Close()

	w.Header().Set("Content-Type", "application/x-tar")
	_, err = io.Copy(w, data)
	return err
}

// // Encode the stat to JSON, base64 encode, and place in a header.
func setContainerPathStatHeader(stat *types.ContainerPathStat, header http.Header) error {
	statJSON, err := json.Marshal(stat)
	if err != nil {
		return err
	}

	header.Set(
		"X-Docker-Container-Path-Stat",
		base64.StdEncoding.EncodeToString(statJSON),
	)

	return nil
}

func (s *containerRouter) headContainersArchive(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	v, err := httputils.ArchiveFormValues(r, vars)
	if err != nil {
		return err
	}

	stat, err := s.backend.ContainerStatPath(v.Name, v.Path)
	if err != nil {
		return err
	}

	return setContainerPathStatHeader(stat, w.Header())
}

func (s *containerRouter) getContainersArchive(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	v, err := httputils.ArchiveFormValues(r, vars)
	if err != nil {
		return err
	}

	tarArchive, stat, err := s.backend.ContainerArchivePath(v.Name, v.Path)
	if err != nil {
		return err
	}
	defer tarArchive.Close()

	if err := setContainerPathStatHeader(stat, w.Header()); err != nil {
		return err
	}

	w.Header().Set("Content-Type", "application/x-tar")
	_, err = io.Copy(w, tarArchive)

	return err
}

func (s *containerRouter) putContainersArchive(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	v, err := httputils.ArchiveFormValues(r, vars)
	if err != nil {
		return err
	}

	noOverwriteDirNonDir := httputils.BoolValue(r, "noOverwriteDirNonDir")
	copyUIDGID := httputils.BoolValue(r, "copyUIDGID")

	return s.backend.ContainerExtractToDir(v.Name, v.Path, copyUIDGID, noOverwriteDirNonDir, r.Body)
}
