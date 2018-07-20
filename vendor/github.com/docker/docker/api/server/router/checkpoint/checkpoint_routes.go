package checkpoint

import (
	"encoding/json"
	"net/http"

	"github.com/docker/docker/api/server/httputils"
	"github.com/docker/docker/api/types"
	"golang.org/x/net/context"
)

func (s *checkpointRouter) postContainerCheckpoint(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	var options types.CheckpointCreateOptions

	decoder := json.NewDecoder(r.Body)
	if err := decoder.Decode(&options); err != nil {
		return err
	}

	err := s.backend.CheckpointCreate(vars["name"], options)
	if err != nil {
		return err
	}

	w.WriteHeader(http.StatusCreated)
	return nil
}

func (s *checkpointRouter) getContainerCheckpoints(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	checkpoints, err := s.backend.CheckpointList(vars["name"], types.CheckpointListOptions{
		CheckpointDir: r.Form.Get("dir"),
	})

	if err != nil {
		return err
	}

	return httputils.WriteJSON(w, http.StatusOK, checkpoints)
}

func (s *checkpointRouter) deleteContainerCheckpoint(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	err := s.backend.CheckpointDelete(vars["name"], types.CheckpointDeleteOptions{
		CheckpointDir: r.Form.Get("dir"),
		CheckpointID:  vars["checkpoint"],
	})

	if err != nil {
		return err
	}

	w.WriteHeader(http.StatusNoContent)
	return nil
}
