package container

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"

	"github.com/docker/docker/api/server/httputils"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/versions"
	"github.com/docker/docker/pkg/stdcopy"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

func (s *containerRouter) getExecByID(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	eConfig, err := s.backend.ContainerExecInspect(vars["id"])
	if err != nil {
		return err
	}

	return httputils.WriteJSON(w, http.StatusOK, eConfig)
}

func (s *containerRouter) postContainerExecCreate(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}
	if err := httputils.CheckForJSON(r); err != nil {
		return err
	}
	name := vars["name"]

	execConfig := &types.ExecConfig{}
	if err := json.NewDecoder(r.Body).Decode(execConfig); err != nil {
		return err
	}

	if len(execConfig.Cmd) == 0 {
		return fmt.Errorf("No exec command specified")
	}

	// Register an instance of Exec in container.
	id, err := s.backend.ContainerExecCreate(name, execConfig)
	if err != nil {
		logrus.Errorf("Error setting up exec command in container %s: %v", name, err)
		return err
	}

	return httputils.WriteJSON(w, http.StatusCreated, &types.IDResponse{
		ID: id,
	})
}

// TODO(vishh): Refactor the code to avoid having to specify stream config as part of both create and start.
func (s *containerRouter) postContainerExecStart(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	version := httputils.VersionFromContext(ctx)
	if versions.GreaterThan(version, "1.21") {
		if err := httputils.CheckForJSON(r); err != nil {
			return err
		}
	}

	var (
		execName                  = vars["name"]
		stdin, inStream           io.ReadCloser
		stdout, stderr, outStream io.Writer
	)

	execStartCheck := &types.ExecStartCheck{}
	if err := json.NewDecoder(r.Body).Decode(execStartCheck); err != nil {
		return err
	}

	if exists, err := s.backend.ExecExists(execName); !exists {
		return err
	}

	if !execStartCheck.Detach {
		var err error
		// Setting up the streaming http interface.
		inStream, outStream, err = httputils.HijackConnection(w)
		if err != nil {
			return err
		}
		defer httputils.CloseStreams(inStream, outStream)

		if _, ok := r.Header["Upgrade"]; ok {
			fmt.Fprint(outStream, "HTTP/1.1 101 UPGRADED\r\nContent-Type: application/vnd.docker.raw-stream\r\nConnection: Upgrade\r\nUpgrade: tcp\r\n")
		} else {
			fmt.Fprint(outStream, "HTTP/1.1 200 OK\r\nContent-Type: application/vnd.docker.raw-stream\r\n")
		}

		// copy headers that were removed as part of hijack
		if err := w.Header().WriteSubset(outStream, nil); err != nil {
			return err
		}
		fmt.Fprint(outStream, "\r\n")

		stdin = inStream
		stdout = outStream
		if !execStartCheck.Tty {
			stderr = stdcopy.NewStdWriter(outStream, stdcopy.Stderr)
			stdout = stdcopy.NewStdWriter(outStream, stdcopy.Stdout)
		}
	}

	// Now run the user process in container.
	// Maybe we should we pass ctx here if we're not detaching?
	if err := s.backend.ContainerExecStart(context.Background(), execName, stdin, stdout, stderr); err != nil {
		if execStartCheck.Detach {
			return err
		}
		stdout.Write([]byte(err.Error() + "\r\n"))
		logrus.Errorf("Error running exec in container: %v", err)
	}
	return nil
}

func (s *containerRouter) postContainerExecResize(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}
	height, err := strconv.Atoi(r.Form.Get("h"))
	if err != nil {
		return err
	}
	width, err := strconv.Atoi(r.Form.Get("w"))
	if err != nil {
		return err
	}

	return s.backend.ContainerExecResize(vars["name"], height, width)
}
