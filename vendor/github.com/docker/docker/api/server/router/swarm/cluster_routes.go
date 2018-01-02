package swarm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"

	"github.com/docker/docker/api/errors"
	"github.com/docker/docker/api/server/httputils"
	basictypes "github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/backend"
	"github.com/docker/docker/api/types/filters"
	types "github.com/docker/docker/api/types/swarm"
	"github.com/docker/docker/api/types/versions"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

func (sr *swarmRouter) initCluster(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	var req types.InitRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		return err
	}
	nodeID, err := sr.backend.Init(req)
	if err != nil {
		logrus.Errorf("Error initializing swarm: %v", err)
		return err
	}
	return httputils.WriteJSON(w, http.StatusOK, nodeID)
}

func (sr *swarmRouter) joinCluster(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	var req types.JoinRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		return err
	}
	return sr.backend.Join(req)
}

func (sr *swarmRouter) leaveCluster(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	force := httputils.BoolValue(r, "force")
	return sr.backend.Leave(force)
}

func (sr *swarmRouter) inspectCluster(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	swarm, err := sr.backend.Inspect()
	if err != nil {
		logrus.Errorf("Error getting swarm: %v", err)
		return err
	}

	return httputils.WriteJSON(w, http.StatusOK, swarm)
}

func (sr *swarmRouter) updateCluster(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	var swarm types.Spec
	if err := json.NewDecoder(r.Body).Decode(&swarm); err != nil {
		return err
	}

	rawVersion := r.URL.Query().Get("version")
	version, err := strconv.ParseUint(rawVersion, 10, 64)
	if err != nil {
		err := fmt.Errorf("invalid swarm version '%s': %v", rawVersion, err)
		return errors.NewBadRequestError(err)
	}

	var flags types.UpdateFlags

	if value := r.URL.Query().Get("rotateWorkerToken"); value != "" {
		rot, err := strconv.ParseBool(value)
		if err != nil {
			err := fmt.Errorf("invalid value for rotateWorkerToken: %s", value)
			return errors.NewBadRequestError(err)
		}

		flags.RotateWorkerToken = rot
	}

	if value := r.URL.Query().Get("rotateManagerToken"); value != "" {
		rot, err := strconv.ParseBool(value)
		if err != nil {
			err := fmt.Errorf("invalid value for rotateManagerToken: %s", value)
			return errors.NewBadRequestError(err)
		}

		flags.RotateManagerToken = rot
	}

	if value := r.URL.Query().Get("rotateManagerUnlockKey"); value != "" {
		rot, err := strconv.ParseBool(value)
		if err != nil {
			return errors.NewBadRequestError(fmt.Errorf("invalid value for rotateManagerUnlockKey: %s", value))
		}

		flags.RotateManagerUnlockKey = rot
	}

	if err := sr.backend.Update(version, swarm, flags); err != nil {
		logrus.Errorf("Error configuring swarm: %v", err)
		return err
	}
	return nil
}

func (sr *swarmRouter) unlockCluster(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	var req types.UnlockRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		return err
	}

	if err := sr.backend.UnlockSwarm(req); err != nil {
		logrus.Errorf("Error unlocking swarm: %v", err)
		return err
	}
	return nil
}

func (sr *swarmRouter) getUnlockKey(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	unlockKey, err := sr.backend.GetUnlockKey()
	if err != nil {
		logrus.WithError(err).Errorf("Error retrieving swarm unlock key")
		return err
	}

	return httputils.WriteJSON(w, http.StatusOK, &basictypes.SwarmUnlockKeyResponse{
		UnlockKey: unlockKey,
	})
}

func (sr *swarmRouter) getServices(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}
	filter, err := filters.FromParam(r.Form.Get("filters"))
	if err != nil {
		return err
	}

	services, err := sr.backend.GetServices(basictypes.ServiceListOptions{Filters: filter})
	if err != nil {
		logrus.Errorf("Error getting services: %v", err)
		return err
	}

	return httputils.WriteJSON(w, http.StatusOK, services)
}

func (sr *swarmRouter) getService(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	var insertDefaults bool
	if value := r.URL.Query().Get("insertDefaults"); value != "" {
		var err error
		insertDefaults, err = strconv.ParseBool(value)
		if err != nil {
			err := fmt.Errorf("invalid value for insertDefaults: %s", value)
			return errors.NewBadRequestError(err)
		}
	}

	service, err := sr.backend.GetService(vars["id"], insertDefaults)
	if err != nil {
		logrus.Errorf("Error getting service %s: %v", vars["id"], err)
		return err
	}

	return httputils.WriteJSON(w, http.StatusOK, service)
}

func (sr *swarmRouter) createService(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	var service types.ServiceSpec
	if err := json.NewDecoder(r.Body).Decode(&service); err != nil {
		return err
	}

	// Get returns "" if the header does not exist
	encodedAuth := r.Header.Get("X-Registry-Auth")
	cliVersion := r.Header.Get("version")
	queryRegistry := false
	if cliVersion != "" && versions.LessThan(cliVersion, "1.30") {
		queryRegistry = true
	}

	resp, err := sr.backend.CreateService(service, encodedAuth, queryRegistry)
	if err != nil {
		logrus.Errorf("Error creating service %s: %v", service.Name, err)
		return err
	}

	return httputils.WriteJSON(w, http.StatusCreated, resp)
}

func (sr *swarmRouter) updateService(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	var service types.ServiceSpec
	if err := json.NewDecoder(r.Body).Decode(&service); err != nil {
		return err
	}

	rawVersion := r.URL.Query().Get("version")
	version, err := strconv.ParseUint(rawVersion, 10, 64)
	if err != nil {
		err := fmt.Errorf("invalid service version '%s': %v", rawVersion, err)
		return errors.NewBadRequestError(err)
	}

	var flags basictypes.ServiceUpdateOptions

	// Get returns "" if the header does not exist
	flags.EncodedRegistryAuth = r.Header.Get("X-Registry-Auth")
	flags.RegistryAuthFrom = r.URL.Query().Get("registryAuthFrom")
	flags.Rollback = r.URL.Query().Get("rollback")
	cliVersion := r.Header.Get("version")
	queryRegistry := false
	if cliVersion != "" && versions.LessThan(cliVersion, "1.30") {
		queryRegistry = true
	}

	resp, err := sr.backend.UpdateService(vars["id"], version, service, flags, queryRegistry)
	if err != nil {
		logrus.Errorf("Error updating service %s: %v", vars["id"], err)
		return err
	}
	return httputils.WriteJSON(w, http.StatusOK, resp)
}

func (sr *swarmRouter) removeService(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := sr.backend.RemoveService(vars["id"]); err != nil {
		logrus.Errorf("Error removing service %s: %v", vars["id"], err)
		return err
	}
	return nil
}

func (sr *swarmRouter) getTaskLogs(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	// make a selector to pass to the helper function
	selector := &backend.LogSelector{
		Tasks: []string{vars["id"]},
	}
	return sr.swarmLogs(ctx, w, r, selector)
}

func (sr *swarmRouter) getServiceLogs(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	// make a selector to pass to the helper function
	selector := &backend.LogSelector{
		Services: []string{vars["id"]},
	}
	return sr.swarmLogs(ctx, w, r, selector)
}

func (sr *swarmRouter) getNodes(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}
	filter, err := filters.FromParam(r.Form.Get("filters"))
	if err != nil {
		return err
	}

	nodes, err := sr.backend.GetNodes(basictypes.NodeListOptions{Filters: filter})
	if err != nil {
		logrus.Errorf("Error getting nodes: %v", err)
		return err
	}

	return httputils.WriteJSON(w, http.StatusOK, nodes)
}

func (sr *swarmRouter) getNode(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	node, err := sr.backend.GetNode(vars["id"])
	if err != nil {
		logrus.Errorf("Error getting node %s: %v", vars["id"], err)
		return err
	}

	return httputils.WriteJSON(w, http.StatusOK, node)
}

func (sr *swarmRouter) updateNode(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	var node types.NodeSpec
	if err := json.NewDecoder(r.Body).Decode(&node); err != nil {
		return err
	}

	rawVersion := r.URL.Query().Get("version")
	version, err := strconv.ParseUint(rawVersion, 10, 64)
	if err != nil {
		err := fmt.Errorf("invalid node version '%s': %v", rawVersion, err)
		return errors.NewBadRequestError(err)
	}

	if err := sr.backend.UpdateNode(vars["id"], version, node); err != nil {
		logrus.Errorf("Error updating node %s: %v", vars["id"], err)
		return err
	}
	return nil
}

func (sr *swarmRouter) removeNode(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	force := httputils.BoolValue(r, "force")

	if err := sr.backend.RemoveNode(vars["id"], force); err != nil {
		logrus.Errorf("Error removing node %s: %v", vars["id"], err)
		return err
	}
	return nil
}

func (sr *swarmRouter) getTasks(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}
	filter, err := filters.FromParam(r.Form.Get("filters"))
	if err != nil {
		return err
	}

	tasks, err := sr.backend.GetTasks(basictypes.TaskListOptions{Filters: filter})
	if err != nil {
		logrus.Errorf("Error getting tasks: %v", err)
		return err
	}

	return httputils.WriteJSON(w, http.StatusOK, tasks)
}

func (sr *swarmRouter) getTask(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	task, err := sr.backend.GetTask(vars["id"])
	if err != nil {
		logrus.Errorf("Error getting task %s: %v", vars["id"], err)
		return err
	}

	return httputils.WriteJSON(w, http.StatusOK, task)
}

func (sr *swarmRouter) getSecrets(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}
	filters, err := filters.FromParam(r.Form.Get("filters"))
	if err != nil {
		return err
	}

	secrets, err := sr.backend.GetSecrets(basictypes.SecretListOptions{Filters: filters})
	if err != nil {
		return err
	}

	return httputils.WriteJSON(w, http.StatusOK, secrets)
}

func (sr *swarmRouter) createSecret(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	var secret types.SecretSpec
	if err := json.NewDecoder(r.Body).Decode(&secret); err != nil {
		return err
	}

	id, err := sr.backend.CreateSecret(secret)
	if err != nil {
		return err
	}

	return httputils.WriteJSON(w, http.StatusCreated, &basictypes.SecretCreateResponse{
		ID: id,
	})
}

func (sr *swarmRouter) removeSecret(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := sr.backend.RemoveSecret(vars["id"]); err != nil {
		return err
	}
	w.WriteHeader(http.StatusNoContent)

	return nil
}

func (sr *swarmRouter) getSecret(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	secret, err := sr.backend.GetSecret(vars["id"])
	if err != nil {
		return err
	}

	return httputils.WriteJSON(w, http.StatusOK, secret)
}

func (sr *swarmRouter) updateSecret(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	var secret types.SecretSpec
	if err := json.NewDecoder(r.Body).Decode(&secret); err != nil {
		return errors.NewBadRequestError(err)
	}

	rawVersion := r.URL.Query().Get("version")
	version, err := strconv.ParseUint(rawVersion, 10, 64)
	if err != nil {
		return errors.NewBadRequestError(fmt.Errorf("invalid secret version"))
	}

	id := vars["id"]
	if err := sr.backend.UpdateSecret(id, version, secret); err != nil {
		return err
	}

	return nil
}

func (sr *swarmRouter) getConfigs(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}
	filters, err := filters.FromParam(r.Form.Get("filters"))
	if err != nil {
		return err
	}

	configs, err := sr.backend.GetConfigs(basictypes.ConfigListOptions{Filters: filters})
	if err != nil {
		return err
	}

	return httputils.WriteJSON(w, http.StatusOK, configs)
}

func (sr *swarmRouter) createConfig(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	var config types.ConfigSpec
	if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
		return err
	}

	id, err := sr.backend.CreateConfig(config)
	if err != nil {
		return err
	}

	return httputils.WriteJSON(w, http.StatusCreated, &basictypes.ConfigCreateResponse{
		ID: id,
	})
}

func (sr *swarmRouter) removeConfig(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := sr.backend.RemoveConfig(vars["id"]); err != nil {
		return err
	}
	w.WriteHeader(http.StatusNoContent)

	return nil
}

func (sr *swarmRouter) getConfig(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	config, err := sr.backend.GetConfig(vars["id"])
	if err != nil {
		return err
	}

	return httputils.WriteJSON(w, http.StatusOK, config)
}

func (sr *swarmRouter) updateConfig(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	var config types.ConfigSpec
	if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
		return errors.NewBadRequestError(err)
	}

	rawVersion := r.URL.Query().Get("version")
	version, err := strconv.ParseUint(rawVersion, 10, 64)
	if err != nil {
		return errors.NewBadRequestError(fmt.Errorf("invalid config version"))
	}

	id := vars["id"]
	if err := sr.backend.UpdateConfig(id, version, config); err != nil {
		return err
	}

	return nil
}
