// +build linux solaris

package libcontainerd

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	containerd "github.com/containerd/containerd/api/grpc/types"
	"github.com/docker/docker/pkg/idtools"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

func (clnt *client) prepareBundleDir(uid, gid int) (string, error) {
	root, err := filepath.Abs(clnt.remote.stateDir)
	if err != nil {
		return "", err
	}
	if uid == 0 && gid == 0 {
		return root, nil
	}
	p := string(filepath.Separator)
	for _, d := range strings.Split(root, string(filepath.Separator))[1:] {
		p = filepath.Join(p, d)
		fi, err := os.Stat(p)
		if err != nil && !os.IsNotExist(err) {
			return "", err
		}
		if os.IsNotExist(err) || fi.Mode()&1 == 0 {
			p = fmt.Sprintf("%s.%d.%d", p, uid, gid)
			if err := idtools.MkdirAndChown(p, 0700, idtools.IDPair{uid, gid}); err != nil && !os.IsExist(err) {
				return "", err
			}
		}
	}
	return p, nil
}

func (clnt *client) Create(containerID string, checkpoint string, checkpointDir string, spec specs.Spec, attachStdio StdioCallback, options ...CreateOption) (err error) {
	clnt.lock(containerID)
	defer clnt.unlock(containerID)

	if _, err := clnt.getContainer(containerID); err == nil {
		return fmt.Errorf("Container %s is already active", containerID)
	}

	uid, gid, err := getRootIDs(specs.Spec(spec))
	if err != nil {
		return err
	}
	dir, err := clnt.prepareBundleDir(uid, gid)
	if err != nil {
		return err
	}

	container := clnt.newContainer(filepath.Join(dir, containerID), options...)
	if err := container.clean(); err != nil {
		return err
	}

	defer func() {
		if err != nil {
			container.clean()
			clnt.deleteContainer(containerID)
		}
	}()

	if err := idtools.MkdirAllAndChown(container.dir, 0700, idtools.IDPair{uid, gid}); err != nil && !os.IsExist(err) {
		return err
	}

	f, err := os.Create(filepath.Join(container.dir, configFilename))
	if err != nil {
		return err
	}
	defer f.Close()
	if err := json.NewEncoder(f).Encode(spec); err != nil {
		return err
	}
	return container.start(&spec, checkpoint, checkpointDir, attachStdio)
}

func (clnt *client) Signal(containerID string, sig int) error {
	clnt.lock(containerID)
	defer clnt.unlock(containerID)
	_, err := clnt.remote.apiClient.Signal(context.Background(), &containerd.SignalRequest{
		Id:     containerID,
		Pid:    InitFriendlyName,
		Signal: uint32(sig),
	})
	return err
}

func (clnt *client) newContainer(dir string, options ...CreateOption) *container {
	container := &container{
		containerCommon: containerCommon{
			process: process{
				dir: dir,
				processCommon: processCommon{
					containerID:  filepath.Base(dir),
					client:       clnt,
					friendlyName: InitFriendlyName,
				},
			},
			processes: make(map[string]*process),
		},
	}
	for _, option := range options {
		if err := option.Apply(container); err != nil {
			logrus.Errorf("libcontainerd: newContainer(): %v", err)
		}
	}
	return container
}

type exitNotifier struct {
	id     string
	client *client
	c      chan struct{}
	once   sync.Once
}

func (en *exitNotifier) close() {
	en.once.Do(func() {
		close(en.c)
		en.client.mapMutex.Lock()
		if en == en.client.exitNotifiers[en.id] {
			delete(en.client.exitNotifiers, en.id)
		}
		en.client.mapMutex.Unlock()
	})
}
func (en *exitNotifier) wait() <-chan struct{} {
	return en.c
}
