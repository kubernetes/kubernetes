package eabac

import (
	"fmt"
	"io/ioutil"
	"os"
	"strings"
	"sync/atomic"
	"time"

	etcd "github.com/coreos/etcd/client"
	"github.com/golang/glog"
	"golang.org/x/net/context"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/auth/authorizer/abac"
	"k8s.io/kubernetes/pkg/util/wait"
)

const (
	retryFailureTime = 10 * time.Second
)

type EABAC struct {
	Auth atomic.Value
}

// path is in the following format:
//   http[s]?://<ip>:<port>[,http[s]?://<ip>:<port>]*@/path/to/policy/file
// e.g., http://1.2.3.4:4001,1,2,3,5:4001@/abac-policy
func New(path string) (*EABAC, error) {

	var auth authorizer.Authorizer
	eabac := &EABAC{}

	arr := strings.Split(path, "@")
	if len(arr) != 2 {
		return nil, fmt.Errorf("eabac authz plugin cannot parse policy file path: %s", path)
	}

	serverList := strings.Split(arr[0], ",")
	key := arr[1]

	cfg := etcd.Config{
		Endpoints:               serverList,
		Transport:               etcd.DefaultTransport,
		HeaderTimeoutPerRequest: time.Second,
	}

	client, err := etcd.New(cfg)
	if err != nil {
		return nil, err
	}

	kapi := etcd.NewKeysAPI(client)
	if resp, err := kapi.Get(context.Background(), key, nil); err != nil {
		return nil, err
	} else {
		tmpFile, err := ioutil.TempFile("", "")
		if err != nil {
			return nil, err
		}
		tmpFileName := tmpFile.Name()
		defer os.Remove(tmpFileName)

		ioutil.WriteFile(tmpFileName, []byte(resp.Node.Value), 0444)
		auth, err = abac.NewFromFile(tmpFileName)
		if err != nil {
			return nil, err
		}

		go wait.Forever(func() {
			watcher := kapi.Watcher(key, nil)
			for {
				// done channel will be closed when no error has occured
				// otherwise, error is sent to the channel
				done := make(chan error)
				go func() {
					resp, err := watcher.Next(context.Background())
					if err != nil {
						done <- err
					} else {
						glog.V(1).Infof("detected ABAC policy has been changed in etcd, reloading...")
						tmpFile, err := ioutil.TempFile("", "")
						if err != nil {
							done <- err
						} else {
							defer os.Remove(tmpFileName)
							tmpFileName := tmpFile.Name()
							ioutil.WriteFile(tmpFileName, []byte(resp.Node.Value), 0444)
							tmpAuth, err := abac.NewFromFile(tmpFileName)
							if err != nil {
								done <- err
							} else {
								eabac.Auth.Store(tmpAuth)
								close(done)
							}
						}
					}
				}()
				select {
				case err, ok := <-done:
					if ok {
						glog.Warningf("error detected: %v, retry in %v", err, retryFailureTime)
						time.Sleep(retryFailureTime)
					}
				}
			}
		}, time.Second)
	}

	eabac.Auth.Store(auth)
	return eabac, nil
}

func (e *EABAC) Authorize(a authorizer.Attributes) error {
	auth, ok := e.Auth.Load().(authorizer.Authorizer)
	if !ok {
		return fmt.Errorf("unexpected error: cannot convert data to authorizer.Authorizer type")
	}
	return auth.Authorize(a)
}
