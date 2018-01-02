package coprhd

import (
	"crypto/tls"
	"fmt"
	"net/http"
	"net/url"

	"gopkg.in/jmcvetta/napping.v3"

	"go.pedge.io/dlog"

	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/volume"
	"github.com/libopenstorage/openstorage/volume/drivers/common"
	"github.com/portworx/kvdb"
)

const (
	// Name of the driver
	Name = "coprhd"
	// Type of the driver
	Type = api.DriverType_DRIVER_TYPE_BLOCK

	// LoginUri path to create a authentication token
	loginUri = "login.json"
	// LoginUri path to create volume
	createVolumeUri = "block/volumes.json"
)

// ApiError represents the default api error code
type ApiError struct {
	Code        string `json:"code"`
	Retryable   string `json:"retryable"`
	Description string `json:"description"`
	Details     string `json:"details"`
}

// CreateVolumeArgs represents the json parameters for the create volume REST call
type CreateVolumeArgs struct {
	ConsistencyGroup string `json:"consistency_group"`
	Count            int    `json:"count"`
	Name             string `json:"name"`
	Project          string `json:"project"`
	Size             string `json:"size"`
	VArray           string `json:"varray"`
	VPool            string `json:"vpool"`
}

// CreateVolumeReply is the reply from the create volume REST call
type CreateVolumeReply struct {
	Task []struct {
		Resource struct {
			Name string `json:"name"`
			Id   string `json:"id"`
		} `json:"resource"`
	} `json:"task"`
}

type driver struct {
	volume.IODriver
	volume.StoreEnumerator
	volume.StatsDriver
	consistencyGroup string
	project          string
	varray           string
	vpool            string
	url              string
	httpClient       *http.Client
	creds            *url.Userinfo
}

// Init initializes the driver
func Init(params map[string]string) (volume.VolumeDriver, error) {
	restUrl, ok := params["restUrl"]
	if !ok {
		return nil, fmt.Errorf("rest api 'url' configuration parameter must be set")
	}

	user, ok := params["user"]
	if !ok {
		return nil, fmt.Errorf("rest auth 'user' must be set")
	}

	pass, ok := params["password"]
	if !ok {
		return nil, fmt.Errorf("rest auth 'password' must be set")
	}

	consistencyGroup, ok := params["consistency_group"]
	if !ok {
		return nil, fmt.Errorf("'consistency_group' configuration parameter must be set")
	}

	project, ok := params["project"]
	if !ok {
		return nil, fmt.Errorf("'project' configuration parameter must be set")
	}

	varray, ok := params["varray"]
	if !ok {
		return nil, fmt.Errorf("'varray' configuration parameter must be set")
	}

	vpool, ok := params["vpool"]
	if !ok {
		return nil, fmt.Errorf("'vpool' configuration parameter must be set")
	}

	d := &driver{
		IODriver:         volume.IONotSupported,
		StoreEnumerator:  common.NewDefaultStoreEnumerator(Name, kvdb.Instance()),
		StatsDriver:      volume.StatsNotSupported,
		consistencyGroup: consistencyGroup,
		project:          project,
		varray:           varray,
		vpool:            vpool,
		url:              restUrl,
		creds:            url.UserPassword(user, pass),
		httpClient: &http.Client{
			Transport: &http.Transport{
				TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
			},
		},
	}

	return d, nil
}

func (d *driver) Name() string {
	return Name
}

func (d *driver) Type() api.DriverType {
	return Type
}

func (d *driver) Create(
	locator *api.VolumeLocator,
	source *api.Source,
	spec *api.VolumeSpec,
) (string, error) {

	s, err := d.getAuthSession()

	if err != nil {
		dlog.Errorf("Failed to create session: %s", err.Error())
		return "", err
	}

	e := ApiError{}

	res := &CreateVolumeReply{}

	sz := int64(spec.Size / (1024 * 1024 * 1000))

	payload := CreateVolumeArgs{
		d.consistencyGroup,        // ConsistencyGroup
		1,                         // Count
		locator.Name,              // Name
		d.project,                 // Project
		fmt.Sprintf("%.6fGB", sz), // Volume Size
		d.varray,                  // Virtual Block Array
		d.vpool,                   // Virtual Block Pool
	}

	url := d.url + createVolumeUri

	resp, err := s.Post(url, &payload, res, &e)

	if resp.Status() != http.StatusAccepted {

		return "", fmt.Errorf("Failed to create volume: %s", resp.Status())
	}

	return res.Task[0].Resource.Id, err
}

func (d *driver) Delete(volumeID string) error {
	return nil
}

func (d *driver) Attach(volumeID string, attachOptions map[string]string) (path string, err error) {
	return "", nil
}

func (d *driver) MountedAt(mountpath string) string {
	return ""
}

func (d *driver) Detach(volumeID string, unmountBeforeDetach bool) error {
	return nil
}

func (d *driver) Mount(volumeID string, mountpath string) error {
	return nil
}

func (d *driver) Unmount(volumeID string, mountpath string) error {

	return nil
}

func (d *driver) Set(
	volumeID string,
	locator *api.VolumeLocator,
	spec *api.VolumeSpec) error {
	return volume.ErrNotSupported
}

func (d *driver) Shutdown() {
	dlog.Infof("%s Shutting down", Name)
}

func (d *driver) Snapshot(
	volumeID string,
	readonly bool,
	locator *api.VolumeLocator) (string, error) {
	return "", nil
}

func (d *driver) Restore(volumeID string, snapID string) error {
	return volume.ErrNotSupported
}

func (d *driver) Status() [][2]string {
	return [][2]string{}
}

// getAuthSession returns an authenticated API Session
func (d *driver) getAuthSession() (session *napping.Session, err error) {
	e := ApiError{}

	s := napping.Session{
		Userinfo: d.creds,
		Client:   d.httpClient,
	}

	url := d.url + loginUri

	resp, err := s.Get(url, nil, nil, &e)

	if err != nil {
		return
	}

	token := resp.HttpResponse().Header.Get("X-SDS-AUTH-TOKEN")

	h := http.Header{}

	h.Set("X-SDS-AUTH-TOKEN", token)

	session = &napping.Session{
		Client: d.httpClient,
		Header: &h,
	}

	return
}
