package gcplogs

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/docker/docker/daemon/logger"

	"cloud.google.com/go/compute/metadata"
	"cloud.google.com/go/logging"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
	mrpb "google.golang.org/genproto/googleapis/api/monitoredres"
)

const (
	name = "gcplogs"

	projectOptKey  = "gcp-project"
	logLabelsKey   = "labels"
	logEnvKey      = "env"
	logEnvRegexKey = "env-regex"
	logCmdKey      = "gcp-log-cmd"
	logZoneKey     = "gcp-meta-zone"
	logNameKey     = "gcp-meta-name"
	logIDKey       = "gcp-meta-id"
)

var (
	// The number of logs the gcplogs driver has dropped.
	droppedLogs uint64

	onGCE bool

	// instance metadata populated from the metadata server if available
	projectID    string
	zone         string
	instanceName string
	instanceID   string
)

func init() {

	if err := logger.RegisterLogDriver(name, New); err != nil {
		logrus.Fatal(err)
	}

	if err := logger.RegisterLogOptValidator(name, ValidateLogOpts); err != nil {
		logrus.Fatal(err)
	}
}

type gcplogs struct {
	logger    *logging.Logger
	instance  *instanceInfo
	container *containerInfo
}

type dockerLogEntry struct {
	Instance  *instanceInfo  `json:"instance,omitempty"`
	Container *containerInfo `json:"container,omitempty"`
	Message   string         `json:"message,omitempty"`
}

type instanceInfo struct {
	Zone string `json:"zone,omitempty"`
	Name string `json:"name,omitempty"`
	ID   string `json:"id,omitempty"`
}

type containerInfo struct {
	Name      string            `json:"name,omitempty"`
	ID        string            `json:"id,omitempty"`
	ImageName string            `json:"imageName,omitempty"`
	ImageID   string            `json:"imageId,omitempty"`
	Created   time.Time         `json:"created,omitempty"`
	Command   string            `json:"command,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

var initGCPOnce sync.Once

func initGCP() {
	initGCPOnce.Do(func() {
		onGCE = metadata.OnGCE()
		if onGCE {
			// These will fail on instances if the metadata service is
			// down or the client is compiled with an API version that
			// has been removed. Since these are not vital, let's ignore
			// them and make their fields in the dockerLogEntry ,omitempty
			projectID, _ = metadata.ProjectID()
			zone, _ = metadata.Zone()
			instanceName, _ = metadata.InstanceName()
			instanceID, _ = metadata.InstanceID()
		}
	})
}

// New creates a new logger that logs to Google Cloud Logging using the application
// default credentials.
//
// See https://developers.google.com/identity/protocols/application-default-credentials
func New(info logger.Info) (logger.Logger, error) {
	initGCP()

	var project string
	if projectID != "" {
		project = projectID
	}
	if projectID, found := info.Config[projectOptKey]; found {
		project = projectID
	}
	if project == "" {
		return nil, fmt.Errorf("No project was specified and couldn't read project from the metadata server. Please specify a project")
	}

	// Issue #29344: gcplogs segfaults (static binary)
	// If HOME is not set, logging.NewClient() will call os/user.Current() via oauth2/google.
	// However, in static binary, os/user.Current() leads to segfault due to a glibc issue that won't be fixed
	// in a short term. (golang/go#13470, https://sourceware.org/bugzilla/show_bug.cgi?id=19341)
	// So we forcibly set HOME so as to avoid call to os/user/Current()
	if err := ensureHomeIfIAmStatic(); err != nil {
		return nil, err
	}

	c, err := logging.NewClient(context.Background(), project)
	if err != nil {
		return nil, err
	}
	var instanceResource *instanceInfo
	if onGCE {
		instanceResource = &instanceInfo{
			Zone: zone,
			Name: instanceName,
			ID:   instanceID,
		}
	} else if info.Config[logZoneKey] != "" || info.Config[logNameKey] != "" || info.Config[logIDKey] != "" {
		instanceResource = &instanceInfo{
			Zone: info.Config[logZoneKey],
			Name: info.Config[logNameKey],
			ID:   info.Config[logIDKey],
		}
	}

	options := []logging.LoggerOption{}
	if instanceResource != nil {
		vmMrpb := logging.CommonResource(
			&mrpb.MonitoredResource{
				Type: "gce_instance",
				Labels: map[string]string{
					"instance_id": instanceResource.ID,
					"zone":        instanceResource.Zone,
				},
			},
		)
		options = []logging.LoggerOption{vmMrpb}
	}
	lg := c.Logger("gcplogs-docker-driver", options...)

	if err := c.Ping(context.Background()); err != nil {
		return nil, fmt.Errorf("unable to connect or authenticate with Google Cloud Logging: %v", err)
	}

	extraAttributes, err := info.ExtraAttributes(nil)
	if err != nil {
		return nil, err
	}

	l := &gcplogs{
		logger: lg,
		container: &containerInfo{
			Name:      info.ContainerName,
			ID:        info.ContainerID,
			ImageName: info.ContainerImageName,
			ImageID:   info.ContainerImageID,
			Created:   info.ContainerCreated,
			Metadata:  extraAttributes,
		},
	}

	if info.Config[logCmdKey] == "true" {
		l.container.Command = info.Command()
	}

	if instanceResource != nil {
		l.instance = instanceResource
	}

	// The logger "overflows" at a rate of 10,000 logs per second and this
	// overflow func is called. We want to surface the error to the user
	// without overly spamming /var/log/docker.log so we log the first time
	// we overflow and every 1000th time after.
	c.OnError = func(err error) {
		if err == logging.ErrOverflow {
			if i := atomic.AddUint64(&droppedLogs, 1); i%1000 == 1 {
				logrus.Errorf("gcplogs driver has dropped %v logs", i)
			}
		} else {
			logrus.Error(err)
		}
	}

	return l, nil
}

// ValidateLogOpts validates the opts passed to the gcplogs driver. Currently, the gcplogs
// driver doesn't take any arguments.
func ValidateLogOpts(cfg map[string]string) error {
	for k := range cfg {
		switch k {
		case projectOptKey, logLabelsKey, logEnvKey, logEnvRegexKey, logCmdKey, logZoneKey, logNameKey, logIDKey:
		default:
			return fmt.Errorf("%q is not a valid option for the gcplogs driver", k)
		}
	}
	return nil
}

func (l *gcplogs) Log(m *logger.Message) error {
	message := string(m.Line)
	ts := m.Timestamp
	logger.PutMessage(m)

	l.logger.Log(logging.Entry{
		Timestamp: ts,
		Payload: &dockerLogEntry{
			Instance:  l.instance,
			Container: l.container,
			Message:   message,
		},
	})
	return nil
}

func (l *gcplogs) Close() error {
	l.logger.Flush()
	return nil
}

func (l *gcplogs) Name() string {
	return name
}
