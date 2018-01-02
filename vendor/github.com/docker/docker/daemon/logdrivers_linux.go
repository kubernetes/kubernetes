package daemon

import (
	// Importing packages here only to make sure their init gets called and
	// therefore they register themselves to the logdriver factory.
	_ "github.com/docker/docker/daemon/logger/awslogs"
	_ "github.com/docker/docker/daemon/logger/fluentd"
	_ "github.com/docker/docker/daemon/logger/gcplogs"
	_ "github.com/docker/docker/daemon/logger/gelf"
	_ "github.com/docker/docker/daemon/logger/journald"
	_ "github.com/docker/docker/daemon/logger/jsonfilelog"
	_ "github.com/docker/docker/daemon/logger/logentries"
	_ "github.com/docker/docker/daemon/logger/splunk"
	_ "github.com/docker/docker/daemon/logger/syslog"
)
