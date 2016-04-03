Runnning Heapster on CoreOS
================================

Heapster enables cluster monitoring in a CoreOS cluster using [cAdvisor](https://github.com/google/cadvisor). 

The assumptions here are that influxdb, heapster, and grafana will run on the same machine, which is easily accomplished via fleet files.

**Step 1: Start cAdvisor on all hosts by default**

This can be accomplished with either a cloud config entry:
```yaml
    - name: cadvisor.service
      runtime: true
      command: start
      content: |
        [Unit]
        Description=Analyzes resource usage and performance characteristics of running containers.
        After=docker.service
        Requires=docker.service

        [Service]
        Restart=always
        ExecStartPre=/usr/bin/docker pull google/cadvisor:latest
        ExecStartPre=-/bin/bash -c "docker inspect cadvisor >/dev/null 2>&1 && docker rm -f cadvisor || true"
        ExecStart=/usr/bin/docker run --volume=/var/run:/var/run:rw --volume=/sys:/sys:ro --volume=/var/lib/docker/:/var/lib/docker:ro --volume=/:/rootfs:ro --publish=8080:8080 --name=cadvisor google/cadvisor:latest
        ExecStop=/usr/bin/docker rm -f cadvisor
```

or with a global fleet file:

```ini
[Unit]
Description=Analyzes resource usage and performance characteristics of running containers.
After=docker.service
Requires=docker.service

[Service]
Restart=always
ExecStartPre=/usr/bin/docker pull google/cadvisor:latest
ExecStartPre=-/bin/bash -c "docker inspect cadvisor >/dev/null 2>&1 && docker rm -f cadvisor || true"
ExecStart=/usr/bin/docker run --volume=/var/run:/var/run:rw --volume=/sys:/sys:ro --volume=/var/lib/docker/:/var/lib/docker:ro --volume=/:/rootfs:ro --publish=8080:8080 --name=cadvisor google/cadvisor:latest
ExecStop=/usr/bin/docker rm -f cadvisor

[X-Fleet]
Global=true
```

**Step 2: Start InfluxDB**

You can use a fleet file like this named `heapster_influxdb.service`:

```ini
[Unit]
Description=influxdb

After=docker.service
Requires=docker_configs.mount

[Service]
TimeoutStartSec=0

# Change killmode from "control-group" to "none" to let Docker remove
# work correctly.
KillMode=none

ExecStartPre=-/usr/bin/docker kill influxdb
ExecStartPre=-/usr/bin/docker rm influxdb
ExecStart=/usr/bin/docker run -p 8083:8083 -p 8086:8086 -v /path/to/data:/data --hostname="influxdb" --name influxdb kubernetes/heapster_influxdb:v0.3

Restart=always
RestartSec=5

# Stop
ExecStop=/usr/bin/docker stop influxdb
```

Notice that this fleet file is using a volume for data. In order to retain your database, you will need to store it somewhere accessible by all machines in your cluster. Be sure to change /path/to/data to the actual path to your data.

**Step 3: Start heapster**

Since we are keeping heapster running on the same machine as influxdb, we can use a link here in our fleet file named heapster.service.

```ini
[Unit]
Description=heapster

After=docker.service
After=heapster_influxdb.service

Requires=docker.service
Requires=heapster_influxdb.service

[Service]
TimeoutStartSec=0

# Change killmode from "control-group" to "none" to let Docker remove
# work correctly.
KillMode=none

ExecStartPre=-/usr/bin/docker kill heapster
ExecStartPre=-/usr/bin/docker rm heapster
ExecStart=/bin/bash -c "HOST_IP=`getent hosts %H|/usr/bin/cut -d\" \" -f1`; /usr/bin/docker run --name heapster --link influxdb:influxdb kubernetes/heapster:v0.13.0 --source=\"cadvisor:coreos?fleetEndpoint=http://$HOST_IP:4001&cadvisorPort=8080\" --sink='influxdb:http://influxdb:8086'"

Restart=always
RestartSec=5

# Stop
ExecStop=/usr/bin/docker stop heapster

[X-Fleet]
X-ConditionMachineOf=heapster_influxdb.service
```

**Step 4: Start Grafana**

Grafana's fleet file is named `heapster_grafana.service` and we are also keeping it on the same system as influxdb for simplicity:

```ini
[Unit]
Description=grafana

After=docker.service
After=heapster_influxdb.service

Requires=docker.service
Requires=heapster_influxdb.service

[Service]
TimeoutStartSec=0

# Change killmode from "control-group" to "none" to let Docker remove
# work correctly.
KillMode=none

ExecStartPre=-/usr/bin/docker kill grafana
ExecStartPre=-/usr/bin/docker rm grafana
ExecStart=/usr/bin/docker run --name=grafana --link influxdb:influxdb -p 80:8080 -e INFLUXDB_HOST=influxdb kubernetes/heapster_grafana:v0.7

Restart=always
RestartSec=5

# Stop
ExecStop=/usr/bin/docker stop grafana

[X-Fleet]
X-ConditionMachineOf=heapster_influxdb.service
```

**Notes**

* We are using --net=host on the heapster container simply to avoid having to find the IP of a fleet node. If this information is available, we can remove this requirement.

* Grafana will be available on whatever machine fleet decides to run influxdb. This means that it will jump around your cluster. It is best to use some sort of proxy setup to get to your service. This can be handled with something like haproxy or nginx, but is something that is left to the reader to find the best way to handle it for their situation.
