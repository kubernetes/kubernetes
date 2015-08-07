###Background
When deploying Kubernetes using something like [Fleet](https://github.com/coreos/fleet), the API Server (and other services) may not stay on the same host (depending on your setup)

In these cases it's ideal to have a dynamic load balancer ([Hipache](https://github.com/hipache/hipache)) that can receive updates from your services.

###Setup
Our example is based on Kelsey Hightower's "[Kubernetes Fleet Tutorial](https://github.com/kelseyhightower/kubernetes-fleet-tutorial)" (The bash variable ${DEFAULT_IPV4} is set in Kelsey's /etc/network-environment file)

For this write-up we are going to assume you have a dedicated [etcd](https://github.com/coreos/etcd) endpoint (10.1.10.10 Private IPV4) and are running kubernetes on systems managed by systemd / fleet.

The Hipache instance is going to run on 172.20.1.20 (Public IPV4) but will have a Private IPV4 address as well (10.1.10.11)


First, create your kube-apiserver.service file (change necessary variables)
`~/hipache/kube-apiserver.service`
```
    [Unit]
    Description=Kubernetes API Server
    Documentation=https://github.com/GoogleCloudPlatform/kubernetes

    [Service]
    EnvironmentFile=/etc/network-environment
    ExecStartPre=/usr/bin/rm /opt/bin/kube-apiserver
    ExecStartPre=/usr/bin/wget -P /opt/bin https://path/to/kube-apiserver/binary
    ExecStartPre=/usr/bin/chmod +x /opt/bin/kube-apiserver
    ExecStart=/opt/bin/kube-apiserver \
    --address=0.0.0.0 \
    --port=8080 \
    --etcd-servers=http://10.1.10.10:4001
    ExecStartPost=/usr/bin/etcdctl -C 10.1.10.10:4001 set /frontend:172.20.1.20 '[ "kubernetes", "http://${DEFAULT_IPV4}:8080" ]'
    Restart=always
    RestartSec=10

    [X-Fleet]
    MachineMetadata=role=kubernetes
```

Next we need a Hipache instance and a config file. In our case, we just rolled our own docker container for it.

`~/workspace/hipache/Dockerfile`
```
    FROM ubuntu:14.04

    RUN apt-get update && \
            apt-get -y install nodejs npm && \
            npm install node-etcd hipache -g
    RUN mkdir /hipache
    ADD . /hipache
    RUN cd /hipache
    ENV NODE_ENV production
    EXPOSE 80


    CMD hipache -c /hipache/config.json
```
`~/workspace/hipache/config.json`
```
    {
        "server": {
            "accessLog": "/tmp/access.log",
            "port": 80,
            "workers": 10,
            "maxSockets": 100,
            "deadBackendTTL": 30,
            "tcpTimeout": 30,
            "retryOnError": 3,
            "deadBackendOn500": true,
            "httpKeepAlive": false
        },
        "driver": ["etcd://10.1.10.10:4001"]
    }

```

We need to build the docker container and set up the systemd service for our Hipache container.
`docker build -t kube-hipache .`

`/etc/systemd/system/kube-hipache.service`
```
    [Unit]
    Description=Hipache Router
    After=docker.service
    Requires=docker.service

    [Service]
    TimeoutStartSec=0
    ExecStartPre=-/usr/bin/docker kill hipache
    ExecStartPre=-/usr/bin/docker rm hipache
    ExecStart=/usr/bin/docker run -d -p 80:80 --name hipache hipache

    [Install]
    WantedBy=multi-user.target
```
Let's put some pieces together! Run the following commands:
- `systemctl enable /etc/systemd/system/kube-hipache.service `
- `systemctl start kube-hipache.service`
- `journalctl -b -u kube-hipache.service` (Make sure it's running)
- `fleetctl start ~/hipache/kube-apiserver.service`

That's it! Fleet will schedule the apiserver on one of your minions and once it's started it will register itself in etcd. Hipache will auto-update once this happens and you should never have to worry which node the apiserver is sitting on.


###Questions
twitter @jeefy

irc.freenode.net #kubernetes jeefy


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/recipes/docs/apiserver_hipache_registration.md?pixel)]()
