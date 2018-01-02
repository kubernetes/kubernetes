# Configuration Flags

etcd is configurable through command-line flags and environment variables. Options set on the command line take precedence over those from the environment.

The format of environment variable for flag `--my-flag` is `ETCD_MY_FLAG`. It applies to all  flags.

The [official etcd ports][iana-ports] are 2379 for client requests, and 2380 for peer communication. Some legacy code and documentation still references ports 4001 and 7001, but all new etcd use and discussion should adopt the assigned ports.

To start etcd automatically using custom settings at startup in Linux, using a [systemd][systemd-intro] unit is highly recommended.

[systemd-intro]: http://freedesktop.org/wiki/Software/systemd/

## Member Flags

### --name
+ Human-readable name for this member.
+ default: "default"
+ env variable: ETCD_NAME
+ This value is referenced as this node's own entries listed in the `--initial-cluster` flag (Ex: `default=http://localhost:2380` or `default=http://localhost:2380,default=http://localhost:7001`). This needs to match the key used in the flag if you're using [static bootstrapping][build-cluster]. When using discovery, each member must have a unique name. `Hostname` or `machine-id` can be a good choice.

### --data-dir
+ Path to the data directory.
+ default: "${name}.etcd"
+ env variable: ETCD_DATA_DIR

### --wal-dir
+ Path to the dedicated wal directory. If this flag is set, etcd will write the WAL files to the walDir rather than the dataDir. This allows a dedicated disk to be used, and helps avoid io competition between logging and other IO operations.
+ default: ""
+ env variable: ETCD_WAL_DIR

### --snapshot-count
+ Number of committed transactions to trigger a snapshot to disk.
+ default: "10000"
+ env variable: ETCD_SNAPSHOT_COUNT

### --heartbeat-interval
+ Time (in milliseconds) of a heartbeat interval.
+ default: "100"
+ env variable: ETCD_HEARTBEAT_INTERVAL

### --election-timeout
+ Time (in milliseconds) for an election to timeout. See [tuning.md](tuning.md#time-parameters) for details.
+ default: "1000"
+ env variable: ETCD_ELECTION_TIMEOUT

### --listen-peer-urls
+ List of URLs to listen on for peer traffic. This flag tells the etcd to accept incoming requests from its peers on the specified scheme://IP:port combinations. Scheme can be either http or https.If 0.0.0.0 is specified as the IP, etcd listens to the given port on all interfaces. If an IP address is given as well as a port, etcd will listen on the given port and interface. Multiple URLs may be used to specify a number of addresses and ports to listen on. The etcd will respond to requests from any of the listed addresses and ports.
+ default: "http://localhost:2380,http://localhost:7001"
+ env variable: ETCD_LISTEN_PEER_URLS
+ example: "http://10.0.0.1:2380"
+ invalid example: "http://example.com:2380" (domain name is invalid for binding)

### --listen-client-urls
+ List of URLs to listen on for client traffic. This flag tells the etcd to accept incoming requests from the clients on the specified scheme://IP:port combinations. Scheme can be either http or https. If 0.0.0.0 is specified as the IP, etcd listens to the given port on all interfaces. If an IP address is given as well as a port, etcd will listen on the given port and interface. Multiple URLs may be used to specify a number of addresses and ports to listen on. The etcd will respond to requests from any of the listed addresses and ports.
+ default: "http://localhost:2379,http://localhost:4001"
+ env variable: ETCD_LISTEN_CLIENT_URLS
+ example: "http://10.0.0.1:2379"
+ invalid example: "http://example.com:2379" (domain name is invalid for binding)

### --max-snapshots
+ Maximum number of snapshot files to retain (0 is unlimited)
+ default: 5
+ env variable: ETCD_MAX_SNAPSHOTS
+ The default for users on Windows is unlimited, and manual purging down to 5 (or your preference for safety) is recommended.

### --max-wals
+ Maximum number of wal files to retain (0 is unlimited)
+ default: 5
+ env variable: ETCD_MAX_WALS
+ The default for users on Windows is unlimited, and manual purging down to 5 (or your preference for safety) is recommended.

### --cors
+ Comma-separated white list of origins for CORS (cross-origin resource sharing).
+ default: none
+ env variable: ETCD_CORS

## Clustering Flags

`--initial` prefix flags are used in bootstrapping ([static bootstrap][build-cluster], [discovery-service bootstrap][discovery] or [runtime reconfiguration][reconfig]) a new member, and ignored when restarting an existing member.

`--discovery` prefix flags need to be set when using [discovery service][discovery].

### --initial-advertise-peer-urls

+ List of this member's peer URLs to advertise to the rest of the cluster. These addresses are used for communicating etcd data around the cluster. At least one must be routable to all cluster members. These URLs can contain domain names.
+ default: "http://localhost:2380,http://localhost:7001"
+ env variable: ETCD_INITIAL_ADVERTISE_PEER_URLS
+ example: "http://example.com:2380, http://10.0.0.1:2380"

### --initial-cluster
+ Initial cluster configuration for bootstrapping.
+ default: "default=http://localhost:2380,default=http://localhost:7001"
+ env variable: ETCD_INITIAL_CLUSTER
+ The key is the value of the `--name` flag for each node provided. The default uses `default` for the key because this is the default for the `--name` flag.

### --initial-cluster-state
+ Initial cluster state ("new" or "existing"). Set to `new` for all members present during initial static or DNS bootstrapping. If this option is set to `existing`, etcd will attempt to join the existing cluster. If the wrong value is set, etcd will attempt to start but fail safely.
+ default: "new"
+ env variable: ETCD_INITIAL_CLUSTER_STATE

[static bootstrap]: clustering.md#static

### --initial-cluster-token
+ Initial cluster token for the etcd cluster during bootstrap.
+ default: "etcd-cluster"
+ env variable: ETCD_INITIAL_CLUSTER_TOKEN

### --advertise-client-urls
+ List of this member's client URLs to advertise to the rest of the cluster. These URLs can contain domain names.
+ default: "http://localhost:2379,http://localhost:4001"
+ env variable: ETCD_ADVERTISE_CLIENT_URLS
+ example: "http://example.com:2379, http://10.0.0.1:2379"
+ Be careful if you are advertising URLs such as http://localhost:2379 from a cluster member and are using the proxy feature of etcd. This will cause loops, because the proxy will be forwarding requests to itself until its resources (memory, file descriptors) are eventually depleted.

### --discovery
+ Discovery URL used to bootstrap the cluster.
+ default: none
+ env variable: ETCD_DISCOVERY

### --discovery-srv
+ DNS srv domain used to bootstrap the cluster.
+ default: none
+ env variable: ETCD_DISCOVERY_SRV

### --discovery-fallback
+ Expected behavior ("exit" or "proxy") when discovery services fails.
+ default: "proxy"
+ env variable: ETCD_DISCOVERY_FALLBACK

### --discovery-proxy
+ HTTP proxy to use for traffic to discovery service.
+ default: none
+ env variable: ETCD_DISCOVERY_PROXY

### --strict-reconfig-check
+ Reject reconfiguration requests that would cause quorum loss.
+ default: false
+ env variable: ETCD_STRICT_RECONFIG_CHECK

## Proxy Flags

`--proxy` prefix flags configures etcd to run in [proxy mode][proxy].

### --proxy
+ Proxy mode setting ("off", "readonly" or "on").
+ default: "off"
+ env variable: ETCD_PROXY

### --proxy-failure-wait
+ Time (in milliseconds) an endpoint will be held in a failed state before being reconsidered for proxied requests.
+ default: 5000
+ env variable: ETCD_PROXY_FAILURE_WAIT

### --proxy-refresh-interval
+ Time (in milliseconds) of the endpoints refresh interval.
+ default: 30000
+ env variable: ETCD_PROXY_REFRESH_INTERVAL

### --proxy-dial-timeout
+ Time (in milliseconds) for a dial to timeout or 0 to disable the timeout
+ default: 1000
+ env variable: ETCD_PROXY_DIAL_TIMEOUT

### --proxy-write-timeout
+ Time (in milliseconds) for a write to timeout or 0 to disable the timeout.
+ default: 5000
+ env variable: ETCD_PROXY_WRITE_TIMEOUT

### --proxy-read-timeout
+ Time (in milliseconds) for a read to timeout or 0 to disable the timeout.
+ Don't change this value if you use watches because they are using long polling requests.
+ default: 0
+ env variable: ETCD_PROXY_READ_TIMEOUT

## Security Flags

The security flags help to [build a secure etcd cluster][security].

### --ca-file [DEPRECATED]
+ Path to the client server TLS CA file. `--ca-file ca.crt` could be replaced by `--trusted-ca-file ca.crt --client-cert-auth` and etcd will perform the same.
+ default: none
+ env variable: ETCD_CA_FILE

### --cert-file
+ Path to the client server TLS cert file.
+ default: none
+ env variable: ETCD_CERT_FILE

### --key-file
+ Path to the client server TLS key file.
+ default: none
+ env variable: ETCD_KEY_FILE

### --client-cert-auth
+ Enable client cert authentication.
+ default: false
+ env variable: ETCD_CLIENT_CERT_AUTH

### --trusted-ca-file
+ Path to the client server TLS trusted CA key file.
+ default: none
+ env variable: ETCD_TRUSTED_CA_FILE

### --peer-ca-file [DEPRECATED]
+ Path to the peer server TLS CA file. `--peer-ca-file ca.crt` could be replaced by `--peer-trusted-ca-file ca.crt --peer-client-cert-auth` and etcd will perform the same.
+ default: none
+ env variable: ETCD_PEER_CA_FILE

### --peer-cert-file
+ Path to the peer server TLS cert file.
+ default: none
+ env variable: ETCD_PEER_CERT_FILE

### --peer-key-file
+ Path to the peer server TLS key file.
+ default: none
+ env variable: ETCD_PEER_KEY_FILE

### --peer-client-cert-auth
+ Enable peer client cert authentication.
+ default: false
+ env variable: ETCD_PEER_CLIENT_CERT_AUTH

### --peer-trusted-ca-file
+ Path to the peer server TLS trusted CA file.
+ default: none
+ env variable: ETCD_PEER_TRUSTED_CA_FILE

## Logging Flags

### --debug
+ Drop the default log level to DEBUG for all subpackages.
+ default: false (INFO for all packages)
+ env variable: ETCD_DEBUG

### --log-package-levels
+ Set individual etcd subpackages to specific log levels. An example being `etcdserver=WARNING,security=DEBUG`
+ default: none (INFO for all packages)
+ env variable: ETCD_LOG_PACKAGE_LEVELS


## Unsafe Flags

Please be CAUTIOUS when using unsafe flags because it will break the guarantees given by the consensus protocol.
For example, it may panic if other members in the cluster are still alive.
Follow the instructions when using these flags.

### --force-new-cluster
+ Force to create a new one-member cluster. It commits configuration changes forcing to remove all existing members in the cluster and add itself. It needs to be set to [restore a backup][restore].
+ default: false
+ env variable: ETCD_FORCE_NEW_CLUSTER

## Experimental Flags

### --experimental-v3demo
+ Enable experimental [v3 demo API][rfc-v3].
+ default: false
+ env variable: ETCD_EXPERIMENTAL_V3DEMO

## Miscellaneous Flags

### --version
+ Print the version and exit.
+ default: false

## Profiling flags

### --enable-pprof
+ Enable runtime profiling data via HTTP server. Address is at client URL + "/debug/pprof/"
+ default: false

[build-cluster]: clustering.md#static
[reconfig]: runtime-configuration.md
[discovery]: clustering.md#discovery
[iana-ports]: http://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.txt
[proxy]: proxy.md
[reconfig]: runtime-configuration.md
[restore]: admin_guide.md#restoring-a-backup
[rfc-v3]: rfc/v3api.md
[security]: security.md
[systemd-intro]: http://freedesktop.org/wiki/Software/systemd/
[tuning]: tuning.md#time-parameters
