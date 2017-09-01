## New Relic Infrastructure Server Monitoring Agent Example

This example shows how to run a New Relic Infrastructure server monitoring agent as a pod in a DaemonSet on an existing Kubernetes cluster.

This example will create a DaemonSet which places the New Relic Infrastructure monitoring agent on every node in the cluster. It's also fairly trivial to exclude specific Kubernetes nodes from the DaemonSet to just monitor specific servers.  (The prior nrsysmond has been deprecated.)

### Step 0: Prerequisites

This process will create privileged containers which have full access to the host system for logging. Beware of the security implications of this.

DaemonSets must be enabled on your cluster. Instructions for enabling DaemonSet can be found [here](../../docs/api.md#enabling-the-extensions-group).

### Step 1: Configure New Relic Infrastructure Agent

The New Relic Infrastructure agent is configured via environment variables. We will configure these environment variables in a sourced bash script, encode the environment file data, and store it in a secret which will be loaded at container runtime. (Reread this sentence a few times, it's *HOW* the entire container works.)

The [New Relic Linux Infrastructure Server configuration page](https://docs.newrelic.com/docs/servers/new-relic-servers-linux/installation-configuration/configuring-servers-linux) lists all the other settings for the Infrastructure process.

To create an environment variable for a setting, prepend NRIA_ to its name and capitalize all of the env variable.  For example,

```console
log_file=/var/log/nr-infra.log
```

translates to

```console
NRIA_LOG_FILE=/var/log/nr-infra.log
```

Edit examples/newrelic-infrastructure/nrconfig.env and configure relevant environment variables for your NewRelic Infrastructure agent.  There are a few defauts defined, but the only required variable is the New Relic license key.

Now, let's vendor the config into a secret.

```console
$ cd examples/newrelic-infrastructure/
$ ./config-to-secret.sh
```

<!-- BEGIN MUNGE: EXAMPLE newrelic-config-template.yaml -->

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: newrelic-config
type: Opaque
data:
  config: {{config_data}}
```

[Download example](newrelic-config-template.yaml?raw=true)
<!-- END MUNGE: EXAMPLE newrelic-config-template.yaml -->

The script will encode the config file and write it to `newrelic-config.yaml`.

Finally, submit the config to the cluster:

```console
$ kubectl create -f examples/newrelic-infrastructure/newrelic-config.yaml
```

### Step 2: Create the DaemonSet definition.

The DaemonSet definition instructs Kubernetes to place a newrelic Infrastructure agent on each Kubernetes node.

<!-- BEGIN MUNGE: EXAMPLE newrelic-infra-daemonset.yaml -->

```yaml
apiVersion: extensions/v1beta1
kind: DaemonSet
metadata:
  name: newrelic-infra-agent
  labels:
    tier: monitoring
    app: newrelic-infra-agent
    version: v1
spec:
  template:
    metadata:
      labels:
        name: newrelic
    spec:
      # Filter to specific nodes:
      # nodeSelector:
      #  app: newrelic
      hostPID: true
      hostIPC: true
      hostNetwork: true
      containers:
        - resources:
            requests:
              cpu: 0.15
          securityContext:
            privileged: true
          image: newrelic/infrastructure
          name: newrelic
          command: [ "bash", "-c", "source /etc/kube-nr-infra/config && /usr/bin/newrelic-infra" ]
          volumeMounts:
            - name: newrelic-config
              mountPath: /etc/kube-nr-infra
              readOnly: true
            - name: dev
              mountPath: /dev
            - name: run
              mountPath: /var/run/docker.sock
            - name: log
              mountPath: /var/log
            - name: host-root
              mountPath: /host
              readOnly: true
      volumes:
        - name: newrelic-config
          secret:
            secretName: newrelic-config
        - name: dev
          hostPath:
              path: /dev
        - name: run
          hostPath:
              path: /var/run/docker.sock
        - name: log
          hostPath:
              path: /var/log
        - name: host-root
          hostPath:
              path: /
```

[Download example](newrelic-infra-daemonset.yaml?raw=true)
<!-- END MUNGE: EXAMPLE newrelic-infra-daemonset.yaml -->

The daemonset instructs Kubernetes to spawn pods on each node, mapping /dev/, /run/, and /var/log to the container.  It also maps the entire kube node / to /host/ in the container with a read-only mount.  It also maps the secrets we set up earlier to /etc/kube-newrelic/config, and sources them in the startup script, configuring the agent properly.

#### DaemonSet customization

- There are more environment variables for fine tuning the infrastructure agent's operation (or a yaml file that you'd have to construct).  See [Infrastructure Agent Environment Variables][(https://docs.newrelic.com/docs/infrastructure/new-relic-infrastructure/configuration/configure-infrastructure-agent) for the full list.


### Known issues

It's a bit cludgy to define the environment variables like we do here in these config files. There is [another issue](https://github.com/kubernetes/kubernetes/issues/4710) to discuss adding mapping secrets to environment variables in Kubernetes.  (Personally I don't like that method and prefer to use the config secrets.)

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/newrelic/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
