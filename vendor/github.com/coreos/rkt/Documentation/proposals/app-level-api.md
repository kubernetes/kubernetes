# Imperative app-level API for pod manipulation

To provide an API for imperative application operations (e.g. start|stop|add)
inside a pod for finer-grained control over rkt containerization concepts and
debugging needs, this proposal introduces new stage1 entrypoints and a
subcommand CLI API that will be used for manipulating applications inside pods.

The primary motivation behind this change is to facilitate the new direction
orchestration systems are taking in how they integrate with container runtimes.
For more details, see
[kubernetes#25899][k8s-25899].

# API

The envisioned workflow for the app-level API is that after a pod has been
started, users will invoke the rkt CLI to manipulate the pod. The
implementation behaviour consists of application logic on top of the
aforementioned stage1 entrypoints.

The proposed app-level commands are described below.

## `rkt app sandbox`
Initializes an empty pod having no applications. This returns a single line
containing the `pod-uuid` which can be used to perform application
operations specified below. This also implies the started pod will be injectable.

```bash
rkt app sandbox
```

## `rkt app add`
Injects an application image into a running pod. After this has been called,
the app is prepared and ready to be run via `rkt app start`.

It first prepares an application rootfs for the application image, creates a
runtime manifest for the app, and then injects the prepared app via an
entrypoint.

```bash
rkt app add <pod-uuid> --app=<app-name> <image-name/hash/address/registry-URL> <arguments>
```

**Note:** Not every pod will be injectable; it will be configured through an
option when the pod is created..

## `rkt app start`
Starts an application that was previously added (injected) to a pod. This
operation is idempotent; if the specified application is already started, it
will have no effect.

```bash
rkt app start <pod-uuid> --app=<app-name> <arguments>
```

## `rkt app stop`
Stops a running application gracefully. Grace is defined in the `app/stop`
entrypoint section. This does not remove associated resources (see `app rm`).

```bash
rkt app stop <pod-uuid> --app=<app-name>
```

## `rkt app rm`
Removes a stopped application from a running pod, including all associated
resources.

```bash
rkt app rm <pod-uuid> --app=<app-name> <arguments>
```

**Note:** currently, when a pod becomes empty (no apps are running), it will
terminate. This proposal will introduce a `--mutable` or `--allow-empty` or
`--dumb` flag to be used when starting pods, so that the lifecycle management
of the pod is configurable by the user (i.e. it will be possible to create a
pod that won't be terminated when it is empty).

### Resources left over by a stopped application (in default stage1 flavor)
- Rootfs (e.g. `/opt/stage2/<app-name>`)
- Mounts from volumes (e.g. `/opt/stage2/<app-name>/<volume-name>`)
- Mounts related to rkt operations (e.g. `/opt/stage2/<app-name>/dev/null`)
- systemd service files (e.g. `<app-name>.service` and `reaper-<app-name>.service`)
- Miscellaneous files (e.g. `/rkt/<app-name>.env`, `/rkt/status...`)

## `rkt app list`
Lists the applications that are inside a pod, running or stopped.

```bash
rkt app list <pod-uuid> <arguments>
```

**Note:** The information returned by list should consist of an app specifier
and status at the very least, the rest is up for discussion.

## `rkt app status`
Returns the execution status of an application inside a pod.

```bash
rkt app status <pod-uuid> --app=<app-name> <arguments>
```

The returned status information for an application would contain the following
details (output format is up for discussion):

```go
type AppStatus struct {
	Name       string
	State      AppState
	CreatedAt  time.Time
	StartedAt  time.Time
	FinishedAt time.Time
	ExitCode   int64 
}
```

**Note:** status will be obtained from an annotated JSON file residing in stage1
that contains the required information.
_**OPEN QUESTION**: what is responsible for updating this file? How is concurrent access handled?_

## `rkt app exec`
Executes a command inside an application.

```bash
rkt app exec <pod-uuid> --app=<app-name> <arguments> -- <command> <command-arguments>
```

# Entrypoints

In order to facilitate the app-level operations API, four new stage1 entrypoints are introduced.
Entrypoints are resolved via annotations found within a pod's stage1 manifest
(e.g. `/var/lib/rkt/pods/run/$uuid/stage1/manifest`).

## `coreos.com/rkt/stage1/app/add`

The responsibility of this entrypoint is to receive a prepared app and inject it
into the pod, where it will be started using the `app/start` entrypoint.

The entrypoint should receive a reference to a runtime manifest of the prepared
app, and perform any necessary setup based on that runtime manifest.

## `coreos.com/rkt/stage1/app/rm`

The responsibility of this entrypoint is to remove an app from a pod. After
`rm`, starting the application again is not possible - the app must be
re-injected to be re-used.

1. receive a reference to an application that resides inside the pod (running or stopped)
2. stop the application if its running.
3. remove the contents of the application (rootfs) from the pod (keep the logs?) and delete references to it (e.g. service files).

## `coreos.com/rkt/stage1/app/start`

The responsibility of this entrypoint is to start an application that is in the
`Prepared` state, which is an app that was recently injected.

## `coreos.com/rkt/stage1/app/stop`

The responsibility of this entrypoint is to stop an application that is
in the `Running` state, by instructing the stage1.

rkt will attempt a _graceful shutdown_: sending a termination signal
(i.e. `SIGTERM`) to application and waiting for a grace period for the
application to exit. If the application does not terminate by the end of the
grace period, rkt will forcefully shut it down (i.e. `SIGKILL`).

# App States

Expected set of app states are listed below:

```go
type AppState string

const (
	UnknownAppState AppState = "unknown"

	PreparingAppState AppState = "preparing"

	// Apps that are ready to be used by `app start`.
	PreparedAppState AppState = "prepared"

	RunningAppState AppState = "running"

	// Apps stopped by `app stop`.
	StoppingAppState AppState = "stopping"

	// Apps that finish their execution naturally.
	ExitedAppState AppState = "exited"

	// Once an app is marked for removal, while the removal is being
	// performed, no further operations can be done on that app.
	DeletingAppState AppState = "deleting"
)
```

**Note:** State transitions are linear; an app that is in state `Exited` cannot
transition into `Running` state.
_**OPEN QUESTION** can a stopped app not be restarted?_

# Use Cases

## Low-level Pod Control

Grant granular access to pods for orchestration systems and allow orchestration
systems to develop their own pod concept on top of the exposed app-level
operations.

### Example Workflow

1. Create an empty pod.
2. Inject applications into the pod.
3. Orchestrate the workflow of applications (e.g. app1 has to terminate successfully before app2).

## Updates

Enable in-place updates of a pod without disrupting the operations of the pod.

### Example Workflow

1. Remove old applications without disturbing/restarting the whole pod.
2. Inject updated applications.
3. Start the updated applications.

## Debugging Pods

Allow users to inject debug applications into a pod in production.

### Example Workflow

1. Deploy an application containing only a Go web service binary.
2. Encounter an error not decipherable via the available information (e.g. status info, logs, etc.).
3. Add a debug app image containing binaries (e.g. `lsof`) for debugging the service.
4. Enter the pod namespace and use the debug binaries.


[k8s-25899]: https://github.com/yujuhong/kubernetes/blob/08dc66113399c89e31f6872f3c638695a6ec6a8d/docs/proposals/container-runtime-interface-v1.md
