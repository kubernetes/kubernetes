# Proxying to the services using `kubectl proxy`

As it was discussed in #28007 and #32073, service discoverability in Kubernetes clusters can be improved by adding possibility to proxy directly to specific services with `kubectl proxy` command.

It would allow project maintainers to guide users how to quick and easy connect with newly-deployed services using improved `kubectl proxy`.

## Current state

Assuming `<service>` and `<namespace>` are respective service and namespace of a service, current version of `kubectl proxy` allows to proxy to `<service>` only, but command is quite complex:

```
kubectl proxy --api-prefix=/api/v1/proxy/namespaces/<namespace>/services/<service>
```

And address where `<service>` is served is quite long:

```
http://localhost:8001/api/v1/proxy/namespaces/<namespace>/services/<service>
```

## Intended state

`kubectl proxy` should allow to proxy to `<service>` using short and easy to understand command:

```
kubectl proxy --service=<namespace>/<service>
```

Address where `<service>` is served should be kept short:

```
http://localhost:8001/
```

As you can see this state seems to be much more user-friendly. It's short and user has no longer to remember whole API path to a service.

Additionally, adding `--service` (`-s`) flag seems to be least disruptive.

## Proposed solution

To achieve intended state we need to start with adding new `--service` (`-s`) flag to the `kubectl proxy`. When it will be added to the command, we should follow these steps:

1. Validate if `--service` value is in valid form (`<namespace>/<service>`).
2. Check if `<service>` exists in `<namespace>`.
3. Run proxy to `<service>` in `<namespace>`, serve it on `http://localhost:8001/` by default (can be changed with `--address` flag).

If error occurs on any of these steps command should stop and proper error message should be displayed.

### Validation

TODO.

### Initial checks

TODO.

### Running proxy

TODO.

## Possible enhancements

- Introducing option to list services, that can be accessed through `kubectl proxy --service=<namespace>/<service>` command.
- Introducing `--open-browser` flag to `kubectl proxy` command, that opens default web browser with served address


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/service-proxy.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
