# Add `kubectl` command to open UI

## Motivation

As it was mentioned in [#28007](https://github.com/kubernetes/kubernetes/issues/28007) `kubectl` should have an option to open the UI that is installed under `http://<master_ip>/ui`, for example `kubectl open-ui`. This would be great discoverability improvement for the UI since many people start with CLI, not even knowing that there is a UI.

## Proposed Solution

`kubectl open-ui` command could use different solutions. First one is to run local proxy (`kubectl proxy`) and then open a browser with UI's URL. Second one (not covering all use cases) is to look for UI's service, get it's external endpoint, if it exists, and open a browser with it.

A mix of these two solutions seems to be the best, so if there is an external endpoint of UI's service it can be opened without starting local proxy. Otherwise, proxy can be started to access UI anyways.

###  Command handler's pseudocode

``` go
func OpenUI(w io.Writer, kubeClient client.Interface) {
	// Check if UI service exists in the cluster
	_, err := kubeClient.Services(api.NamespaceSystem).Get(DashboardServiceName)
	if err != nil {
		fmt.Printf("Couldn't find UI service in system namespace: %v\n", err)
		os.Exit(1)
	}

	// Check if there are any UI service endpoints
	uiAddress, err := getUIEndpoints(kubeClient)
	if err != nil {
		// Run proxy in background
		...
	}

    // Run browser with UI's URL
    ...
}
```

### Getting external endpoint of UI's service

First thing, that needs to be verified is availability of UI's service in the cluster. It should be done by checking if there is a `kubernetes-dashboard` service in the `kube-system` namespace. If it cannot be found, then opening UI is not possible and user should be informed about it in returned error.

After UI's service is found it could be checked if it has any external `kubernetes-dashboard` endpoints in the `kube-system` namespace with valid port and address. In this case, assuming that everything is valid, in this case UI's URL is `http://<endpoint-adress>:<endpoint-port>/`

### Accessing UI through the proxy

There is a possibility, that there won't be any external endpoint of UI's service in the cluster. In that case, the only way to access it is to do it using proxy.

`kubectl open-ui` can reuse mechanism used in `kubectl proxy` to start it in the background and then access UI's URL, which in this case should be `<proxy-host>/api/v1/proxy/namespaces/kube-system/services/kubernetes-dashboard` or just `<proxy-host>/ui`.

### Command flags

There is only one flag to handle:

- `-p`, `--print-only` - gives possibility to display UI's URL on stdout without opening it in the browser.

### Possible improvements

Before running the proxy it could be checked if proxy isn't already running in the cluster and so it could be reused.

## Used technologies

To run default browser on any OS `kubectl open-ui` might use it's own algorithm or use existing libraries like:

- https://github.com/toqueteos/webbrowser
- https://github.com/pkg/browser
- https://github.com/skratchdot/open-golang
