# govc.el

Interface to govc for managing VMware ESXi and vCenter.


The goal of this package is to provide a simple interface for commonly used
govc commands within Emacs.  This includes table based inventory/state modes
for vms, hosts, datastores and pools.  The keymap for each mode provides
shortcuts for easily feeding the data in view to other govc commands.

Within the various govc modes, press `?` to see a popup menu of options.
A menu bar is enabled for certain modes, such as `govc-vm-mode` and `govc-host-mode`.
There is also a `govc` menu at all times under the `Tools` menu.

The recommended way to install govc.el is via MELPA (http://melpa.org/).

## govc-mode

Running `govc-global-mode` creates key bindings to the various govc modes.
The default prefix is `C-c ;` and can be changed by setting `govc-keymap-prefix`.

### govc-command-map

Keybinding     | Description
---------------|------------------------------------------------------------
<kbd>h</kbd>   | Host info via govc
<kbd>p</kbd>   | Pool info via govc
<kbd>v</kbd>   | VM info via govc
<kbd>s</kbd>   | Datastore info via govc

### govc-urls

List of URLs for use with `govc-session`.
The `govc-session-name` displayed by `govc-mode-line` uses `url-target` (anchor)
if set, otherwise `url-host` is used.

Example:
```
  (setq govc-urls `("root:vagrant@localhost:18443#Vagrant-ESXi"
                    "root:password@192.168.1.192#Intel-NUC"
                    "Administrator@vsphere.local:password!@vcva-clovervm"))
```
To enter a URL that is not in the list, prefix `universal-argument`, for example:

  `C-u M-x govc-vm`

To avoid putting your credentials in a variable, you can use the
auth-source search integration.

```
  (setq govc-urls `("myserver-vmware-2"))
```

And then put this line in your `auth-sources` (e.g. `~/.authinfo.gpg`):
```
    machine myserver-vmware-2 login tzz password mypass url "myserver-vmware-2.some.domain.here:443?insecure=true"
```

Which will result in the URL "tzz:mypass@myserver-vmware-2.some.domain.here:443?insecure=true".
For more details on `auth-sources`, see Info node `(auth) Help for users`.

When in `govc-vm` or `govc-host` mode, a default URL is composed with the
current session credentials and the IP address of the current vm/host and
the vm/host name as the session name.  This makes it easier to connect to
nested ESX/vCenter VMs or directly to an ESX host.

### govc-session-url

ESX or vCenter URL set by `govc-session` via `govc-urls` selection.

### govc-session-insecure

Skip verification of server certificate when true.
This variable is set to the value of the `GOVC_INSECURE` env var by default.
It can also be set per-url via the query string (insecure=true).  For example:
```
  (setq govc-urls `("root:password@hostname?insecure=true"))
```

### govc-session-datacenter

Datacenter to use for the current `govc-session`.
If the endpoint has a single Datacenter it will be used by default, otherwise
`govc-session` will prompt for selection.  It can also be set per-url via the
query string.  For example:
```
  (setq govc-urls `("root:password@hostname?datacenter=dc1"))
```

### govc-session-datastore

Datastore to use for the current `govc-session`.
If the endpoint has a single Datastore it will be used by default, otherwise
`govc-session` will prompt for selection.  It can also be set per-url via the
query string.  For example:
```
  (setq govc-urls `("root:password@hostname?datastore=vsanDatastore"))
```

### govc-session-network

Network to use for the current `govc-session`.

## govc-tabulated-list-mode

Generic table bindings to mark/unmark rows.

In addition to any hooks its parent mode `tabulated-list-mode` might have run,
this mode runs the hook `govc-tabulated-list-mode-hook`, as the final step
during initialization.

### govc-tabulated-list-mode-map

Keybinding     | Description
---------------|------------------------------------------------------------
<kbd>m</kbd>   | Mark and move to the next line
<kbd>u</kbd>   | Unmark and move to the next line
<kbd>t</kbd>   | Toggle mark
<kbd>U</kbd>   | Unmark all
<kbd>M-&</kbd> | Shell CMD in BUFFER with current `govc-session` exported as GOVC_ env vars
<kbd>M-w</kbd> | Copy current selection or region to the kill ring
<kbd>M-E</kbd> | Export session to `process-environment` and `kill-ring`

## govc-host-mode

Major mode for handling a list of govc hosts.

In addition to any hooks its parent mode `govc-tabulated-list-mode` might have run,
this mode runs the hook `govc-host-mode-hook`, as the final step
during initialization.

### govc-host-mode-map

Keybinding     | Description
---------------|------------------------------------------------------------
<kbd>E</kbd>   | Events via govc events -n `govc-max-events`
<kbd>L</kbd>   | Logs via govc logs -n `govc-max-events`
<kbd>J</kbd>   | JSON via govc host
<kbd>M</kbd>   | Metrics info
<kbd>N</kbd>   | Netstat via `govc-esxcli-netstat-info` with current host id
<kbd>O</kbd>   | Object browser via govc object
<kbd>T</kbd>   | Tasks via govc tasks
<kbd>c</kbd>   | Connect new session for the current govc mode
<kbd>p</kbd>   | Pool-mode with current session
<kbd>s</kbd>   | Datastore-mode with current session
<kbd>v</kbd>   | VM-mode with current session

## govc-pool-mode

Major mode for handling a list of govc pools.

In addition to any hooks its parent mode `govc-tabulated-list-mode` might have run,
this mode runs the hook `govc-pool-mode-hook`, as the final step
during initialization.

### govc-pool-mode-map

Keybinding     | Description
---------------|------------------------------------------------------------
<kbd>D</kbd>   | Destroy via `govc-pool-destroy` on the pool selection
<kbd>E</kbd>   | Events via govc events -n `govc-max-events`
<kbd>J</kbd>   | JSON via govc pool
<kbd>M</kbd>   | Metrics info
<kbd>O</kbd>   | Object browser via govc object
<kbd>T</kbd>   | Tasks via govc tasks
<kbd>c</kbd>   | Connect new session for the current govc mode
<kbd>h</kbd>   | Host-mode with current session
<kbd>s</kbd>   | Datastore-mode with current session
<kbd>v</kbd>   | VM-mode with current session

## govc-datastore-mode

Major mode for govc datastore.info.

In addition to any hooks its parent mode `tabulated-list-mode` might have run,
this mode runs the hook `govc-datastore-mode-hook`, as the final step
during initialization.

### govc-datastore-mode-map

Keybinding     | Description
---------------|------------------------------------------------------------
<kbd>J</kbd>   | JSON via govc datastore
<kbd>M</kbd>   | Metrics info
<kbd>O</kbd>   | Object browser via govc object
<kbd>RET</kbd> | Browse datastore
<kbd>c</kbd>   | Connect new session for the current govc mode
<kbd>h</kbd>   | Host-mode with current session
<kbd>p</kbd>   | Pool-mode with current session
<kbd>v</kbd>   | VM-mode with current session

## govc-datastore-ls-mode

Major mode govc datastore.ls.

In addition to any hooks its parent mode `govc-tabulated-list-mode` might have run,
this mode runs the hook `govc-datastore-ls-mode-hook`, as the final step
during initialization.

### govc-datastore-ls-mode-map

Keybinding     | Description
---------------|------------------------------------------------------------
<kbd>I</kbd>   | Info datastore disk
<kbd>J</kbd>   | JSON via govc datastore
<kbd>S</kbd>   | Search via govc datastore
<kbd>D</kbd>   | Delete selected datastore paths
<kbd>T</kbd>   | Tail datastore FILE
<kbd>+</kbd>   | Mkdir via govc datastore
<kbd>DEL</kbd> | Up to parent folder
<kbd>RET</kbd> | Open datastore folder or file

## govc-vm-mode

Major mode for handling a list of govc vms.

In addition to any hooks its parent mode `govc-tabulated-list-mode` might have run,
this mode runs the hook `govc-vm-mode-hook`, as the final step
during initialization.

### govc-vm-mode-map

Keybinding     | Description
---------------|------------------------------------------------------------
<kbd>E</kbd>   | Events via govc events -n `govc-max-events`
<kbd>L</kbd>   | Logs via `govc-datastore-tail` with logDirectory of current selection
<kbd>J</kbd>   | JSON via govc vm
<kbd>O</kbd>   | Object browser via govc object
<kbd>T</kbd>   | Tasks via govc tasks
<kbd>X</kbd>   | ExtraConfig via `govc-vm-extra-config` on the current selection
<kbd>RET</kbd> | Devices via `govc-device` on the current selection
<kbd>C</kbd>   | Console via `govc-vm-console` on the current selection
<kbd>V</kbd>   | VNC via `govc-vm-vnc` on the current selection
<kbd>D</kbd>   | Destroy via `govc-vm-destroy` on the current selection
<kbd>^</kbd>   | Start via `govc-vm-start` on the current selection
<kbd>!</kbd>   | Shutdown via `govc-vm-shutdown` on the current selection
<kbd>@</kbd>   | Reboot via `govc-vm-reboot` on the current selection
<kbd>&</kbd>   | Suspend via `govc-vm-suspend` on the current selection
<kbd>H</kbd>   | Host info via `govc-host` with host(s) of current selection
<kbd>M</kbd>   | Metrics info
<kbd>P</kbd>   | Ping VM
<kbd>S</kbd>   | Datastore via `govc-datastore-ls` with datastore of current selection
<kbd>c</kbd>   | Connect new session for the current govc mode
<kbd>h</kbd>   | Host-mode with current session
<kbd>p</kbd>   | Pool-mode with current session
<kbd>s</kbd>   | Datastore-mode with current session

## govc-device-mode

Major mode for handling a govc device.

In addition to any hooks its parent mode `govc-tabulated-list-mode` might have run,
this mode runs the hook `govc-device-mode-hook`, as the final step
during initialization.

### govc-device-mode-map

Keybinding     | Description
---------------|------------------------------------------------------------
<kbd>J</kbd>   | JSON via govc device
<kbd>RET</kbd> | Tabulated govc device

## govc-object-mode

Major mode for handling a govc object.

In addition to any hooks its parent mode `govc-tabulated-list-mode` might have run,
this mode runs the hook `govc-object-mode-hook`, as the final step
during initialization.

### govc-object-mode-map

Keybinding     | Description
---------------|------------------------------------------------------------
<kbd>J</kbd>   | JSON object selection via govc object
<kbd>N</kbd>   | Next managed object reference
<kbd>O</kbd>   | Object browser via govc object
<kbd>DEL</kbd> | Parent object selection if reachable, otherwise prompt with `govc-object-history`
<kbd>RET</kbd> | Expand object selection via govc object

## govc-metric-mode

Major mode for handling a govc metric.

In addition to any hooks its parent mode `govc-tabulated-list-mode` might have run,
this mode runs the hook `govc-metric-mode-hook`, as the final step
during initialization.

### govc-metric-mode-map

Keybinding     | Description
---------------|------------------------------------------------------------
<kbd>RET</kbd> | Sample metrics
<kbd>P</kbd>   | Plot metric sample
<kbd>s</kbd>   | Select metric names
