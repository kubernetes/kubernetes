# Package: sysctl

## Purpose
The `sysctl` package validates and manages sysctl settings for pods. It implements allowlists for safe sysctls and provides pod admission handling to enforce sysctl policies.

## Key Types/Structs

- **patternAllowlist**: Implements `lifecycle.PodAdmitHandler`. Stores allowed sysctls and sysctl prefixes (patterns ending in *) with their namespace information.
- **sysctl**: Internal struct holding sysctl name and minimum kernel version requirement.

## Key Functions

- **NewAllowlist**: Creates a new allowlist from a list of sysctl patterns. Validates patterns and checks they are known to be namespaced.
- **SafeSysctlAllowlist**: Returns the list of safe sysctls based on kernel version. Safe sysctls are namespaced and isolated per container/pod.
- **Admit**: PodAdmitHandler implementation that validates all sysctls in a pod's SecurityContext against the allowlist.
- **validateSysctl**: Checks if a sysctl is allowlisted and validates namespace constraints (e.g., net sysctls forbidden with hostNetwork).
- **ConvertPodSysctlsVariableToDotsSeparator**: Normalizes sysctl names to use dots as separators per Linux sysctl conventions.

## Safe Sysctls (Linux)

Built-in safe sysctls include:
- `kernel.shm_rmid_forced`
- `net.ipv4.ip_local_port_range`
- `net.ipv4.tcp_syncookies`
- `net.ipv4.ping_group_range`
- `net.ipv4.ip_unprivileged_port_start`
- `net.ipv4.ip_local_reserved_ports` (kernel 3.16+)
- `net.ipv4.tcp_keepalive_time` (kernel 4.5+)
- `net.ipv4.tcp_fin_timeout` (kernel 4.6+)
- Various TCP memory/keepalive sysctls (version-dependent)

## Design Notes

- Sysctls must be namespaced (IPC or Net namespace) to be allowed.
- Net sysctls are forbidden when `hostNetwork: true`.
- IPC sysctls are forbidden when `hostIPC: true`.
- Kernel version is checked to filter out sysctls not available in older kernels.
- ForbiddenReason constant is used for pod admission rejection messages.
