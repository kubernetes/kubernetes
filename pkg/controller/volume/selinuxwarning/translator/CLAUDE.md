# Package: translator

## Purpose
Provides SELinux label translation utilities for the kube-controller-manager's SELinux warning controller. Since KCM cannot access node SELinux configuration, this translator handles label conversion without OS defaults.

## Key Types

- **ControllerSELinuxTranslator**: Implementation of SELinuxLabelTranslator for use in kube-controller-manager.

## Key Functions

- **SELinuxEnabled()**: Always returns true (controller is explicitly enabled).
- **SELinuxOptionsToFileLabel(opts)**: Converts SELinuxOptions to a colon-separated label string without defaulting missing components.
- **Conflicts(labelA, labelB)**: Determines if two SELinux labels conflict.

## Conflict Logic

The Conflicts function handles partial labels gracefully:
- Empty components (user, role, type) are incomparable and don't cause conflicts.
- The level component (4th part) always causes a conflict if different.
- Examples:
  - `system_u:system_r:container_t:s0:c1,c2` does NOT conflict with `:::s0:c1,c2` (may expand to same)
  - `system_u:system_r:container_t:s0:c1,c2` DOES conflict with `:::s0:c98,c99` (different level)
  - `:::s0:c1,c2` DOES conflict with `""` or `:::`

## Design Notes

- KCM runs in a container and cannot access /etc/selinux on nodes.
- Different nodes may have different SELinux configurations.
- The translator is intentionally conservative: it cannot detect all conflicts but avoids false positives.
- Implements the util.SELinuxLabelTranslator interface.
