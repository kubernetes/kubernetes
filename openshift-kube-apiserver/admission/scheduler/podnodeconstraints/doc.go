/*
Package podnodeconstraints contains the PodNodeConstraints admission
control plugin. This plugin allows administrators to set policy
governing the use of the NodeName and NodeSelector attributes in pod
specs.

Enabling this plugin will prevent the use of the NodeName field in Pod
templates for users and serviceaccounts which lack the "pods/binding"
permission, and which don't belong to groups which have the
"pods/binding" permission.

This plugin will also prevent users, serviceaccounts and groups which
lack the "pods/binding" permission from specifying the NodeSelector field
in Pod templates for labels which appear in the
nodeSelectorLabelBlacklist list field.

Configuration

The plugin is configured via a PodNodeConstraintsConfig object in the
origin and kubernetes Master configs:

admissionConfig:
  pluginConfig:
    PodNodeConstraints:
      configuration:
        apiVersion: v1
        kind: PodNodeConstraintsConfig
        nodeSelectorLabelBlacklist:
          - label1
          - label2
...
kubernetesMasterConfig:
  admissionConfig:
    pluginConfig:
      PodNodeConstraints:
        configuration:
          apiVersion: v1
          kind: PodNodeConstraintsConfig
          nodeSelectorLabelBlacklist:
            - label1
            - label2
*/

package podnodeconstraints
