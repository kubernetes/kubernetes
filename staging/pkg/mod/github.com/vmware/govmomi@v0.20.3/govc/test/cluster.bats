#!/usr/bin/env bats

load test_helper

@test "cluster.group" {
  vcsim_env -cluster 2 -host 4 -vm 8

  run govc cluster.group.ls -cluster DC0_C0
  assert_success "" # no groups

  run govc cluster.group.ls -cluster DC0_C0 -name my_vm_group
  assert_failure # group does not exist

  run govc cluster.group.create -cluster DC0_C0 -name my_vm_group -vm DC0_C0_H{0,1}
  assert_failure # -vm or -host required

  run govc cluster.group.create -cluster DC0_C0 -name my_vm_group -vm DC0_C0_H{0,1}
  assert_failure # -vm with HostSystem type args

  run govc cluster.group.create -cluster DC0_C0 -name my_vm_group -vm DC0_C0_RP0_VM{0,1}
  assert_success

  run govc cluster.group.ls -cluster DC0_C0
  assert_success "my_vm_group"

  run govc cluster.group.create -cluster DC0_C0 -name my_vm_group -vm DC0_C0_RP0_VM{0,1}
  assert_failure # group exists

  run govc cluster.group.ls -cluster DC0_C0 -name my_vm_group
  assert_success "$(printf "%s\n" DC0_C0_RP0_VM{0,1})"

  run govc cluster.group.change -cluster DC0_C0 -name my_vm_group DC0_C0_RP0_VM{0,1,2}
  assert_success

  run govc cluster.group.ls -cluster DC0_C0 -name my_vm_group
  assert_success "$(printf "%s\n" DC0_C0_RP0_VM{0,1,2})"

  run govc cluster.group.create -cluster DC0_C0 -name my_host_group -host DC0_C0_RP0_VM{0,1}
  assert_failure # -host with VirtualMachine type args

  run govc cluster.group.create -cluster DC0_C0 -name my_host_group -host DC0_C0_H{0,1}
  assert_success

  run govc cluster.group.ls -cluster DC0_C0 -name my_host_group
  assert_success

  run govc cluster.group.remove -cluster DC0_C0 -name my_vm_group
  assert_success

  run govc cluster.group.remove -cluster DC0_C0 -name my_vm_group
  assert_failure # group does not exist

  run govc cluster.group.ls -cluster DC0_C0
  assert_success "my_host_group"
}

@test "cluster.rule" {
  vcsim_env -cluster 2 -host 4 -vm 8

  run govc cluster.rule.ls -cluster DC0_C0
  assert_success "" # no rules

  run govc object.collect -json /DC0/host/DC0_C0 configurationEx.rule
  assert_success

  run govc cluster.rule.ls -cluster DC0_C0 -name pod1
  assert_failure # rule does not exist

  run govc cluster.rule.create -cluster DC0_C0 -name pod1 -affinity DC0_C0_RP0_VM0
  assert_failure # requires >= 2 VMs

  run govc cluster.rule.create -cluster DC0_C0 -name pod1 -affinity DC0_C0_RP0_VM{0,1,2,3}
  assert_success

  run govc cluster.rule.ls -cluster DC0_C0
  assert_success "pod1"

  run govc cluster.rule.ls -cluster DC0_C0 -l=true
  assert_success "pod1 (ClusterAffinityRuleSpec)"

  run govc cluster.rule.ls -cluster DC0_C0 -name pod1
  assert_success "$(printf "%s\n" DC0_C0_RP0_VM{0,1,2,3})"

  run govc cluster.rule.ls -cluster DC0_C0 -name pod1 -l=true
  assert_success "$(printf "%s (VM)\n" DC0_C0_RP0_VM{0,1,2,3})"

  run govc cluster.rule.info -cluster DC0_C0
  assert_success "$(cat <<_EOF_
Rule: pod1
  Type: ClusterAffinityRuleSpec
  VM: DC0_C0_RP0_VM0
  VM: DC0_C0_RP0_VM1
  VM: DC0_C0_RP0_VM2
  VM: DC0_C0_RP0_VM3
_EOF_
)"

  run govc cluster.rule.change -cluster DC0_C0 -name pod1 DC0_C0_RP0_VM{2,3,4}
  assert_success

  run govc cluster.rule.ls -cluster DC0_C0 -name pod1
  assert_success "$(printf "%s\n" DC0_C0_RP0_VM{2,3,4})"

  run govc object.collect -json /DC0/host/DC0_C0 configurationEx.rule
  assert_success

  run govc cluster.group.create -cluster DC0_C0 -name my_vms -vm DC0_C0_RP0_VM{0,1,2,3}
  assert_success

  run govc cluster.group.create -cluster DC0_C0 -name even_hosts -host DC0_C0_H{0,2}
  assert_success

  run govc cluster.group.create -cluster DC0_C0 -name odd_hosts -host DC0_C0_H{1,3}
  assert_success

  run govc cluster.rule.create -cluster DC0_C0 -name pod2 -enable -mandatory -vm-host -vm-group my_vms -host-affine-group even_hosts -host-anti-affine-group odd_hosts
  assert_success

  run govc cluster.rule.remove -cluster DC0_C0 -name pod1
  assert_success

  run govc cluster.rule.ls -cluster DC0_C0 -l
  assert_success "pod2 (ClusterVmHostRuleInfo)"

  run govc cluster.rule.ls -cluster DC0_C0 -name pod2
  assert_success "$(printf "%s\n" {my_vms,even_hosts,odd_hosts})"

  run govc cluster.rule.ls -cluster DC0_C0 -name pod2 -l
  assert_success "$(printf "%s\n" {'my_vms (vmGroupName)','even_hosts (affineHostGroupName)','odd_hosts (antiAffineHostGroupName)'})"

  run govc cluster.rule.remove -cluster DC0_C0 -name pod1 -depends
  assert_failure # rule does not exist

  run govc cluster.rule.create -cluster DC0_C0 -name my_deps -depends
  assert_failure # requires 2 groups

  run govc cluster.group.create -cluster DC0_C0 -name my_app -vm DC0_C0_RP0_VM{4,5}
  assert_success

  run govc cluster.group.create -cluster DC0_C0 -name my_db -vm DC0_C0_RP0_VM{6,7}
  assert_success

  run govc cluster.rule.create -cluster DC0_C0 -name my_deps -depends my_app my_db
  assert_success

  run govc cluster.rule.ls -cluster DC0_C0 -l
  assert_success "$(printf "%s\n" {'pod2 (ClusterVmHostRuleInfo)','my_deps (ClusterDependencyRuleInfo)'})"

  run govc cluster.rule.ls -cluster DC0_C0 -name my_deps
  assert_success "$(printf "%s\n" {'my_app','my_db'})"

  run govc cluster.rule.ls -cluster DC0_C0 -name my_deps -l
  assert_success "$(printf "%s\n" {'my_app (VmGroup)','my_db (DependsOnVmGroup)'})"

  run govc cluster.rule.info -cluster DC0_C0
  assert_success "$(cat <<_EOF_
Rule: pod2
  Type: ClusterVmHostRuleInfo
  vmGroupName: my_vms
  affineHostGroupName even_hosts
  antiAffineHostGroupName odd_hosts
Rule: my_deps
  Type: ClusterDependencyRuleInfo
  VmGroup my_app
  DependsOnVmGroup my_db
_EOF_
)"

}

@test "cluster.vm" {
  vcsim_env -host 4 -vm 8

  run govc cluster.override.info
  assert_success "" # no overrides == empty output

  run govc cluster.override.change
  assert_failure # -vm required

  run govc cluster.override.change -vm DC0_C0_RP0_VM0
  assert_failure # no changes specified

  # DRS override
  query=".Overrides[] | select(.Name == \"DC0_C0_RP0_VM0\") | .DRS.Enabled"

  run govc cluster.override.change -vm DC0_C0_RP0_VM0 -drs-enabled=false
  assert_success
  [ "$(govc cluster.override.info -json | jq "$query")" == "false" ]

  run govc cluster.override.change -vm DC0_C0_RP0_VM0 -drs-enabled=true
  assert_success
  [ "$(govc cluster.override.info -json | jq "$query")" == "true" ]

  # DAS override
  query=".Overrides[] | select(.Name == \"DC0_C0_RP0_VM0\") | .DAS.DasSettings.RestartPriority"

  [ "$(govc cluster.override.info -json | jq -r "$query")" != "high" ]

  run govc cluster.override.change -vm DC0_C0_RP0_VM0 -ha-restart-priority high
  assert_success
  [ "$(govc cluster.override.info -json | jq -r "$query")" == "high" ]

  run govc cluster.override.remove -vm DC0_C0_RP0_VM0
  assert_success
  run govc cluster.override.info
  assert_success "" # no overrides == empty output

  run govc cluster.override.change -vm DC0_C0_RP0_VM0 -drs-mode=manual
  assert_success

  run govc cluster.override.remove -vm DC0_C0_RP0_VM0
  assert_success
}
