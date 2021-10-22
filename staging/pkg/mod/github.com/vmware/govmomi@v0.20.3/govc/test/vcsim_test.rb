#!/usr/bin/env ruby

require 'rbvmomi'

VIM = RbVmomi::VIM

vim = VIM.connect :user => "user", :password => "pass", :insecure => true, :host => "localhost", :port => ARGV[0] || 8989

dc = vim.serviceInstance.content.rootFolder.traverse("DC0", VIM::Datacenter) or abort "datacenter not found"
vm = dc.vmFolder.traverse("DC0_H0_VM1", VIM::VirtualMachine) or abort "VM not found"

if vm.runtime.powerState == "poweredOn"
  vm.PowerOffVM_Task.wait_for_completion
end

vm.PowerOnVM_Task.wait_for_completion

begin
  vm.PowerOnVM_Task.wait_for_completion
  raise "expected InvalidPowerState"
rescue VIM::InvalidPowerState
  # ok
end
