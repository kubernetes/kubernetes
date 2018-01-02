#!/bin/bash -e

types="Datacenter HostSystem ClusterComputeResource ResourcePool VirtualMachine Datastore VirtualApp"

for type in $types ; do
  echo "$type..."

  obj=$(govc ls -t "$type" ./... | head -n 1)
  if [ -z "$obj" ] ; then
    echo "...no instances found"
    continue
  fi

  if ! govc metric.info "$obj" 2>/dev/null ; then
    echo "...N/A" # Datacenter, Datastore on ESX for example
    continue
  fi

  govc metric.ls "$obj" | xargs govc metric.sample -n 5 "$obj"
done
