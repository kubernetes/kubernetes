/*
Copyright (c) 2014 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package mo

type IsManagedEntity interface {
	GetManagedEntity() ManagedEntity
}

func (m ComputeResource) GetManagedEntity() ManagedEntity {
	return m.ManagedEntity
}

func (m Datacenter) GetManagedEntity() ManagedEntity {
	return m.ManagedEntity
}

func (m Datastore) GetManagedEntity() ManagedEntity {
	return m.ManagedEntity
}

func (m DistributedVirtualSwitch) GetManagedEntity() ManagedEntity {
	return m.ManagedEntity
}

func (m Folder) GetManagedEntity() ManagedEntity {
	return m.ManagedEntity
}

func (m HostSystem) GetManagedEntity() ManagedEntity {
	return m.ManagedEntity
}

func (m Network) GetManagedEntity() ManagedEntity {
	return m.ManagedEntity
}

func (m ResourcePool) GetManagedEntity() ManagedEntity {
	return m.ManagedEntity
}

func (m VirtualMachine) GetManagedEntity() ManagedEntity {
	return m.ManagedEntity
}
