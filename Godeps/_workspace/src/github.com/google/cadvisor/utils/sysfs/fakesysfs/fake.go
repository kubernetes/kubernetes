// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package fakesysfs

import (
	"os"
	"time"

	"github.com/google/cadvisor/utils/sysfs"
)

// If we extend sysfs to support more interfaces, it might be worth making this a mock instead of a fake.
type FileInfo struct {
	EntryName string
}

func (self *FileInfo) Name() string {
	return self.EntryName
}

func (self *FileInfo) Size() int64 {
	return 1234567
}

func (self *FileInfo) Mode() os.FileMode {
	return 0
}

func (self *FileInfo) ModTime() time.Time {
	return time.Time{}
}

func (self *FileInfo) IsDir() bool {
	return true
}

func (self *FileInfo) Sys() interface{} {
	return nil
}

type FakeSysFs struct {
	info  FileInfo
	cache sysfs.CacheInfo
}

func (self *FakeSysFs) GetBlockDevices() ([]os.FileInfo, error) {
	self.info.EntryName = "sda"
	return []os.FileInfo{&self.info}, nil
}

func (self *FakeSysFs) GetBlockDeviceSize(name string) (string, error) {
	return "1234567", nil
}

func (self *FakeSysFs) GetBlockDeviceScheduler(name string) (string, error) {
	return "noop deadline [cfq]", nil
}

func (self *FakeSysFs) GetBlockDeviceNumbers(name string) (string, error) {
	return "8:0\n", nil
}

func (self *FakeSysFs) GetNetworkDevices() ([]os.FileInfo, error) {
	return []os.FileInfo{&self.info}, nil
}

func (self *FakeSysFs) GetNetworkAddress(name string) (string, error) {
	return "42:01:02:03:04:f4\n", nil
}

func (self *FakeSysFs) GetNetworkMtu(name string) (string, error) {
	return "1024\n", nil
}

func (self *FakeSysFs) GetNetworkSpeed(name string) (string, error) {
	return "1000\n", nil
}

func (self *FakeSysFs) GetNetworkStatValue(name string, stat string) (uint64, error) {
	return 1024, nil
}

func (self *FakeSysFs) GetCaches(id int) ([]os.FileInfo, error) {
	self.info.EntryName = "index0"
	return []os.FileInfo{&self.info}, nil
}

func (self *FakeSysFs) GetCacheInfo(cpu int, cache string) (sysfs.CacheInfo, error) {
	return self.cache, nil
}

func (self *FakeSysFs) SetCacheInfo(cache sysfs.CacheInfo) {
	self.cache = cache
}

func (self *FakeSysFs) SetEntryName(name string) {
	self.info.EntryName = name
}

func (self *FakeSysFs) GetSystemUUID() (string, error) {
	return "1F862619-BA9F-4526-8F85-ECEAF0C97430", nil
}
