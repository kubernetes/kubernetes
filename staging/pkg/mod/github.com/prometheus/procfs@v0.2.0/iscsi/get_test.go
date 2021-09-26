// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package iscsi_test

import (
	"reflect"
	"testing"

	"github.com/prometheus/procfs/iscsi"
)

func TestGetStats(t *testing.T) {
	tests := []struct {
		stat *iscsi.Stats
	}{
		{
			stat: &iscsi.Stats{
				Name: "iqn.2003-01.org.linux-iscsi.osd1.x8664:sn.8888bbbbddd0",
				Tpgt: []iscsi.TPGT{
					{
						Name:     "tpgt_1",
						TpgtPath: "../fixtures/sys/kernel/config/target/iscsi/iqn.2003-01.org.linux-iscsi.osd1.x8664:sn.8888bbbbddd0/tpgt_1",
						IsEnable: true,
						Luns: []iscsi.LUN{
							{
								Name:       "lun_0",
								LunPath:    "../fixtures/sys/kernel/config/target/iscsi/iqn.2003-01.org.linux-iscsi.osd1.x8664:sn.8888bbbbddd0/tpgt_1/lun/lun_0",
								Backstore:  "rd_mcp",
								ObjectName: "ramdisk_lio_1G",
								TypeNumber: "119",
							},
						},
					},
				},
				RootPath: "../fixtures/sys/kernel/config/target/iscsi",
			},
		},
		{
			stat: &iscsi.Stats{
				Name: "iqn.2003-01.org.linux-iscsi.osd1.x8664:sn.abcd1abcd2ab",
				Tpgt: []iscsi.TPGT{
					{
						Name:     "tpgt_1",
						TpgtPath: "../fixtures/sys/kernel/config/target/iscsi/iqn.2003-01.org.linux-iscsi.osd1.x8664:sn.abcd1abcd2ab/tpgt_1",
						IsEnable: true,
						Luns: []iscsi.LUN{
							{
								Name:       "lun_0",
								LunPath:    "../fixtures/sys/kernel/config/target/iscsi/iqn.2003-01.org.linux-iscsi.osd1.x8664:sn.abcd1abcd2ab/tpgt_1/lun/lun_0",
								Backstore:  "iblock",
								ObjectName: "block_lio_rbd1",
								TypeNumber: "0",
							},
						},
					},
				},
				RootPath: "../fixtures/sys/kernel/config/target/iscsi",
			},
		},
		{
			stat: &iscsi.Stats{
				Name: "iqn.2016-11.org.linux-iscsi.igw.x86:dev.rbd0",
				Tpgt: []iscsi.TPGT{
					{
						Name:     "tpgt_1",
						TpgtPath: "../fixtures/sys/kernel/config/target/iscsi/iqn.2016-11.org.linux-iscsi.igw.x86:dev.rbd0/tpgt_1",
						IsEnable: true,
						Luns: []iscsi.LUN{
							{
								Name:       "lun_0",
								LunPath:    "../fixtures/sys/kernel/config/target/iscsi/iqn.2016-11.org.linux-iscsi.igw.x86:dev.rbd0/tpgt_1/lun/lun_0",
								Backstore:  "fileio",
								ObjectName: "file_lio_1G",
								TypeNumber: "1",
							},
						},
					},
				},
				RootPath: "../fixtures/sys/kernel/config/target/iscsi",
			},
		},
		{
			stat: &iscsi.Stats{
				Name: "iqn.2016-11.org.linux-iscsi.igw.x86:sn.ramdemo",
				Tpgt: []iscsi.TPGT{
					{
						Name:     "tpgt_1",
						TpgtPath: "../fixtures/sys/kernel/config/target/iscsi/iqn.2016-11.org.linux-iscsi.igw.x86:sn.ramdemo/tpgt_1",
						IsEnable: true,
						Luns: []iscsi.LUN{
							{
								Name:       "lun_0",
								LunPath:    "../fixtures/sys/kernel/config/target/iscsi/iqn.2016-11.org.linux-iscsi.igw.x86:sn.ramdemo/tpgt_1/lun/lun_0",
								Backstore:  "rbd",
								ObjectName: "iscsi-images-demo",
								TypeNumber: "0",
							},
						},
					},
				},
				RootPath: "../fixtures/sys/kernel/config/target/iscsi",
			},
		},
	}

	readTests := []struct {
		read  uint64
		write uint64
		iops  uint64
	}{
		{10325, 40325, 204950},
		{20095, 71235, 104950},
		{10195, 30195, 301950},
		{1504, 4733, 1234},
	}

	sysconfigfs, err := iscsi.NewFS("../fixtures/sys", "../fixtures/sys/kernel/config")
	if err != nil {
		t.Fatalf("failed to access xfs fs: %v", err)
	}
	sysfsStat, err := sysconfigfs.ISCSIStats()
	statSize := len(sysfsStat)
	if statSize != 4 {
		t.Errorf("fixtures size does not match %d", statSize)
	}
	if err != nil {
		t.Errorf("unexpected test fixtures")
	}

	for i, stat := range sysfsStat {
		want, have := tests[i].stat, stat
		if !reflect.DeepEqual(want, have) {
			t.Errorf("unexpected iSCSI stats:\nwant:\n%v\nhave:\n%v", want, have)
		} else {
			readMB, writeMB, iops, err := iscsi.ReadWriteOPS(stat.RootPath+"/"+stat.Name,
				stat.Tpgt[0].Name, stat.Tpgt[0].Luns[0].Name)
			if err != nil {
				t.Errorf("unexpected iSCSI ReadWriteOPS path %s %s %s",
					stat.Name, stat.Tpgt[0].Name, stat.Tpgt[0].Luns[0].Name)
				t.Errorf("%v", err)
			}
			if !reflect.DeepEqual(readTests[i].read, readMB) {
				t.Errorf("unexpected iSCSI read data :\nwant:\n%v\nhave:\n%v", readTests[i].read, readMB)
			}
			if !reflect.DeepEqual(readTests[i].write, writeMB) {
				t.Errorf("unexpected iSCSI write data :\nwant:\n%v\nhave:\n%v", readTests[i].write, writeMB)
			}
			if !reflect.DeepEqual(readTests[i].iops, iops) {
				t.Errorf("unexpected iSCSI iops data :\nwant:\n%v\nhave:\n%v", readTests[i].iops, iops)
			}
			if stat.Tpgt[0].Luns[0].Backstore == "rd_mcp" {
				haveRdmcp, err := sysconfigfs.GetRDMCPPath("119", "ramdisk_lio_1G")
				if err != nil {
					t.Errorf("fail rdmcp error %v", err)
				}
				// Name ObjectName
				wantRdmcp := &iscsi.RDMCP{"rd_mcp_" + stat.Tpgt[0].Luns[0].TypeNumber, stat.Tpgt[0].Luns[0].ObjectName}

				if !reflect.DeepEqual(wantRdmcp, haveRdmcp) {
					t.Errorf("unexpected rdmcp data :\nwant:\n%v\nhave:\n%v", wantRdmcp, haveRdmcp)
				}
			} else if stat.Tpgt[0].Luns[0].Backstore == "iblock" {
				haveIblock, err := sysconfigfs.GetIblockUdev("0", "block_lio_rbd1")
				if err != nil {
					t.Errorf("fail iblock error %v", err)
				}
				// Name Bnumber ObjectName Iblock
				wantIblock := &iscsi.IBLOCK{"iblock_" + stat.Tpgt[0].Luns[0].TypeNumber, stat.Tpgt[0].Luns[0].TypeNumber, stat.Tpgt[0].Luns[0].ObjectName, "/dev/rbd1"}
				if !reflect.DeepEqual(wantIblock, haveIblock) {
					t.Errorf("unexpected iblock data :\nwant:\n%v\nhave:\n%v", wantIblock, haveIblock)
				}
			} else if stat.Tpgt[0].Luns[0].Backstore == "fileio" {
				haveFileIO, err := sysconfigfs.GetFileioUdev("1", "file_lio_1G")
				if err != nil {
					t.Errorf("fail fileio error %v", err)
				}
				// Name, Fnumber, ObjectName, Filename
				wantFileIO := &iscsi.FILEIO{"fileio_" + stat.Tpgt[0].Luns[0].TypeNumber, stat.Tpgt[0].Luns[0].TypeNumber, "file_lio_1G", "/home/iscsi/file_back_1G"}
				if !reflect.DeepEqual(wantFileIO, haveFileIO) {
					t.Errorf("unexpected fileio data :\nwant:\n%v\nhave:\n%v", wantFileIO, haveFileIO)
				}
			} else if stat.Tpgt[0].Luns[0].Backstore == "rbd" {
				haveRBD, err := sysconfigfs.GetRBDMatch("0", "iscsi-images-demo")
				if err != nil {
					t.Errorf("fail rbd error %v", err)
				}
				// Name, Rnumber, Pool, Image
				wantRBD := &iscsi.RBD{"rbd_" + stat.Tpgt[0].Luns[0].TypeNumber, stat.Tpgt[0].Luns[0].TypeNumber, "iscsi-images", "demo"}
				if !reflect.DeepEqual(wantRBD, haveRBD) {
					t.Errorf("unexpected fileio data :\nwant:\n%v\nhave:\n%v", wantRBD, haveRBD)
				}
			}
		}
	}
}
