// Copyright 2015 Google Inc. All Rights Reserved.
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

package oomparser

import (
	"fmt"
	"testing"
	"time"

	"github.com/euank/go-kmsg-parser/kmsgparser"
	"github.com/stretchr/testify/assert"
)

const startLine = "ruby invoked oom-killer: gfp_mask=0x201da, order=0, oom_score_adj=0"
const endLine = "Killed process 19667 (evil-program2) total-vm:1460016kB, anon-rss:1414008kB, file-rss:4kB"
const containerLine = "Task in /mem2 killed as a result of limit of /mem3"
const containerLogFile = "containerOomExampleLog.txt"
const systemLogFile = "systemOomExampleLog.txt"

func createExpectedContainerOomInstance(t *testing.T) *OomInstance {
	const longForm = "Jan _2 15:04:05 2006"
	deathTime, err := time.ParseInLocation(longForm, fmt.Sprintf("Jan  5 15:19:27 %d", time.Now().Year()), time.Local)
	if err != nil {
		t.Fatalf("could not parse expected time when creating expected container oom instance. Had error %v", err)
		return nil
	}
	return &OomInstance{
		Pid:                 13536,
		ProcessName:         "memorymonster",
		TimeOfDeath:         deathTime,
		ContainerName:       "/mem2",
		VictimContainerName: "/mem2",
	}
}

func createExpectedSystemOomInstance(t *testing.T) *OomInstance {
	const longForm = "Jan _2 15:04:05 2006"
	deathTime, err := time.ParseInLocation(longForm, fmt.Sprintf("Jan 28 19:58:45 %d", time.Now().Year()), time.Local)
	if err != nil {
		t.Fatalf("could not parse expected time when creating expected system oom instance. Had error %v", err)
		return nil
	}
	return &OomInstance{
		Pid:                 1532,
		ProcessName:         "badsysprogram",
		TimeOfDeath:         deathTime,
		ContainerName:       "/",
		VictimContainerName: "",
	}
}

func TestGetContainerName(t *testing.T) {
	currentOomInstance := new(OomInstance)
	err := getContainerName(startLine, currentOomInstance)
	if err != nil {
		t.Errorf("bad line fed to getContainerName should yield no error, but had error %v", err)
	}
	if currentOomInstance.ContainerName != "" {
		t.Errorf("bad line fed to getContainerName yielded no container name but set it to %s", currentOomInstance.ContainerName)
	}
	err = getContainerName(containerLine, currentOomInstance)
	if err != nil {
		t.Errorf("container line fed to getContainerName should yield no error, but had error %v", err)
	}
	if currentOomInstance.ContainerName != "/mem2" {
		t.Errorf("getContainerName should have set containerName to /mem2, not %s", currentOomInstance.ContainerName)
	}
	if currentOomInstance.VictimContainerName != "/mem3" {
		t.Errorf("getContainerName should have set victimContainerName to /mem3, not %s", currentOomInstance.VictimContainerName)
	}
}

func TestGetProcessNamePid(t *testing.T) {
	currentOomInstance := new(OomInstance)
	couldParseLine, err := getProcessNamePid(startLine, currentOomInstance)
	if err != nil {
		t.Errorf("bad line fed to getProcessNamePid should yield no error, but had error %v", err)
	}
	if couldParseLine {
		t.Errorf("bad line fed to getProcessNamePid should return false but returned %v", couldParseLine)
	}

	couldParseLine, err = getProcessNamePid(endLine, currentOomInstance)
	if err != nil {
		t.Errorf("good line fed to getProcessNamePid should yield no error, but had error %v", err)
	}
	if !couldParseLine {
		t.Errorf("good line fed to getProcessNamePid should return true but returned %v", couldParseLine)
	}
	if currentOomInstance.ProcessName != "evil-program2" {
		t.Errorf("getProcessNamePid should have set processName to evil-program2, not %s", currentOomInstance.ProcessName)
	}
	if currentOomInstance.Pid != 19667 {
		t.Errorf("getProcessNamePid should have set PID to 19667, not %d", currentOomInstance.Pid)
	}
}

func TestCheckIfStartOfMessages(t *testing.T) {
	couldParseLine := checkIfStartOfOomMessages(endLine)
	if couldParseLine {
		t.Errorf("bad line fed to checkIfStartOfMessages should return false but returned %v", couldParseLine)
	}
	couldParseLine = checkIfStartOfOomMessages(startLine)
	if !couldParseLine {
		t.Errorf("start line fed to checkIfStartOfMessages should return true but returned %v", couldParseLine)
	}
}

func TestLastLineRegex(t *testing.T) {
	processNames := []string{"foo", "python3.4", "foo-bar", "Plex Media Server", "x86_64-pc-linux-gnu-c++-5.4.0", "[", "()", `"with quotes"`}
	for _, name := range processNames {
		line := fmt.Sprintf("Jan 21 22:01:49 localhost kernel: [62279.421192] Killed process 1234 (%s) total-vm:1460016kB, anon-rss:1414008kB, file-rss:4kB", name)
		oomInfo := &OomInstance{}
		getProcessNamePid(line, oomInfo)
		assert.Equal(t, 1234, oomInfo.Pid)
		assert.Equal(t, name, oomInfo.ProcessName)
	}
}

func TestStreamOOMs(t *testing.T) {
	mockMsgs := make(chan kmsgparser.Message)
	p := &OomParser{
		parser: &mockKmsgParser{
			messages: mockMsgs,
		},
	}

	oomsOut := make(chan *OomInstance)

	go func() {
		p.StreamOoms(oomsOut)
	}()

	writeAll := func(m []string, t time.Time) {
		for _, msg := range m {
			mockMsgs <- kmsgparser.Message{
				Message:   msg,
				Timestamp: t,
			}
		}
	}

	type in struct {
		msgs []string
		time time.Time
	}

	testTime := time.Unix(0xf331f4ee, 0)
	testTime2 := time.Unix(0xfa51f001, 0)
	testPairs := []struct {
		in  []in
		out []*OomInstance
	}{
		{
			in: []in{{
				time: testTime,
				msgs: []string{
					"memorymonster invoked oom-killer: gfp_mask=0xd0, order=0, oom_score_adj=0",
					"memorymonster cpuset=/ mems_allowed=0",
					"CPU: 5 PID: 13536 Comm: memorymonster Tainted: P           OX 3.13.0-43-generic #72-Ubuntu",
					"Hardware name: Hewlett-Packard HP Z420 Workstation/1589, BIOS J61 v03.65 12/19/2013",
					" ffff88072ae10800 ffff8807a4835c48 ffffffff81720bf6 ffff8807a8e86000",
					" ffff8807a4835cd0 ffffffff8171b4b1 0000000000000246 ffff88072ae10800",
					" ffff8807a4835c90 ffff8807a4835ca0 ffffffff811522a7 0000000000000001",
					"Call Trace:",
					" [<ffffffff81720bf6>] dump_stack+0x45/0x56",
					" [<ffffffff8171b4b1>] dump_header+0x7f/0x1f1",
					" [<ffffffff811522a7>] ? find_lock_task_mm+0x27/0x70",
					" [<ffffffff811526de>] oom_kill_process+0x1ce/0x330",
					" [<ffffffff812d6ce5>] ? security_capable_noaudit+0x15/0x20",
					" [<ffffffff811b491c>] mem_cgroup_oom_synchronize+0x51c/0x560",
					" [<ffffffff811b3e50>] ? mem_cgroup_charge_common+0xa0/0xa0",
					" [<ffffffff81152e64>] pagefault_out_of_memory+0x14/0x80",
					" [<ffffffff81719aa1>] mm_fault_error+0x8e/0x180",
					" [<ffffffff8172cf31>] __do_page_fault+0x4a1/0x560",
					" [<ffffffff810a0255>] ? set_next_entity+0x95/0xb0",
					" [<ffffffff81012609>] ? __switch_to+0x169/0x4c0",
					" [<ffffffff8172d00a>] do_page_fault+0x1a/0x70",
					" [<ffffffff81729468>] page_fault+0x28/0x30",
					"Task in /mem2 killed as a result of limit of /mem2",
					"memory: usage 980kB, limit 980kB, failcnt 4152239",
					"memory+swap: usage 0kB, limit 18014398509481983kB, failcnt 0",
					"kmem: usage 0kB, limit 18014398509481983kB, failcnt 0",
					"Memory cgroup stats for /mem2: cache:0KB rss:980KB rss_huge:0KB mapped_file:0KB writeback:20KB inactive_anon:560KB active_anon:420KB inactive_file:0KB active_file:0KB unevictable:0KB",
					"[ pid ]   uid  tgid total_vm      rss nr_ptes swapents oom_score_adj name",
					"[13536] 275858 13536  8389663      343   16267  8324326             0 memorymonster",
					"Memory cgroup out of memory: Kill process 13536 (memorymonster) score 996 or sacrifice child",
					"Killed process 13536 (memorymonster) total-vm:33558652kB, anon-rss:920kB, file-rss:452kB",
				},
			}},
			out: []*OomInstance{{
				TimeOfDeath:         testTime,
				ContainerName:       "/mem2",
				ProcessName:         "memorymonster",
				Pid:                 13536,
				VictimContainerName: "/mem2",
			}},
		},
		{
			in: []in{{
				time: testTime,
				msgs: []string{
					"badsysprogram invoked oom-killer: gfp_mask=0x280da, order=0, oom_score_adj=0",
					"badsysprogram cpuset=/ mems_allowed=0",
					"CPU: 0 PID: 1532 Comm: badsysprogram Not tainted 3.13.0-27-generic #50-Ubuntu",
					"Hardware name: Google Google, BIOS Google 01/01/2011",
					" 0000000000000000 ffff880069715a90 ffffffff817199c4 ffff8800680d8000",
					" ffff880069715b18 ffffffff817142ff 0000000000000000 0000000000000000",
					" 0000000000000000 0000000000000000 0000000000000000 0000000000000000",
					"Call Trace:",
					" [<ffffffff817199c4>] dump_stack+0x45/0x56",
					" [<ffffffff817142ff>] dump_header+0x7f/0x1f1",
					" [<ffffffff8115196e>] oom_kill_process+0x1ce/0x330",
					" [<ffffffff812d3395>] ? security_capable_noaudit+0x15/0x20",
					" [<ffffffff811520a4>] out_of_memory+0x414/0x450",
					" [<ffffffff81158377>] __alloc_pages_nodemask+0xa87/0xb20",
					" [<ffffffff811985da>] alloc_pages_vma+0x9a/0x140",
					" [<ffffffff8117909b>] handle_mm_fault+0xb2b/0xf10",
					" [<ffffffff81725924>] __do_page_fault+0x184/0x560",
					" [<ffffffff8101b7d9>] ? sched_clock+0x9/0x10",
					" [<ffffffff8109d13d>] ? sched_clock_local+0x1d/0x80",
					" [<ffffffff811112ec>] ? acct_account_cputime+0x1c/0x20",
					" [<ffffffff8109d76b>] ? account_user_time+0x8b/0xa0",
					" [<ffffffff8109dd84>] ? vtime_account_user+0x54/0x60",
					" [<ffffffff81725d1a>] do_page_fault+0x1a/0x70",
					" [<ffffffff81722188>] page_fault+0x28/0x30",
					"Mem-Info:",
					"Node 0 DMA per-cpu:",
					"CPU    0: hi:    0, btch:   1 usd:   0",
					"Node 0 DMA32 per-cpu:",
					"CPU    0: hi:  186, btch:  31 usd:  86",
					"active_anon:405991 inactive_anon:57 isolated_anon:0",
					" active_file:35 inactive_file:69 isolated_file:0",
					" unevictable:0 dirty:0 writeback:0 unstable:0",
					" free:12929 slab_reclaimable:1635 slab_unreclaimable:1919",
					" mapped:34 shmem:70 pagetables:1423 bounce:0",
					" free_cma:0",
					"Node 0 DMA free:7124kB min:412kB low:512kB high:616kB active_anon:8508kB inactive_anon:4kB active_file:0kB inactive_file:0kB unevictable:0kB isolated(anon):0kB isolated(file):0kB present:15992kB managed:15908kB mlocked:0kB dirty:0kB writeback:0kB mapped:0kB shmem:4kB slab_reclaimable:16kB slab_unreclaimable:16kB kernel_stack:0kB pagetables:12kB unstable:0kB bounce:0kB free_cma:0kB writeback_tmp:0kB pages_scanned:0 all_unreclaimable? yes",
					"lowmem_reserve[]: 0 1679 1679 1679",
					"Node 0 DMA32 free:44592kB min:44640kB low:55800kB high:66960kB active_anon:1615456kB inactive_anon:224kB active_file:140kB inactive_file:276kB unevictable:0kB isolated(anon):0kB isolated(file):0kB present:1765368kB managed:1722912kB mlocked:0kB dirty:0kB writeback:0kB mapped:136kB shmem:276kB slab_reclaimable:6524kB slab_unreclaimable:7660kB kernel_stack:592kB pagetables:5680kB unstable:0kB bounce:0kB free_cma:0kB writeback_tmp:0kB pages_scanned:819 all_unreclaimable? yes",
					"lowmem_reserve[]: 0 0 0 0",
					"Node 0 DMA: 5*4kB (UM) 6*8kB (UEM) 7*16kB (UEM) 1*32kB (M) 2*64kB (UE) 3*128kB (UEM) 1*256kB (E) 2*512kB (EM) 3*1024kB (UEM) 1*2048kB (R) 0*4096kB = 7124kB",
					"Node 0 DMA32: 74*4kB (UEM) 125*8kB (UEM) 78*16kB (UEM) 26*32kB (UE) 12*64kB (UEM) 4*128kB (UE) 4*256kB (UE) 2*512kB (E) 11*1024kB (UE) 7*2048kB (UE) 3*4096kB (UR) = 44592kB",
					"Node 0 hugepages_total=0 hugepages_free=0 hugepages_surp=0 hugepages_size=2048kB",
					"204 total pagecache pages",
					"0 pages in swap cache",
					"Swap cache stats: add 0, delete 0, find 0/0",
					"Free swap  = 0kB",
					"Total swap = 0kB",
					"445340 pages RAM",
					"0 pages HighMem/MovableOnly",
					"10614 pages reserved",
					"[ pid ]   uid  tgid total_vm      rss nr_ptes swapents oom_score_adj name",
					"[  273]     0   273     4869       50      13        0             0 upstart-udev-br",
					"[  293]     0   293    12802      154      28        0         -1000 systemd-udevd",
					"[  321]     0   321     3819       54      12        0             0 upstart-file-br",
					"[  326]   102   326     9805      109      24        0             0 dbus-daemon",
					"[  334]   101   334    63960       94      26        0             0 rsyslogd",
					"[  343]     0   343    10863      102      26        0             0 systemd-logind",
					"[  546]     0   546     3815       60      13        0             0 upstart-socket-",
					"[  710]     0   710     2556      587       8        0             0 dhclient",
					"[  863]     0   863     3955       48      13        0             0 getty",
					"[  865]     0   865     3955       50      13        0             0 getty",
					"[  867]     0   867     3955       51      13        0             0 getty",
					"[  868]     0   868     3955       51      12        0             0 getty",
					"[  870]     0   870     3955       49      13        0             0 getty",
					"[  915]     0   915     5914       61      16        0             0 cron",
					"[ 1015]     0  1015    10885     1524      25        0             0 manage_addresse",
					"[ 1028]     0  1028     3955       49      13        0             0 getty",
					"[ 1033]     0  1033     3197       48      12        0             0 getty",
					"[ 1264]     0  1264    11031     1635      26        0             0 manage_accounts",
					"[ 1268]     0  1268    15341      180      33        0         -1000 sshd",
					"[ 1313]   104  1313     6804      154      17        0             0 ntpd",
					"[ 1389]     0  1389    25889      255      55        0             0 sshd",
					"[ 1407]  1020  1407    25889      255      52        0             0 sshd",
					"[ 1408]  1020  1408     5711      581      17        0             0 bash",
					"[ 1425]     0  1425    25889      256      53        0             0 sshd",
					"[ 1443]  1020  1443    25889      257      52        0             0 sshd",
					"[ 1444]  1020  1444     5711      581      16        0             0 bash",
					"[ 1476]  1020  1476     1809       25       9        0             0 tail",
					"[ 1532]  1020  1532   410347   398810     788        0             0 badsysprogram",
					"Out of memory: Kill process 1532 (badsysprogram) score 919 or sacrifice child",
					"Killed process 1532 (badsysprogram) total-vm:1641388kB, anon-rss:1595164kB, file-rss:76kB",
				},
			}},
			out: []*OomInstance{{
				Pid:                 1532,
				ProcessName:         "badsysprogram",
				TimeOfDeath:         testTime,
				ContainerName:       "/",
				VictimContainerName: "",
			}},
		},
		{ // Multiple OOMs
			// These were generated via `docker run -m 20M euank/gunpowder-memhog 2G; docker run -m 300M euank/gunpowder-memhog 800M`
			// followed by nabbing output from `/dev/kmsg` and stripping the syslog-ish prefixes `kmsgparser` will handle anyways.
			in: []in{
				{
					time: testTime,
					msgs: []string{
						"docker0: port 2(veth380a1cd) entered disabled state",
						"device veth380a1cd left promiscuous mode",
						"docker0: port 2(veth380a1cd) entered disabled state",
						"docker0: port 2(vethcd0dbfb) entered blocking state",
						"docker0: port 2(vethcd0dbfb) entered disabled state",
						"device vethcd0dbfb entered promiscuous mode",
						"IPv6: ADDRCONF(NETDEV_UP): vethcd0dbfb: link is not ready",
						"IPv6: ADDRCONF(NETDEV_CHANGE): vethcd0dbfb: link becomes ready",
						"docker0: port 2(vethcd0dbfb) entered blocking state",
						"docker0: port 2(vethcd0dbfb) entered forwarding state",
						"docker0: port 2(vethcd0dbfb) entered disabled state",
						"eth0: renamed from vethbcd01c4",
						"docker0: port 2(vethcd0dbfb) entered blocking state",
						"docker0: port 2(vethcd0dbfb) entered forwarding state",
						"gunpowder-memho invoked oom-killer: gfp_mask=0x24000c0(GFP_KERNEL), order=0, oom_score_adj=0",
						"gunpowder-memho cpuset=2e088fe462e25e60be1dafafe2c05c47bda1a97978648d10ad2b7484fc0b8f50 mems_allowed=0",
						"CPU: 0 PID: 1381 Comm: gunpowder-memho Tainted: G           O    4.8.0-gentoo #2",
						"Hardware name: LENOVO 20BSCTO1WW/20BSCTO1WW, BIOS N14ET32W (1.10 ) 08/13/2015",
						" 0000000000000000 ffff8800968e3ca0 ffffffff8137ad47 ffff8800968e3d68",
						" ffff8800b74ee540 ffff8800968e3d00 ffffffff811261dd 0000000000000003",
						" 0000000000000000 0000000000000001 0000000000000246 0000000000000202",
						"Call Trace:",
						" [<ffffffff8137ad47>] dump_stack+0x4d/0x63",
						" [<ffffffff811261dd>] dump_header+0x58/0x1c8",
						" [<ffffffff810e85fe>] oom_kill_process+0x7e/0x362",
						" [<ffffffff811221a8>] ? mem_cgroup_iter+0x109/0x23e",
						" [<ffffffff811239dc>] mem_cgroup_out_of_memory+0x241/0x299",
						" [<ffffffff81124447>] mem_cgroup_oom_synchronize+0x273/0x28c",
						" [<ffffffff81120839>] ? __mem_cgroup_insert_exceeded+0x76/0x76",
						" [<ffffffff810e8b46>] pagefault_out_of_memory+0x1f/0x76",
						" [<ffffffff81038f38>] mm_fault_error+0x56/0x108",
						" [<ffffffff81039355>] __do_page_fault+0x36b/0x3ee",
						" [<ffffffff81039405>] do_page_fault+0xc/0xe",
						" [<ffffffff81560082>] page_fault+0x22/0x30",
						"Task in /docker/2e088fe462e25e60be1dafafe2c05c47bda1a97978648d10ad2b7484fc0b8f50 killed as a result of limit of /docker/2e088fe462e25e60be1dafafe2c05c47bda1a97978648d10ad2b7484fc0b8f50",
						"memory: usage 20480kB, limit 20480kB, failcnt 1204",
						"memory+swap: usage 40940kB, limit 40960kB, failcnt 6",
						"kmem: usage 220kB, limit 9007199254740988kB, failcnt 0",
						"Memory cgroup stats for /docker/2e088fe462e25e60be1dafafe2c05c47bda1a97978648d10ad2b7484fc0b8f50: cache:0KB rss:20260KB rss_huge:0KB mapped_file:0KB dirty:0KB writeback:1016KB swap:20460KB inactive_anon:10232KB active_anon:10028KB inactive_file:0KB active_file:0KB unevictable:0KB",
						"[ pid ]   uid  tgid total_vm      rss nr_ptes nr_pmds swapents oom_score_adj name",
						"[ 1381]     0  1381   530382     5191      34       4     5489             0 gunpowder-memho",
						"Memory cgroup out of memory: Kill process 1381 (gunpowder-memho) score 1046 or sacrifice child",
						"Killed process 1381 (gunpowder-memho) total-vm:2121528kB, anon-rss:18624kB, file-rss:2140kB, shmem-rss:0kB",
						"oom_reaper: reaped process 1381 (gunpowder-memho), now anon-rss:0kB, file-rss:0kB, shmem-rss:0kB",
						"docker0: port 2(vethcd0dbfb) entered disabled state",
						"vethbcd01c4: renamed from eth0",
						"docker0: port 2(vethcd0dbfb) entered disabled state",
						"device vethcd0dbfb left promiscuous mode",
						"docker0: port 2(vethcd0dbfb) entered disabled state",
						"docker0: port 2(veth4cb51e1) entered blocking state",
						"docker0: port 2(veth4cb51e1) entered disabled state",
						"device veth4cb51e1 entered promiscuous mode",
					},
				},
				{
					time: testTime2,
					msgs: []string{
						"IPv6: ADDRCONF(NETDEV_UP): veth4cb51e1: link is not ready",
						"docker0: port 2(veth4cb51e1) entered blocking state",
						"docker0: port 2(veth4cb51e1) entered forwarding state",
						"IPv6: ADDRCONF(NETDEV_CHANGE): veth4cb51e1: link becomes ready",
						"docker0: port 2(veth4cb51e1) entered disabled state",
						"eth0: renamed from veth4b89c12",
						"docker0: port 2(veth4cb51e1) entered blocking state",
						"docker0: port 2(veth4cb51e1) entered forwarding state",
						"gunpowder-memho invoked oom-killer: gfp_mask=0x24000c0(GFP_KERNEL), order=0, oom_score_adj=0",
						"gunpowder-memho cpuset=6c6fcab8562fd3150854986b78552c732f234fd405b624207b8843528a145e70 mems_allowed=0",
						"CPU: 0 PID: 1667 Comm: gunpowder-memho Tainted: G           O    4.8.0-gentoo #2",
						"Hardware name: LENOVO 20BSCTO1WW/20BSCTO1WW, BIOS N14ET32W (1.10 ) 08/13/2015",
						" 0000000000000000 ffff88008137fca0 ffffffff8137ad47 ffff88008137fd68",
						" ffff8801c75b0c40 ffff88008137fd00 ffffffff811261dd 0000000000000003",
						" 0000000000000000 0000000000000001 0000000000000246 0000000000000202",
						"Call Trace:",
						" [<ffffffff8137ad47>] dump_stack+0x4d/0x63",
						" [<ffffffff811261dd>] dump_header+0x58/0x1c8",
						" [<ffffffff810e85fe>] oom_kill_process+0x7e/0x362",
						" [<ffffffff811221a8>] ? mem_cgroup_iter+0x109/0x23e",
						" [<ffffffff811239dc>] mem_cgroup_out_of_memory+0x241/0x299",
						" [<ffffffff81124447>] mem_cgroup_oom_synchronize+0x273/0x28c",
						" [<ffffffff81120839>] ? __mem_cgroup_insert_exceeded+0x76/0x76",
						" [<ffffffff810e8b46>] pagefault_out_of_memory+0x1f/0x76",
						" [<ffffffff81038f38>] mm_fault_error+0x56/0x108",
						" [<ffffffff81039355>] __do_page_fault+0x36b/0x3ee",
						" [<ffffffff81039405>] do_page_fault+0xc/0xe",
						" [<ffffffff81560082>] page_fault+0x22/0x30",
						"Task in /docker/6c6fcab8562fd3150854986b78552c732f234fd405b624207b8843528a145e70 killed as a result of limit of /docker/6c6fcab8562fd3150854986b78552c732f234fd405b624207b8843528a145e70",
						"memory: usage 307112kB, limit 307200kB, failcnt 35982",
						"memory+swap: usage 614400kB, limit 614400kB, failcnt 11",
						"kmem: usage 1308kB, limit 9007199254740988kB, failcnt 0",
						"Memory cgroup stats for /docker/6c6fcab8562fd3150854986b78552c732f234fd405b624207b8843528a145e70: cache:0KB rss:305804KB rss_huge:0KB mapped_file:0KB dirty:0KB writeback:55884KB swap:307288KB inactive_anon:152940KB active_anon:152832KB inactive_file:0KB active_file:0KB unevictable:0KB",
						"[ pid ]   uid  tgid total_vm      rss nr_ptes nr_pmds swapents oom_score_adj name",
						"[ 1667]     0  1667   210894    62557     315       4    91187             0 gunpowder-memho",
						"Memory cgroup out of memory: Kill process 1667 (gunpowder-memho) score 1003 or sacrifice child",
						"Killed process 1667 (gunpowder-memho) total-vm:843576kB, anon-rss:248180kB, file-rss:2048kB, shmem-rss:0kB",
						"oom_reaper: reaped process 1667 (gunpowder-memho), now anon-rss:0kB, file-rss:0kB, shmem-rss:0kB",
						"docker0: port 2(veth4cb51e1) entered disabled state",
						"veth4b89c12: renamed from eth0",
						"docker0: port 2(veth4cb51e1) entered blocking state",
						"docker0: port 2(veth4cb51e1) entered forwarding state",
						"docker0: port 2(veth4cb51e1) entered disabled state",
						"device veth4cb51e1 left promiscuous mode",
						"docker0: port 2(veth4cb51e1) entered disabled state",
					},
				},
			},
			out: []*OomInstance{
				{
					Pid:                 1381,
					ProcessName:         "gunpowder-memho",
					TimeOfDeath:         testTime,
					ContainerName:       "/docker/2e088fe462e25e60be1dafafe2c05c47bda1a97978648d10ad2b7484fc0b8f50",
					VictimContainerName: "/docker/2e088fe462e25e60be1dafafe2c05c47bda1a97978648d10ad2b7484fc0b8f50",
				},
				{
					Pid:                 1667,
					ProcessName:         "gunpowder-memho",
					TimeOfDeath:         testTime2,
					ContainerName:       "/docker/6c6fcab8562fd3150854986b78552c732f234fd405b624207b8843528a145e70",
					VictimContainerName: "/docker/6c6fcab8562fd3150854986b78552c732f234fd405b624207b8843528a145e70",
				},
			},
		},
	}

	for _, pair := range testPairs {
		go func() {
			for _, x := range pair.in {
				writeAll(x.msgs, x.time)
			}
		}()
		for _, expected := range pair.out {
			oom := <-oomsOut
			assert.Equal(t, expected, oom)
		}

		select {
		case oom := <-oomsOut:
			t.Errorf("did not expect any remaining OOMs, got %+v", oom)
		default:
		}

	}
}

type mockKmsgParser struct {
	messages chan kmsgparser.Message
}

func (m *mockKmsgParser) SeekEnd() error {
	return nil
}

func (m *mockKmsgParser) Parse() <-chan kmsgparser.Message {
	return m.messages
}

func (m *mockKmsgParser) SetLogger(kmsgparser.Logger) {}
func (m *mockKmsgParser) Close() error {
	close(m.messages)
	return nil
}
