// +build linux

package mount

import (
	"bytes"
	"testing"
)

const (
	fedoraMountinfo = `15 35 0:3 / /proc rw,nosuid,nodev,noexec,relatime shared:5 - proc proc rw
    16 35 0:14 / /sys rw,nosuid,nodev,noexec,relatime shared:6 - sysfs sysfs rw,seclabel
    17 35 0:5 / /dev rw,nosuid shared:2 - devtmpfs devtmpfs rw,seclabel,size=8056484k,nr_inodes=2014121,mode=755
    18 16 0:15 / /sys/kernel/security rw,nosuid,nodev,noexec,relatime shared:7 - securityfs securityfs rw
    19 16 0:13 / /sys/fs/selinux rw,relatime shared:8 - selinuxfs selinuxfs rw
    20 17 0:16 / /dev/shm rw,nosuid,nodev shared:3 - tmpfs tmpfs rw,seclabel
    21 17 0:10 / /dev/pts rw,nosuid,noexec,relatime shared:4 - devpts devpts rw,seclabel,gid=5,mode=620,ptmxmode=000
    22 35 0:17 / /run rw,nosuid,nodev shared:21 - tmpfs tmpfs rw,seclabel,mode=755
    23 16 0:18 / /sys/fs/cgroup rw,nosuid,nodev,noexec shared:9 - tmpfs tmpfs rw,seclabel,mode=755
    24 23 0:19 / /sys/fs/cgroup/systemd rw,nosuid,nodev,noexec,relatime shared:10 - cgroup cgroup rw,xattr,release_agent=/usr/lib/systemd/systemd-cgroups-agent,name=systemd
    25 16 0:20 / /sys/fs/pstore rw,nosuid,nodev,noexec,relatime shared:20 - pstore pstore rw
    26 23 0:21 / /sys/fs/cgroup/cpuset rw,nosuid,nodev,noexec,relatime shared:11 - cgroup cgroup rw,cpuset,clone_children
    27 23 0:22 / /sys/fs/cgroup/cpu,cpuacct rw,nosuid,nodev,noexec,relatime shared:12 - cgroup cgroup rw,cpuacct,cpu,clone_children
    28 23 0:23 / /sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime shared:13 - cgroup cgroup rw,memory,clone_children
    29 23 0:24 / /sys/fs/cgroup/devices rw,nosuid,nodev,noexec,relatime shared:14 - cgroup cgroup rw,devices,clone_children
    30 23 0:25 / /sys/fs/cgroup/freezer rw,nosuid,nodev,noexec,relatime shared:15 - cgroup cgroup rw,freezer,clone_children
    31 23 0:26 / /sys/fs/cgroup/net_cls rw,nosuid,nodev,noexec,relatime shared:16 - cgroup cgroup rw,net_cls,clone_children
    32 23 0:27 / /sys/fs/cgroup/blkio rw,nosuid,nodev,noexec,relatime shared:17 - cgroup cgroup rw,blkio,clone_children
    33 23 0:28 / /sys/fs/cgroup/perf_event rw,nosuid,nodev,noexec,relatime shared:18 - cgroup cgroup rw,perf_event,clone_children
    34 23 0:29 / /sys/fs/cgroup/hugetlb rw,nosuid,nodev,noexec,relatime shared:19 - cgroup cgroup rw,hugetlb,clone_children
    35 1 253:2 / / rw,relatime shared:1 - ext4 /dev/mapper/ssd-root--f20 rw,seclabel,data=ordered
    36 15 0:30 / /proc/sys/fs/binfmt_misc rw,relatime shared:22 - autofs systemd-1 rw,fd=38,pgrp=1,timeout=300,minproto=5,maxproto=5,direct
    37 17 0:12 / /dev/mqueue rw,relatime shared:23 - mqueue mqueue rw,seclabel
    38 35 0:31 / /tmp rw shared:24 - tmpfs tmpfs rw,seclabel
    39 17 0:32 / /dev/hugepages rw,relatime shared:25 - hugetlbfs hugetlbfs rw,seclabel
    40 16 0:7 / /sys/kernel/debug rw,relatime shared:26 - debugfs debugfs rw
    41 16 0:33 / /sys/kernel/config rw,relatime shared:27 - configfs configfs rw
    42 35 0:34 / /var/lib/nfs/rpc_pipefs rw,relatime shared:28 - rpc_pipefs sunrpc rw
    43 15 0:35 / /proc/fs/nfsd rw,relatime shared:29 - nfsd sunrpc rw
    45 35 8:17 / /boot rw,relatime shared:30 - ext4 /dev/sdb1 rw,seclabel,data=ordered
    46 35 253:4 / /home rw,relatime shared:31 - ext4 /dev/mapper/ssd-home rw,seclabel,data=ordered
    47 35 253:5 / /var/lib/libvirt/images rw,noatime,nodiratime shared:32 - ext4 /dev/mapper/ssd-virt rw,seclabel,discard,data=ordered
    48 35 253:12 / /mnt/old rw,relatime shared:33 - ext4 /dev/mapper/HelpDeskRHEL6-FedoraRoot rw,seclabel,data=ordered
    121 22 0:36 / /run/user/1000/gvfs rw,nosuid,nodev,relatime shared:104 - fuse.gvfsd-fuse gvfsd-fuse rw,user_id=1000,group_id=1000
    124 16 0:37 / /sys/fs/fuse/connections rw,relatime shared:107 - fusectl fusectl rw
    165 38 253:3 / /tmp/mnt rw,relatime shared:147 - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered
    167 35 253:15 / /var/lib/docker/devicemapper/mnt/aae4076022f0e2b80a2afbf8fc6df450c52080191fcef7fb679a73e6f073e5c2 rw,relatime shared:149 - ext4 /dev/mapper/docker-253:2-425882-aae4076022f0e2b80a2afbf8fc6df450c52080191fcef7fb679a73e6f073e5c2 rw,seclabel,discard,stripe=16,data=ordered
    171 35 253:16 / /var/lib/docker/devicemapper/mnt/c71be651f114db95180e472f7871b74fa597ee70a58ccc35cb87139ddea15373 rw,relatime shared:153 - ext4 /dev/mapper/docker-253:2-425882-c71be651f114db95180e472f7871b74fa597ee70a58ccc35cb87139ddea15373 rw,seclabel,discard,stripe=16,data=ordered
    175 35 253:17 / /var/lib/docker/devicemapper/mnt/1bac6ab72862d2d5626560df6197cf12036b82e258c53d981fa29adce6f06c3c rw,relatime shared:157 - ext4 /dev/mapper/docker-253:2-425882-1bac6ab72862d2d5626560df6197cf12036b82e258c53d981fa29adce6f06c3c rw,seclabel,discard,stripe=16,data=ordered
    179 35 253:18 / /var/lib/docker/devicemapper/mnt/d710a357d77158e80d5b2c55710ae07c94e76d34d21ee7bae65ce5418f739b09 rw,relatime shared:161 - ext4 /dev/mapper/docker-253:2-425882-d710a357d77158e80d5b2c55710ae07c94e76d34d21ee7bae65ce5418f739b09 rw,seclabel,discard,stripe=16,data=ordered
    183 35 253:19 / /var/lib/docker/devicemapper/mnt/6479f52366114d5f518db6837254baab48fab39f2ac38d5099250e9a6ceae6c7 rw,relatime shared:165 - ext4 /dev/mapper/docker-253:2-425882-6479f52366114d5f518db6837254baab48fab39f2ac38d5099250e9a6ceae6c7 rw,seclabel,discard,stripe=16,data=ordered
    187 35 253:20 / /var/lib/docker/devicemapper/mnt/8d9df91c4cca5aef49eeb2725292aab324646f723a7feab56be34c2ad08268e1 rw,relatime shared:169 - ext4 /dev/mapper/docker-253:2-425882-8d9df91c4cca5aef49eeb2725292aab324646f723a7feab56be34c2ad08268e1 rw,seclabel,discard,stripe=16,data=ordered
    191 35 253:21 / /var/lib/docker/devicemapper/mnt/c8240b768603d32e920d365dc9d1dc2a6af46cd23e7ae819947f969e1b4ec661 rw,relatime shared:173 - ext4 /dev/mapper/docker-253:2-425882-c8240b768603d32e920d365dc9d1dc2a6af46cd23e7ae819947f969e1b4ec661 rw,seclabel,discard,stripe=16,data=ordered
    195 35 253:22 / /var/lib/docker/devicemapper/mnt/2eb3a01278380bbf3ed12d86ac629eaa70a4351301ee307a5cabe7b5f3b1615f rw,relatime shared:177 - ext4 /dev/mapper/docker-253:2-425882-2eb3a01278380bbf3ed12d86ac629eaa70a4351301ee307a5cabe7b5f3b1615f rw,seclabel,discard,stripe=16,data=ordered
    199 35 253:23 / /var/lib/docker/devicemapper/mnt/37a17fb7c9d9b80821235d5f2662879bd3483915f245f9b49cdaa0e38779b70b rw,relatime shared:181 - ext4 /dev/mapper/docker-253:2-425882-37a17fb7c9d9b80821235d5f2662879bd3483915f245f9b49cdaa0e38779b70b rw,seclabel,discard,stripe=16,data=ordered
    203 35 253:24 / /var/lib/docker/devicemapper/mnt/aea459ae930bf1de913e2f29428fd80ee678a1e962d4080019d9f9774331ee2b rw,relatime shared:185 - ext4 /dev/mapper/docker-253:2-425882-aea459ae930bf1de913e2f29428fd80ee678a1e962d4080019d9f9774331ee2b rw,seclabel,discard,stripe=16,data=ordered
    207 35 253:25 / /var/lib/docker/devicemapper/mnt/928ead0bc06c454bd9f269e8585aeae0a6bd697f46dc8754c2a91309bc810882 rw,relatime shared:189 - ext4 /dev/mapper/docker-253:2-425882-928ead0bc06c454bd9f269e8585aeae0a6bd697f46dc8754c2a91309bc810882 rw,seclabel,discard,stripe=16,data=ordered
    211 35 253:26 / /var/lib/docker/devicemapper/mnt/0f284d18481d671644706e7a7244cbcf63d590d634cc882cb8721821929d0420 rw,relatime shared:193 - ext4 /dev/mapper/docker-253:2-425882-0f284d18481d671644706e7a7244cbcf63d590d634cc882cb8721821929d0420 rw,seclabel,discard,stripe=16,data=ordered
    215 35 253:27 / /var/lib/docker/devicemapper/mnt/d9dd16722ab34c38db2733e23f69e8f4803ce59658250dd63e98adff95d04919 rw,relatime shared:197 - ext4 /dev/mapper/docker-253:2-425882-d9dd16722ab34c38db2733e23f69e8f4803ce59658250dd63e98adff95d04919 rw,seclabel,discard,stripe=16,data=ordered
    219 35 253:28 / /var/lib/docker/devicemapper/mnt/bc4500479f18c2c08c21ad5282e5f826a016a386177d9874c2764751c031d634 rw,relatime shared:201 - ext4 /dev/mapper/docker-253:2-425882-bc4500479f18c2c08c21ad5282e5f826a016a386177d9874c2764751c031d634 rw,seclabel,discard,stripe=16,data=ordered
    223 35 253:29 / /var/lib/docker/devicemapper/mnt/7770c8b24eb3d5cc159a065910076938910d307ab2f5d94e1dc3b24c06ee2c8a rw,relatime shared:205 - ext4 /dev/mapper/docker-253:2-425882-7770c8b24eb3d5cc159a065910076938910d307ab2f5d94e1dc3b24c06ee2c8a rw,seclabel,discard,stripe=16,data=ordered
    227 35 253:30 / /var/lib/docker/devicemapper/mnt/c280cd3d0bf0aa36b478b292279671624cceafc1a67eaa920fa1082601297adf rw,relatime shared:209 - ext4 /dev/mapper/docker-253:2-425882-c280cd3d0bf0aa36b478b292279671624cceafc1a67eaa920fa1082601297adf rw,seclabel,discard,stripe=16,data=ordered
    231 35 253:31 / /var/lib/docker/devicemapper/mnt/8b59a7d9340279f09fea67fd6ad89ddef711e9e7050eb647984f8b5ef006335f rw,relatime shared:213 - ext4 /dev/mapper/docker-253:2-425882-8b59a7d9340279f09fea67fd6ad89ddef711e9e7050eb647984f8b5ef006335f rw,seclabel,discard,stripe=16,data=ordered
    235 35 253:32 / /var/lib/docker/devicemapper/mnt/1a28059f29eda821578b1bb27a60cc71f76f846a551abefabce6efd0146dce9f rw,relatime shared:217 - ext4 /dev/mapper/docker-253:2-425882-1a28059f29eda821578b1bb27a60cc71f76f846a551abefabce6efd0146dce9f rw,seclabel,discard,stripe=16,data=ordered
    239 35 253:33 / /var/lib/docker/devicemapper/mnt/e9aa60c60128cad1 rw,relatime shared:221 - ext4 /dev/mapper/docker-253:2-425882-e9aa60c60128cad1 rw,seclabel,discard,stripe=16,data=ordered
    243 35 253:34 / /var/lib/docker/devicemapper/mnt/5fec11304b6f4713fea7b6ccdcc1adc0a1966187f590fe25a8227428a8df275d-init rw,relatime shared:225 - ext4 /dev/mapper/docker-253:2-425882-5fec11304b6f4713fea7b6ccdcc1adc0a1966187f590fe25a8227428a8df275d-init rw,seclabel,discard,stripe=16,data=ordered
    247 35 253:35 / /var/lib/docker/devicemapper/mnt/5fec11304b6f4713fea7b6ccdcc1adc0a1966187f590fe25a8227428a8df275d rw,relatime shared:229 - ext4 /dev/mapper/docker-253:2-425882-5fec11304b6f4713fea7b6ccdcc1adc0a1966187f590fe25a8227428a8df275d rw,seclabel,discard,stripe=16,data=ordered
    31 21 0:23 / /DATA/foo_bla_bla rw,relatime - cifs //foo/BLA\040BLA\040BLA/ rw,sec=ntlm,cache=loose,unc=\\foo\BLA BLA BLA,username=my_login,domain=mydomain.com,uid=12345678,forceuid,gid=12345678,forcegid,addr=10.1.30.10,file_mode=0755,dir_mode=0755,nounix,rsize=61440,wsize=65536,actimeo=1`

	ubuntuMountInfo = `15 20 0:14 / /sys rw,nosuid,nodev,noexec,relatime - sysfs sysfs rw
16 20 0:3 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw
17 20 0:5 / /dev rw,relatime - devtmpfs udev rw,size=1015140k,nr_inodes=253785,mode=755
18 17 0:11 / /dev/pts rw,nosuid,noexec,relatime - devpts devpts rw,gid=5,mode=620,ptmxmode=000
19 20 0:15 / /run rw,nosuid,noexec,relatime - tmpfs tmpfs rw,size=205044k,mode=755
20 1 253:0 / / rw,relatime - ext4 /dev/disk/by-label/DOROOT rw,errors=remount-ro,data=ordered
21 15 0:16 / /sys/fs/cgroup rw,relatime - tmpfs none rw,size=4k,mode=755
22 15 0:17 / /sys/fs/fuse/connections rw,relatime - fusectl none rw
23 15 0:6 / /sys/kernel/debug rw,relatime - debugfs none rw
24 15 0:10 / /sys/kernel/security rw,relatime - securityfs none rw
25 19 0:18 / /run/lock rw,nosuid,nodev,noexec,relatime - tmpfs none rw,size=5120k
26 21 0:19 / /sys/fs/cgroup/cpuset rw,relatime - cgroup cgroup rw,cpuset,clone_children
27 19 0:20 / /run/shm rw,nosuid,nodev,relatime - tmpfs none rw
28 21 0:21 / /sys/fs/cgroup/cpu rw,relatime - cgroup cgroup rw,cpu
29 19 0:22 / /run/user rw,nosuid,nodev,noexec,relatime - tmpfs none rw,size=102400k,mode=755
30 15 0:23 / /sys/fs/pstore rw,relatime - pstore none rw
31 21 0:24 / /sys/fs/cgroup/cpuacct rw,relatime - cgroup cgroup rw,cpuacct
32 21 0:25 / /sys/fs/cgroup/memory rw,relatime - cgroup cgroup rw,memory
33 21 0:26 / /sys/fs/cgroup/devices rw,relatime - cgroup cgroup rw,devices
34 21 0:27 / /sys/fs/cgroup/freezer rw,relatime - cgroup cgroup rw,freezer
35 21 0:28 / /sys/fs/cgroup/blkio rw,relatime - cgroup cgroup rw,blkio
36 21 0:29 / /sys/fs/cgroup/perf_event rw,relatime - cgroup cgroup rw,perf_event
37 21 0:30 / /sys/fs/cgroup/hugetlb rw,relatime - cgroup cgroup rw,hugetlb
38 21 0:31 / /sys/fs/cgroup/systemd rw,nosuid,nodev,noexec,relatime - cgroup systemd rw,name=systemd
39 20 0:32 / /var/lib/docker/aufs/mnt/b750fe79269d2ec9a3c593ef05b4332b1d1a02a62b4accb2c21d589ff2f5f2dc rw,relatime - aufs none rw,si=caafa54fdc06525
40 20 0:33 / /var/lib/docker/aufs/mnt/2eed44ac7ce7c75af04f088ed6cb4ce9d164801e91d78c6db65d7ef6d572bba8-init rw,relatime - aufs none rw,si=caafa54f882b525
41 20 0:34 / /var/lib/docker/aufs/mnt/2eed44ac7ce7c75af04f088ed6cb4ce9d164801e91d78c6db65d7ef6d572bba8 rw,relatime - aufs none rw,si=caafa54f8829525
42 20 0:35 / /var/lib/docker/aufs/mnt/16f4d7e96dd612903f425bfe856762f291ff2e36a8ecd55a2209b7d7cd81c30b rw,relatime - aufs none rw,si=caafa54f882d525
43 20 0:36 / /var/lib/docker/aufs/mnt/63ca08b75d7438a9469a5954e003f48ffede73541f6286ce1cb4d7dd4811da7e-init rw,relatime - aufs none rw,si=caafa54f882f525
44 20 0:37 / /var/lib/docker/aufs/mnt/63ca08b75d7438a9469a5954e003f48ffede73541f6286ce1cb4d7dd4811da7e rw,relatime - aufs none rw,si=caafa54f88ba525
45 20 0:38 / /var/lib/docker/aufs/mnt/283f35a910233c756409313be71ecd8fcfef0df57108b8d740b61b3e88860452 rw,relatime - aufs none rw,si=caafa54f88b8525
46 20 0:39 / /var/lib/docker/aufs/mnt/2c6c7253d4090faa3886871fb21bd660609daeb0206588c0602007f7d0f254b1-init rw,relatime - aufs none rw,si=caafa54f88be525
47 20 0:40 / /var/lib/docker/aufs/mnt/2c6c7253d4090faa3886871fb21bd660609daeb0206588c0602007f7d0f254b1 rw,relatime - aufs none rw,si=caafa54f882c525
48 20 0:41 / /var/lib/docker/aufs/mnt/de2b538c97d6366cc80e8658547c923ea1d042f85580df379846f36a4df7049d rw,relatime - aufs none rw,si=caafa54f85bb525
49 20 0:42 / /var/lib/docker/aufs/mnt/94a3d8ed7c27e5b0aa71eba46c736bfb2742afda038e74f2dd6035fb28415b49-init rw,relatime - aufs none rw,si=caafa54fdc00525
50 20 0:43 / /var/lib/docker/aufs/mnt/94a3d8ed7c27e5b0aa71eba46c736bfb2742afda038e74f2dd6035fb28415b49 rw,relatime - aufs none rw,si=caafa54fbaec525
51 20 0:44 / /var/lib/docker/aufs/mnt/6ac1cace985c9fc9bea32234de8b36dba49bdd5e29a2972b327ff939d78a6274 rw,relatime - aufs none rw,si=caafa54f8e1a525
52 20 0:45 / /var/lib/docker/aufs/mnt/dff147033e3a0ef061e1de1ad34256b523d4a8c1fa6bba71a0ab538e8628ff0b-init rw,relatime - aufs none rw,si=caafa54f8e1d525
53 20 0:46 / /var/lib/docker/aufs/mnt/dff147033e3a0ef061e1de1ad34256b523d4a8c1fa6bba71a0ab538e8628ff0b rw,relatime - aufs none rw,si=caafa54f8e1b525
54 20 0:47 / /var/lib/docker/aufs/mnt/cabb117d997f0f93519185aea58389a9762770b7496ed0b74a3e4a083fa45902 rw,relatime - aufs none rw,si=caafa54f810a525
55 20 0:48 / /var/lib/docker/aufs/mnt/e1c8a94ffaa9d532bbbdc6ef771ce8a6c2c06757806ecaf8b68e9108fec65f33-init rw,relatime - aufs none rw,si=caafa54f8529525
56 20 0:49 / /var/lib/docker/aufs/mnt/e1c8a94ffaa9d532bbbdc6ef771ce8a6c2c06757806ecaf8b68e9108fec65f33 rw,relatime - aufs none rw,si=caafa54f852f525
57 20 0:50 / /var/lib/docker/aufs/mnt/16a1526fa445b84ce84f89506d219e87fa488a814063baf045d88b02f21166b3 rw,relatime - aufs none rw,si=caafa54f9e1d525
58 20 0:51 / /var/lib/docker/aufs/mnt/57b9c92e1e368fa7dbe5079f7462e917777829caae732828b003c355fe49da9f-init rw,relatime - aufs none rw,si=caafa54f854d525
59 20 0:52 / /var/lib/docker/aufs/mnt/57b9c92e1e368fa7dbe5079f7462e917777829caae732828b003c355fe49da9f rw,relatime - aufs none rw,si=caafa54f854e525
60 20 0:53 / /var/lib/docker/aufs/mnt/e370c3e286bea027917baa0e4d251262681a472a87056e880dfd0513516dffd9 rw,relatime - aufs none rw,si=caafa54f840a525
61 20 0:54 / /var/lib/docker/aufs/mnt/6b00d3b4f32b41997ec07412b5e18204f82fbe643e7122251cdeb3582abd424e-init rw,relatime - aufs none rw,si=caafa54f8408525
62 20 0:55 / /var/lib/docker/aufs/mnt/6b00d3b4f32b41997ec07412b5e18204f82fbe643e7122251cdeb3582abd424e rw,relatime - aufs none rw,si=caafa54f8409525
63 20 0:56 / /var/lib/docker/aufs/mnt/abd0b5ea5d355a67f911475e271924a5388ee60c27185fcd60d095afc4a09dc7 rw,relatime - aufs none rw,si=caafa54f9eb1525
64 20 0:57 / /var/lib/docker/aufs/mnt/336222effc3f7b89867bb39ff7792ae5412c35c749f127c29159d046b6feedd2-init rw,relatime - aufs none rw,si=caafa54f85bf525
65 20 0:58 / /var/lib/docker/aufs/mnt/336222effc3f7b89867bb39ff7792ae5412c35c749f127c29159d046b6feedd2 rw,relatime - aufs none rw,si=caafa54f85b8525
66 20 0:59 / /var/lib/docker/aufs/mnt/912e1bf28b80a09644503924a8a1a4fb8ed10b808ca847bda27a369919aa52fa rw,relatime - aufs none rw,si=caafa54fbaea525
67 20 0:60 / /var/lib/docker/aufs/mnt/386f722875013b4a875118367abc783fc6617a3cb7cf08b2b4dcf550b4b9c576-init rw,relatime - aufs none rw,si=caafa54f8472525
68 20 0:61 / /var/lib/docker/aufs/mnt/386f722875013b4a875118367abc783fc6617a3cb7cf08b2b4dcf550b4b9c576 rw,relatime - aufs none rw,si=caafa54f8474525
69 20 0:62 / /var/lib/docker/aufs/mnt/5aaebb79ef3097dfca377889aeb61a0c9d5e3795117d2b08d0751473c671dfb2 rw,relatime - aufs none rw,si=caafa54f8c5e525
70 20 0:63 / /var/lib/docker/aufs/mnt/5ba3e493279d01277d583600b81c7c079e691b73c3a2bdea8e4b12a35a418be2-init rw,relatime - aufs none rw,si=caafa54f8c3b525
71 20 0:64 / /var/lib/docker/aufs/mnt/5ba3e493279d01277d583600b81c7c079e691b73c3a2bdea8e4b12a35a418be2 rw,relatime - aufs none rw,si=caafa54f8c3d525
72 20 0:65 / /var/lib/docker/aufs/mnt/2777f0763da4de93f8bebbe1595cc77f739806a158657b033eca06f827b6028a rw,relatime - aufs none rw,si=caafa54f8c3e525
73 20 0:66 / /var/lib/docker/aufs/mnt/5d7445562acf73c6f0ae34c3dd0921d7457de1ba92a587d9e06a44fa209eeb3e-init rw,relatime - aufs none rw,si=caafa54f8c39525
74 20 0:67 / /var/lib/docker/aufs/mnt/5d7445562acf73c6f0ae34c3dd0921d7457de1ba92a587d9e06a44fa209eeb3e rw,relatime - aufs none rw,si=caafa54f854f525
75 20 0:68 / /var/lib/docker/aufs/mnt/06400b526ec18b66639c96efc41a84f4ae0b117cb28dafd56be420651b4084a0 rw,relatime - aufs none rw,si=caafa54f840b525
76 20 0:69 / /var/lib/docker/aufs/mnt/e051d45ec42d8e3e1cc57bb39871a40de486dc123522e9c067fbf2ca6a357785-init rw,relatime - aufs none rw,si=caafa54fdddf525
77 20 0:70 / /var/lib/docker/aufs/mnt/e051d45ec42d8e3e1cc57bb39871a40de486dc123522e9c067fbf2ca6a357785 rw,relatime - aufs none rw,si=caafa54f854b525
78 20 0:71 / /var/lib/docker/aufs/mnt/1ff414fa93fd61ec81b0ab7b365a841ff6545accae03cceac702833aaeaf718f rw,relatime - aufs none rw,si=caafa54f8d85525
79 20 0:72 / /var/lib/docker/aufs/mnt/c661b2f871dd5360e46a2aebf8f970f6d39a2ff64e06979aa0361227c88128b8-init rw,relatime - aufs none rw,si=caafa54f8da3525
80 20 0:73 / /var/lib/docker/aufs/mnt/c661b2f871dd5360e46a2aebf8f970f6d39a2ff64e06979aa0361227c88128b8 rw,relatime - aufs none rw,si=caafa54f8da2525
81 20 0:74 / /var/lib/docker/aufs/mnt/b68b1d4fe4d30016c552398e78b379a39f651661d8e1fa5f2460c24a5e723420 rw,relatime - aufs none rw,si=caafa54f8d81525
82 20 0:75 / /var/lib/docker/aufs/mnt/c5c5979c936cd0153a4c626fa9d69ce4fce7d924cc74fa68b025d2f585031739-init rw,relatime - aufs none rw,si=caafa54f8da1525
83 20 0:76 / /var/lib/docker/aufs/mnt/c5c5979c936cd0153a4c626fa9d69ce4fce7d924cc74fa68b025d2f585031739 rw,relatime - aufs none rw,si=caafa54f8da0525
84 20 0:77 / /var/lib/docker/aufs/mnt/53e10b0329afc0e0d3322d31efaed4064139dc7027fe6ae445cffd7104bcc94f rw,relatime - aufs none rw,si=caafa54f8c35525
85 20 0:78 / /var/lib/docker/aufs/mnt/3bfafd09ff2603e2165efacc2215c1f51afabba6c42d04a68cc2df0e8cc31494-init rw,relatime - aufs none rw,si=caafa54f8db8525
86 20 0:79 / /var/lib/docker/aufs/mnt/3bfafd09ff2603e2165efacc2215c1f51afabba6c42d04a68cc2df0e8cc31494 rw,relatime - aufs none rw,si=caafa54f8dba525
87 20 0:80 / /var/lib/docker/aufs/mnt/90fdd2c03eeaf65311f88f4200e18aef6d2772482712d9aea01cd793c64781b5 rw,relatime - aufs none rw,si=caafa54f8315525
88 20 0:81 / /var/lib/docker/aufs/mnt/7bdf2591c06c154ceb23f5e74b1d03b18fbf6fe96e35fbf539b82d446922442f-init rw,relatime - aufs none rw,si=caafa54f8fc6525
89 20 0:82 / /var/lib/docker/aufs/mnt/7bdf2591c06c154ceb23f5e74b1d03b18fbf6fe96e35fbf539b82d446922442f rw,relatime - aufs none rw,si=caafa54f8468525
90 20 0:83 / /var/lib/docker/aufs/mnt/8cf9a993f50f3305abad3da268c0fc44ff78a1e7bba595ef9de963497496c3f9 rw,relatime - aufs none rw,si=caafa54f8c59525
91 20 0:84 / /var/lib/docker/aufs/mnt/ecc896fd74b21840a8d35e8316b92a08b1b9c83d722a12acff847e9f0ff17173-init rw,relatime - aufs none rw,si=caafa54f846a525
92 20 0:85 / /var/lib/docker/aufs/mnt/ecc896fd74b21840a8d35e8316b92a08b1b9c83d722a12acff847e9f0ff17173 rw,relatime - aufs none rw,si=caafa54f846b525
93 20 0:86 / /var/lib/docker/aufs/mnt/d8c8288ec920439a48b5796bab5883ee47a019240da65e8d8f33400c31bac5df rw,relatime - aufs none rw,si=caafa54f8dbf525
94 20 0:87 / /var/lib/docker/aufs/mnt/ecba66710bcd03199b9398e46c005cd6b68d0266ec81dc8b722a29cc417997c6-init rw,relatime - aufs none rw,si=caafa54f810f525
95 20 0:88 / /var/lib/docker/aufs/mnt/ecba66710bcd03199b9398e46c005cd6b68d0266ec81dc8b722a29cc417997c6 rw,relatime - aufs none rw,si=caafa54fbae9525
96 20 0:89 / /var/lib/docker/aufs/mnt/befc1c67600df449dddbe796c0d06da7caff1d2bbff64cde1f0ba82d224996b5 rw,relatime - aufs none rw,si=caafa54f8dab525
97 20 0:90 / /var/lib/docker/aufs/mnt/c9f470e73d2742629cdc4084a1b2c1a8302914f2aa0d0ec4542371df9a050562-init rw,relatime - aufs none rw,si=caafa54fdc02525
98 20 0:91 / /var/lib/docker/aufs/mnt/c9f470e73d2742629cdc4084a1b2c1a8302914f2aa0d0ec4542371df9a050562 rw,relatime - aufs none rw,si=caafa54f9eb0525
99 20 0:92 / /var/lib/docker/aufs/mnt/2a31f10029f04ff9d4381167a9b739609853d7220d55a56cb654779a700ee246 rw,relatime - aufs none rw,si=caafa54f8c37525
100 20 0:93 / /var/lib/docker/aufs/mnt/8c4261b8e3e4b21ebba60389bd64b6261217e7e6b9fd09e201d5a7f6760f6927-init rw,relatime - aufs none rw,si=caafa54fd173525
101 20 0:94 / /var/lib/docker/aufs/mnt/8c4261b8e3e4b21ebba60389bd64b6261217e7e6b9fd09e201d5a7f6760f6927 rw,relatime - aufs none rw,si=caafa54f8108525
102 20 0:95 / /var/lib/docker/aufs/mnt/eaa0f57403a3dc685268f91df3fbcd7a8423cee50e1a9ee5c3e1688d9d676bb4 rw,relatime - aufs none rw,si=caafa54f852d525
103 20 0:96 / /var/lib/docker/aufs/mnt/9cfe69a2cbffd9bfc7f396d4754f6fe5cc457ef417b277797be3762dfe955a6b-init rw,relatime - aufs none rw,si=caafa54f8d80525
104 20 0:97 / /var/lib/docker/aufs/mnt/9cfe69a2cbffd9bfc7f396d4754f6fe5cc457ef417b277797be3762dfe955a6b rw,relatime - aufs none rw,si=caafa54f8fc3525
105 20 0:98 / /var/lib/docker/aufs/mnt/d1b322ae17613c6adee84e709641a9244ac56675244a89a64dc0075075fcbb83 rw,relatime - aufs none rw,si=caafa54f8c58525
106 20 0:99 / /var/lib/docker/aufs/mnt/d46c2a8e9da7e91ab34fd9c192851c246a4e770a46720bda09e55c7554b9dbbd-init rw,relatime - aufs none rw,si=caafa54f8c63525
107 20 0:100 / /var/lib/docker/aufs/mnt/d46c2a8e9da7e91ab34fd9c192851c246a4e770a46720bda09e55c7554b9dbbd rw,relatime - aufs none rw,si=caafa54f8c67525
108 20 0:101 / /var/lib/docker/aufs/mnt/bc9d2a264158f83a617a069bf17cbbf2a2ba453db7d3951d9dc63cc1558b1c2b rw,relatime - aufs none rw,si=caafa54f8dbe525
109 20 0:102 / /var/lib/docker/aufs/mnt/9e6abb8d72bbeb4d5cf24b96018528015ba830ce42b4859965bd482cbd034e99-init rw,relatime - aufs none rw,si=caafa54f9e0d525
110 20 0:103 / /var/lib/docker/aufs/mnt/9e6abb8d72bbeb4d5cf24b96018528015ba830ce42b4859965bd482cbd034e99 rw,relatime - aufs none rw,si=caafa54f9e1b525
111 20 0:104 / /var/lib/docker/aufs/mnt/d4dca7b02569c732e740071e1c654d4ad282de5c41edb619af1f0aafa618be26 rw,relatime - aufs none rw,si=caafa54f8dae525
112 20 0:105 / /var/lib/docker/aufs/mnt/fea63da40fa1c5ffbad430dde0bc64a8fc2edab09a051fff55b673c40a08f6b7-init rw,relatime - aufs none rw,si=caafa54f8c5c525
113 20 0:106 / /var/lib/docker/aufs/mnt/fea63da40fa1c5ffbad430dde0bc64a8fc2edab09a051fff55b673c40a08f6b7 rw,relatime - aufs none rw,si=caafa54fd172525
114 20 0:107 / /var/lib/docker/aufs/mnt/e60c57499c0b198a6734f77f660cdbbd950a5b78aa23f470ca4f0cfcc376abef rw,relatime - aufs none rw,si=caafa54909c4525
115 20 0:108 / /var/lib/docker/aufs/mnt/099c78e7ccd9c8717471bb1bbfff838c0a9913321ba2f214fbeaf92c678e5b35-init rw,relatime - aufs none rw,si=caafa54909c3525
116 20 0:109 / /var/lib/docker/aufs/mnt/099c78e7ccd9c8717471bb1bbfff838c0a9913321ba2f214fbeaf92c678e5b35 rw,relatime - aufs none rw,si=caafa54909c7525
117 20 0:110 / /var/lib/docker/aufs/mnt/2997be666d58b9e71469759bcb8bd9608dad0e533a1a7570a896919ba3388825 rw,relatime - aufs none rw,si=caafa54f8557525
118 20 0:111 / /var/lib/docker/aufs/mnt/730694eff438ef20569df38dfb38a920969d7ff2170cc9aa7cb32a7ed8147a93-init rw,relatime - aufs none rw,si=caafa54c6e88525
119 20 0:112 / /var/lib/docker/aufs/mnt/730694eff438ef20569df38dfb38a920969d7ff2170cc9aa7cb32a7ed8147a93 rw,relatime - aufs none rw,si=caafa54c6e8e525
120 20 0:113 / /var/lib/docker/aufs/mnt/a672a1e2f2f051f6e19ed1dfbe80860a2d774174c49f7c476695f5dd1d5b2f67 rw,relatime - aufs none rw,si=caafa54c6e15525
121 20 0:114 / /var/lib/docker/aufs/mnt/aba3570e17859f76cf29d282d0d150659c6bd80780fdc52a465ba05245c2a420-init rw,relatime - aufs none rw,si=caafa54f8dad525
122 20 0:115 / /var/lib/docker/aufs/mnt/aba3570e17859f76cf29d282d0d150659c6bd80780fdc52a465ba05245c2a420 rw,relatime - aufs none rw,si=caafa54f8d84525
123 20 0:116 / /var/lib/docker/aufs/mnt/2abc86007aca46fb4a817a033e2a05ccacae40b78ea4b03f8ea616b9ada40e2e rw,relatime - aufs none rw,si=caafa54c6e8b525
124 20 0:117 / /var/lib/docker/aufs/mnt/36352f27f7878e648367a135bd1ec3ed497adcb8ac13577ee892a0bd921d2374-init rw,relatime - aufs none rw,si=caafa54c6e8d525
125 20 0:118 / /var/lib/docker/aufs/mnt/36352f27f7878e648367a135bd1ec3ed497adcb8ac13577ee892a0bd921d2374 rw,relatime - aufs none rw,si=caafa54f8c34525
126 20 0:119 / /var/lib/docker/aufs/mnt/2f95ca1a629cea8363b829faa727dd52896d5561f2c96ddee4f697ea2fc872c2 rw,relatime - aufs none rw,si=caafa54c6e8a525
127 20 0:120 / /var/lib/docker/aufs/mnt/f108c8291654f179ef143a3e07de2b5a34adbc0b28194a0ab17742b6db9a7fb2-init rw,relatime - aufs none rw,si=caafa54f8e19525
128 20 0:121 / /var/lib/docker/aufs/mnt/f108c8291654f179ef143a3e07de2b5a34adbc0b28194a0ab17742b6db9a7fb2 rw,relatime - aufs none rw,si=caafa54fa8c6525
129 20 0:122 / /var/lib/docker/aufs/mnt/c1d04dfdf8cccb3676d5a91e84e9b0781ce40623d127d038bcfbe4c761b27401 rw,relatime - aufs none rw,si=caafa54f8c30525
130 20 0:123 / /var/lib/docker/aufs/mnt/3f4898ffd0e1239aeebf1d1412590cdb7254207fa3883663e2c40cf772e5f05a-init rw,relatime - aufs none rw,si=caafa54c6e1a525
131 20 0:124 / /var/lib/docker/aufs/mnt/3f4898ffd0e1239aeebf1d1412590cdb7254207fa3883663e2c40cf772e5f05a rw,relatime - aufs none rw,si=caafa54c6e1c525
132 20 0:125 / /var/lib/docker/aufs/mnt/5ae3b6fccb1539fc02d420e86f3e9637bef5b711fed2ca31a2f426c8f5deddbf rw,relatime - aufs none rw,si=caafa54c4fea525
133 20 0:126 / /var/lib/docker/aufs/mnt/310bfaf80d57020f2e73b06aeffb0b9b0ca2f54895f88bf5e4d1529ccac58fe0-init rw,relatime - aufs none rw,si=caafa54c6e1e525
134 20 0:127 / /var/lib/docker/aufs/mnt/310bfaf80d57020f2e73b06aeffb0b9b0ca2f54895f88bf5e4d1529ccac58fe0 rw,relatime - aufs none rw,si=caafa54fa8c0525
135 20 0:128 / /var/lib/docker/aufs/mnt/f382bd5aaccaf2d04a59089ac7cb12ec87efd769fd0c14d623358fbfd2a3f896 rw,relatime - aufs none rw,si=caafa54c4fec525
136 20 0:129 / /var/lib/docker/aufs/mnt/50d45e9bb2d779bc6362824085564c7578c231af5ae3b3da116acf7e17d00735-init rw,relatime - aufs none rw,si=caafa54c4fef525
137 20 0:130 / /var/lib/docker/aufs/mnt/50d45e9bb2d779bc6362824085564c7578c231af5ae3b3da116acf7e17d00735 rw,relatime - aufs none rw,si=caafa54c4feb525
138 20 0:131 / /var/lib/docker/aufs/mnt/a9c5ee0854dc083b6bf62b7eb1e5291aefbb10702289a446471ce73aba0d5d7d rw,relatime - aufs none rw,si=caafa54909c6525
139 20 0:134 / /var/lib/docker/aufs/mnt/03a613e7bd5078819d1fd92df4e671c0127559a5e0b5a885cc8d5616875162f0-init rw,relatime - aufs none rw,si=caafa54804fe525
140 20 0:135 / /var/lib/docker/aufs/mnt/03a613e7bd5078819d1fd92df4e671c0127559a5e0b5a885cc8d5616875162f0 rw,relatime - aufs none rw,si=caafa54804fa525
141 20 0:136 / /var/lib/docker/aufs/mnt/7ec3277e5c04c907051caf9c9c35889f5fcd6463e5485971b25404566830bb70 rw,relatime - aufs none rw,si=caafa54804f9525
142 20 0:139 / /var/lib/docker/aufs/mnt/26b5b5d71d79a5b2bfcf8bc4b2280ee829f261eb886745dd90997ed410f7e8b8-init rw,relatime - aufs none rw,si=caafa54c6ef6525
143 20 0:140 / /var/lib/docker/aufs/mnt/26b5b5d71d79a5b2bfcf8bc4b2280ee829f261eb886745dd90997ed410f7e8b8 rw,relatime - aufs none rw,si=caafa54c6ef5525
144 20 0:356 / /var/lib/docker/aufs/mnt/e6ecde9e2c18cd3c75f424c67b6d89685cfee0fc67abf2cb6bdc0867eb998026 rw,relatime - aufs none rw,si=caafa548068e525`

	gentooMountinfo = `15 1 8:6 / / rw,noatime,nodiratime - ext4 /dev/sda6 rw,data=ordered
16 15 0:3 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw
17 15 0:14 / /run rw,nosuid,nodev,relatime - tmpfs tmpfs rw,size=3292172k,mode=755
18 15 0:5 / /dev rw,nosuid,relatime - devtmpfs udev rw,size=10240k,nr_inodes=4106451,mode=755
19 18 0:12 / /dev/mqueue rw,nosuid,nodev,noexec,relatime - mqueue mqueue rw
20 18 0:10 / /dev/pts rw,nosuid,noexec,relatime - devpts devpts rw,gid=5,mode=620,ptmxmode=000
21 18 0:15 / /dev/shm rw,nosuid,nodev,noexec,relatime - tmpfs shm rw
22 15 0:16 / /sys rw,nosuid,nodev,noexec,relatime - sysfs sysfs rw
23 22 0:7 / /sys/kernel/debug rw,nosuid,nodev,noexec,relatime - debugfs debugfs rw
24 22 0:17 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime - tmpfs cgroup_root rw,size=10240k,mode=755
25 24 0:18 / /sys/fs/cgroup/openrc rw,nosuid,nodev,noexec,relatime - cgroup openrc rw,release_agent=/lib64/rc/sh/cgroup-release-agent.sh,name=openrc
26 24 0:19 / /sys/fs/cgroup/cpuset rw,nosuid,nodev,noexec,relatime - cgroup cpuset rw,cpuset,clone_children
27 24 0:20 / /sys/fs/cgroup/cpu rw,nosuid,nodev,noexec,relatime - cgroup cpu rw,cpu,clone_children
28 24 0:21 / /sys/fs/cgroup/cpuacct rw,nosuid,nodev,noexec,relatime - cgroup cpuacct rw,cpuacct,clone_children
29 24 0:22 / /sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime - cgroup memory rw,memory,clone_children
30 24 0:23 / /sys/fs/cgroup/devices rw,nosuid,nodev,noexec,relatime - cgroup devices rw,devices,clone_children
31 24 0:24 / /sys/fs/cgroup/freezer rw,nosuid,nodev,noexec,relatime - cgroup freezer rw,freezer,clone_children
32 24 0:25 / /sys/fs/cgroup/blkio rw,nosuid,nodev,noexec,relatime - cgroup blkio rw,blkio,clone_children
33 15 8:1 / /boot rw,noatime,nodiratime - vfat /dev/sda1 rw,fmask=0022,dmask=0022,codepage=437,iocharset=iso8859-1,shortname=mixed,errors=remount-ro
34 15 8:18 / /mnt/xfs rw,noatime,nodiratime - xfs /dev/sdb2 rw,attr2,inode64,noquota
35 15 0:26 / /tmp rw,relatime - tmpfs tmpfs rw
36 16 0:27 / /proc/sys/fs/binfmt_misc rw,nosuid,nodev,noexec,relatime - binfmt_misc binfmt_misc rw
42 15 0:33 / /var/lib/nfs/rpc_pipefs rw,relatime - rpc_pipefs rpc_pipefs rw
43 16 0:34 / /proc/fs/nfsd rw,nosuid,nodev,noexec,relatime - nfsd nfsd rw
44 15 0:35 / /home/tianon/.gvfs rw,nosuid,nodev,relatime - fuse.gvfs-fuse-daemon gvfs-fuse-daemon rw,user_id=1000,group_id=1000
68 15 0:3336 / /var/lib/docker/aufs/mnt/3597a1a6d6298c1decc339ebb90aad6f7d6ba2e15af3131b1f85e7ee4787a0cd rw,relatime - aufs none rw,si=9b4a7640128db39c
85 68 8:6 /var/lib/docker/init/dockerinit-0.7.2-dev//deleted /var/lib/docker/aufs/mnt/3597a1a6d6298c1decc339ebb90aad6f7d6ba2e15af3131b1f85e7ee4787a0cd/.dockerinit rw,noatime,nodiratime - ext4 /dev/sda6 rw,data=ordered
86 68 8:6 /var/lib/docker/containers/3597a1a6d6298c1decc339ebb90aad6f7d6ba2e15af3131b1f85e7ee4787a0cd/config.env /var/lib/docker/aufs/mnt/3597a1a6d6298c1decc339ebb90aad6f7d6ba2e15af3131b1f85e7ee4787a0cd/.dockerenv rw,noatime,nodiratime - ext4 /dev/sda6 rw,data=ordered
87 68 8:6 /etc/resolv.conf /var/lib/docker/aufs/mnt/3597a1a6d6298c1decc339ebb90aad6f7d6ba2e15af3131b1f85e7ee4787a0cd/etc/resolv.conf rw,noatime,nodiratime - ext4 /dev/sda6 rw,data=ordered
88 68 8:6 /var/lib/docker/containers/3597a1a6d6298c1decc339ebb90aad6f7d6ba2e15af3131b1f85e7ee4787a0cd/hostname /var/lib/docker/aufs/mnt/3597a1a6d6298c1decc339ebb90aad6f7d6ba2e15af3131b1f85e7ee4787a0cd/etc/hostname rw,noatime,nodiratime - ext4 /dev/sda6 rw,data=ordered
89 68 8:6 /var/lib/docker/containers/3597a1a6d6298c1decc339ebb90aad6f7d6ba2e15af3131b1f85e7ee4787a0cd/hosts /var/lib/docker/aufs/mnt/3597a1a6d6298c1decc339ebb90aad6f7d6ba2e15af3131b1f85e7ee4787a0cd/etc/hosts rw,noatime,nodiratime - ext4 /dev/sda6 rw,data=ordered
38 15 0:3384 / /var/lib/docker/aufs/mnt/0292005a9292401bb5197657f2b682d97d8edcb3b72b5e390d2a680139985b55 rw,relatime - aufs none rw,si=9b4a7642b584939c
39 15 0:3385 / /var/lib/docker/aufs/mnt/59db98c889de5f71b70cfb82c40cbe47b64332f0f56042a2987a9e5df6e5e3aa rw,relatime - aufs none rw,si=9b4a7642b584e39c
40 15 0:3386 / /var/lib/docker/aufs/mnt/0545f0f2b6548eb9601d08f35a08f5a0a385407d36027a28f58e06e9f61e0278 rw,relatime - aufs none rw,si=9b4a7642b584b39c
41 15 0:3387 / /var/lib/docker/aufs/mnt/d882cfa16d1aa8fe0331a36e79be3d80b151e49f24fc39a39c3fed1735d5feb5 rw,relatime - aufs none rw,si=9b4a76453040039c
45 15 0:3388 / /var/lib/docker/aufs/mnt/055ca3befcb1626e74f5344b3398724ff05c0de0e20021683d04305c9e70a3f6 rw,relatime - aufs none rw,si=9b4a76453040739c
46 15 0:3389 / /var/lib/docker/aufs/mnt/b899e4567a351745d4285e7f1c18fdece75d877deb3041981cd290be348b7aa6 rw,relatime - aufs none rw,si=9b4a7647def4039c
47 15 0:3390 / /var/lib/docker/aufs/mnt/067ca040292c58954c5129f953219accfae0d40faca26b4d05e76ca76a998f16 rw,relatime - aufs none rw,si=9b4a7647def4239c
48 15 0:3391 / /var/lib/docker/aufs/mnt/8c995e7cb6e5082742daeea720e340b021d288d25d92e0412c03d200df308a11 rw,relatime - aufs none rw,si=9b4a764479c1639c
49 15 0:3392 / /var/lib/docker/aufs/mnt/07cc54dfae5b45300efdacdd53cc72c01b9044956a86ce7bff42d087e426096d rw,relatime - aufs none rw,si=9b4a764479c1739c
50 15 0:3393 / /var/lib/docker/aufs/mnt/0a9c95cf4c589c05b06baa79150b0cc1d8e7102759fe3ce4afaabb8247ca4f85 rw,relatime - aufs none rw,si=9b4a7644059c839c
51 15 0:3394 / /var/lib/docker/aufs/mnt/468fa98cececcf4e226e8370f18f4f848d63faf287fb8321a07f73086441a3a0 rw,relatime - aufs none rw,si=9b4a7644059ca39c
52 15 0:3395 / /var/lib/docker/aufs/mnt/0b826192231c5ce066fffb5beff4397337b5fc19a377aa7c6282c7c0ce7f111f rw,relatime - aufs none rw,si=9b4a764479c1339c
53 15 0:3396 / /var/lib/docker/aufs/mnt/93b8ba1b772fbe79709b909c43ea4b2c30d712e53548f467db1ffdc7a384f196 rw,relatime - aufs none rw,si=9b4a7640798a739c
54 15 0:3397 / /var/lib/docker/aufs/mnt/0c0d0acfb506859b12ef18cdfef9ebed0b43a611482403564224bde9149d373c rw,relatime - aufs none rw,si=9b4a7640798a039c
55 15 0:3398 / /var/lib/docker/aufs/mnt/33648c39ab6c7c74af0243d6d6a81b052e9e25ad1e04b19892eb2dde013e358b rw,relatime - aufs none rw,si=9b4a7644b439b39c
56 15 0:3399 / /var/lib/docker/aufs/mnt/0c12bea97a1c958a3c739fb148536c1c89351d48e885ecda8f0499b5cc44407e rw,relatime - aufs none rw,si=9b4a7640798a239c
57 15 0:3400 / /var/lib/docker/aufs/mnt/ed443988ce125f172d7512e84a4de2627405990fd767a16adefa8ce700c19ce8 rw,relatime - aufs none rw,si=9b4a7644c8ed339c
59 15 0:3402 / /var/lib/docker/aufs/mnt/f61612c324ff3c924d3f7a82fb00a0f8d8f73c248c41897061949e9f5ab7e3b1 rw,relatime - aufs none rw,si=9b4a76442810c39c
60 15 0:3403 / /var/lib/docker/aufs/mnt/0f1ee55c6c4e25027b80de8e64b8b6fb542b3b41aa0caab9261da75752e22bfd rw,relatime - aufs none rw,si=9b4a76442810e39c
61 15 0:3404 / /var/lib/docker/aufs/mnt/956f6cc4af5785cb3ee6963dcbca668219437d9b28f513290b1453ac64a34f97 rw,relatime - aufs none rw,si=9b4a7644303ec39c
62 15 0:3405 / /var/lib/docker/aufs/mnt/1099769158c4b4773e2569e38024e8717e400f87a002c41d8cf47cb81b051ba6 rw,relatime - aufs none rw,si=9b4a7644303ee39c
63 15 0:3406 / /var/lib/docker/aufs/mnt/11890ceb98d4442595b676085cd7b21550ab85c5df841e0fba997ff54e3d522d rw,relatime - aufs none rw,si=9b4a7644303ed39c
64 15 0:3407 / /var/lib/docker/aufs/mnt/acdb90dc378e8ed2420b43a6d291f1c789a081cd1904018780cc038fcd7aae53 rw,relatime - aufs none rw,si=9b4a76434be2139c
65 15 0:3408 / /var/lib/docker/aufs/mnt/120e716f19d4714fbe63cc1ed246204f2c1106eefebc6537ba2587d7e7711959 rw,relatime - aufs none rw,si=9b4a76434be2339c
66 15 0:3409 / /var/lib/docker/aufs/mnt/b197b7fffb61d89e0ba1c40de9a9fc0d912e778b3c1bd828cf981ff37c1963bc rw,relatime - aufs none rw,si=9b4a76434be2039c
70 15 0:3412 / /var/lib/docker/aufs/mnt/1434b69d2e1bb18a9f0b96b9cdac30132b2688f5d1379f68a39a5e120c2f93eb rw,relatime - aufs none rw,si=9b4a76434be2639c
71 15 0:3413 / /var/lib/docker/aufs/mnt/16006e83caf33ab5eb0cd6afc92ea2ee8edeff897496b0bb3ec3a75b767374b3 rw,relatime - aufs none rw,si=9b4a7644d790439c
72 15 0:3414 / /var/lib/docker/aufs/mnt/55bfa5f44e94d27f91f79ba901b118b15098449165c87abf1b53ffff147ff164 rw,relatime - aufs none rw,si=9b4a7644d790239c
73 15 0:3415 / /var/lib/docker/aufs/mnt/1912b97a07ab21ccd98a2a27bc779bf3cf364a3138afa3c3e6f7f169a3c3eab5 rw,relatime - aufs none rw,si=9b4a76441822739c
76 15 0:3418 / /var/lib/docker/aufs/mnt/1a7c3292e8879bd91ffd9282e954f643b1db5683093574c248ff14a9609f2f56 rw,relatime - aufs none rw,si=9b4a76438cb7239c
77 15 0:3419 / /var/lib/docker/aufs/mnt/bb1faaf0d076ddba82c2318305a85f490dafa4e8a8640a8db8ed657c439120cc rw,relatime - aufs none rw,si=9b4a76438cb7339c
78 15 0:3420 / /var/lib/docker/aufs/mnt/1ab869f21d2241a73ac840c7f988490313f909ac642eba71d092204fec66dd7c rw,relatime - aufs none rw,si=9b4a76438cb7639c
79 15 0:3421 / /var/lib/docker/aufs/mnt/fd7245b2cfe3890fa5f5b452260e4edf9e7fb7746532ed9d83f7a0d7dbaa610e rw,relatime - aufs none rw,si=9b4a7644bdc0139c
80 15 0:3422 / /var/lib/docker/aufs/mnt/1e5686c5301f26b9b3cd24e322c608913465cc6c5d0dcd7c5e498d1314747d61 rw,relatime - aufs none rw,si=9b4a7644bdc0639c
81 15 0:3423 / /var/lib/docker/aufs/mnt/52edf6ee6e40bfec1e9301a4d4a92ab83d144e2ae4ce5099e99df6138cb844bf rw,relatime - aufs none rw,si=9b4a7644bdc0239c
82 15 0:3424 / /var/lib/docker/aufs/mnt/1ea10fb7085d28cda4904657dff0454e52598d28e1d77e4f2965bbc3666e808f rw,relatime - aufs none rw,si=9b4a76438cb7139c
83 15 0:3425 / /var/lib/docker/aufs/mnt/9c03e98c3593946dbd4087f8d83f9ca262f4a2efdc952ce60690838b9ba6c526 rw,relatime - aufs none rw,si=9b4a76443020639c
84 15 0:3426 / /var/lib/docker/aufs/mnt/220a2344d67437602c6d2cee9a98c46be13f82c2a8063919dd2fad52bf2fb7dd rw,relatime - aufs none rw,si=9b4a76434bff339c
94 15 0:3427 / /var/lib/docker/aufs/mnt/3b32876c5b200312c50baa476ff342248e88c8ea96e6a1032cd53a88738a1cf2 rw,relatime - aufs none rw,si=9b4a76434bff139c
95 15 0:3428 / /var/lib/docker/aufs/mnt/23ee2b8b0d4ae8db6f6d1e168e2c6f79f8a18f953b09f65e0d22cc1e67a3a6fa rw,relatime - aufs none rw,si=9b4a7646c305c39c
96 15 0:3429 / /var/lib/docker/aufs/mnt/e86e6daa70b61b57945fa178222615f3c3d6bcef12c9f28e9f8623d44dc2d429 rw,relatime - aufs none rw,si=9b4a7646c305f39c
97 15 0:3430 / /var/lib/docker/aufs/mnt/2413d07623e80860bb2e9e306fbdee699afd07525785c025c591231e864aa162 rw,relatime - aufs none rw,si=9b4a76434bff039c
98 15 0:3431 / /var/lib/docker/aufs/mnt/adfd622eb22340fc80b429e5564b125668e260bf9068096c46dd59f1386a4b7d rw,relatime - aufs none rw,si=9b4a7646a7a1039c
102 15 0:3435 / /var/lib/docker/aufs/mnt/27cd92e7a91d02e2d6b44d16679a00fb6d169b19b88822891084e7fd1a84882d rw,relatime - aufs none rw,si=9b4a7646f25ec39c
103 15 0:3436 / /var/lib/docker/aufs/mnt/27dfdaf94cfbf45055c748293c37dd68d9140240bff4c646cb09216015914a88 rw,relatime - aufs none rw,si=9b4a7646732f939c
104 15 0:3437 / /var/lib/docker/aufs/mnt/5ed7524aff68dfbf0fc601cbaeac01bab14391850a973dabf3653282a627920f rw,relatime - aufs none rw,si=9b4a7646732f839c
105 15 0:3438 / /var/lib/docker/aufs/mnt/2a0d4767e536beb5785b60e071e3ac8e5e812613ab143a9627bee77d0c9ab062 rw,relatime - aufs none rw,si=9b4a7646732fe39c
106 15 0:3439 / /var/lib/docker/aufs/mnt/dea3fc045d9f4ae51ba952450b948a822cf85c39411489ca5224f6d9a8d02bad rw,relatime - aufs none rw,si=9b4a764012ad839c
107 15 0:3440 / /var/lib/docker/aufs/mnt/2d140a787160798da60cb67c21b1210054ad4dafecdcf832f015995b9aa99cfd rw,relatime - aufs none rw,si=9b4a764012add39c
108 15 0:3441 / /var/lib/docker/aufs/mnt/cb190b2a8e984475914430fbad2382e0d20b9b659f8ef83ae8d170cc672e519c rw,relatime - aufs none rw,si=9b4a76454d9c239c
109 15 0:3442 / /var/lib/docker/aufs/mnt/2f4a012d5a7ffd90256a6e9aa479054b3dddbc3c6a343f26dafbf3196890223b rw,relatime - aufs none rw,si=9b4a76454d9c439c
110 15 0:3443 / /var/lib/docker/aufs/mnt/63cc77904b80c4ffbf49cb974c5d8733dc52ad7640d3ae87554b325d7312d87f rw,relatime - aufs none rw,si=9b4a76454d9c339c
111 15 0:3444 / /var/lib/docker/aufs/mnt/30333e872c451482ea2d235ff2192e875bd234006b238ae2bdde3b91a86d7522 rw,relatime - aufs none rw,si=9b4a76422cebf39c
112 15 0:3445 / /var/lib/docker/aufs/mnt/6c54fc1125da3925cae65b5c9a98f3be55b0a2c2666082e5094a4ba71beb5bff rw,relatime - aufs none rw,si=9b4a7646dd5a439c
113 15 0:3446 / /var/lib/docker/aufs/mnt/3087d48cb01cda9d0a83a9ca301e6ea40e8593d18c4921be4794c91a420ab9a3 rw,relatime - aufs none rw,si=9b4a7646dd5a739c
114 15 0:3447 / /var/lib/docker/aufs/mnt/cc2607462a8f55b179a749b144c3fdbb50678e1a4f3065ea04e283e9b1f1d8e2 rw,relatime - aufs none rw,si=9b4a7646dd5a239c
117 15 0:3450 / /var/lib/docker/aufs/mnt/310c5e8392b29e8658a22e08d96d63936633b7e2c38e8d220047928b00a03d24 rw,relatime - aufs none rw,si=9b4a7647932d739c
118 15 0:3451 / /var/lib/docker/aufs/mnt/38a1f0029406ba9c3b6058f2f406d8a1d23c855046cf355c91d87d446fcc1460 rw,relatime - aufs none rw,si=9b4a76445abc939c
119 15 0:3452 / /var/lib/docker/aufs/mnt/42e109ab7914ae997a11ccd860fd18e4d488c50c044c3240423ce15774b8b62e rw,relatime - aufs none rw,si=9b4a76445abca39c
120 15 0:3453 / /var/lib/docker/aufs/mnt/365d832af0402d052b389c1e9c0d353b48487533d20cd4351df8e24ec4e4f9d8 rw,relatime - aufs none rw,si=9b4a7644066aa39c
121 15 0:3454 / /var/lib/docker/aufs/mnt/d3fa8a24d695b6cda9b64f96188f701963d28bef0473343f8b212df1a2cf1d2b rw,relatime - aufs none rw,si=9b4a7644066af39c
122 15 0:3455 / /var/lib/docker/aufs/mnt/37d4f491919abc49a15d0c7a7cc8383f087573525d7d288accd14f0b4af9eae0 rw,relatime - aufs none rw,si=9b4a7644066ad39c
123 15 0:3456 / /var/lib/docker/aufs/mnt/93902707fe12cbdd0068ce73f2baad4b3a299189b1b19cb5f8a2025e106ae3f5 rw,relatime - aufs none rw,si=9b4a76444445f39c
126 15 0:3459 / /var/lib/docker/aufs/mnt/3b49291670a625b9bbb329ffba99bf7fa7abff80cefef040f8b89e2b3aad4f9f rw,relatime - aufs none rw,si=9b4a7640798a339c
127 15 0:3460 / /var/lib/docker/aufs/mnt/8d9c7b943cc8f854f4d0d4ec19f7c16c13b0cc4f67a41472a072648610cecb59 rw,relatime - aufs none rw,si=9b4a76427383039c
128 15 0:3461 / /var/lib/docker/aufs/mnt/3b6c90036526c376307df71d49c9f5fce334c01b926faa6a78186842de74beac rw,relatime - aufs none rw,si=9b4a7644badd439c
130 15 0:3463 / /var/lib/docker/aufs/mnt/7b24158eeddfb5d31b7e932e406ea4899fd728344335ff8e0765e89ddeb351dd rw,relatime - aufs none rw,si=9b4a7644badd539c
131 15 0:3464 / /var/lib/docker/aufs/mnt/3ead6dd5773765c74850cf6c769f21fe65c29d622ffa712664f9f5b80364ce27 rw,relatime - aufs none rw,si=9b4a7642f469939c
132 15 0:3465 / /var/lib/docker/aufs/mnt/3f825573b29547744a37b65597a9d6d15a8350be4429b7038d126a4c9a8e178f rw,relatime - aufs none rw,si=9b4a7642f469c39c
133 15 0:3466 / /var/lib/docker/aufs/mnt/f67aaaeb3681e5dcb99a41f847087370bd1c206680cb8c7b6a9819fd6c97a331 rw,relatime - aufs none rw,si=9b4a7647cc25939c
134 15 0:3467 / /var/lib/docker/aufs/mnt/41afe6cfb3c1fc2280b869db07699da88552786e28793f0bc048a265c01bd942 rw,relatime - aufs none rw,si=9b4a7647cc25c39c
135 15 0:3468 / /var/lib/docker/aufs/mnt/b8092ea59da34a40b120e8718c3ae9fa8436996edc4fc50e4b99c72dfd81e1af rw,relatime - aufs none rw,si=9b4a76445abc439c
136 15 0:3469 / /var/lib/docker/aufs/mnt/42c69d2cc179e2684458bb8596a9da6dad182c08eae9b74d5f0e615b399f75a5 rw,relatime - aufs none rw,si=9b4a76455ddbe39c
137 15 0:3470 / /var/lib/docker/aufs/mnt/ea0871954acd2d62a211ac60e05969622044d4c74597870c4f818fbb0c56b09b rw,relatime - aufs none rw,si=9b4a76455ddbf39c
138 15 0:3471 / /var/lib/docker/aufs/mnt/4307906b275ab3fc971786b3841ae3217ac85b6756ddeb7ad4ba09cd044c2597 rw,relatime - aufs none rw,si=9b4a76455ddb839c
139 15 0:3472 / /var/lib/docker/aufs/mnt/4390b872928c53500a5035634f3421622ed6299dc1472b631fc45de9f56dc180 rw,relatime - aufs none rw,si=9b4a76402f2fd39c
140 15 0:3473 / /var/lib/docker/aufs/mnt/6bb41e78863b85e4aa7da89455314855c8c3bda64e52a583bab15dc1fa2e80c2 rw,relatime - aufs none rw,si=9b4a76402f2fa39c
141 15 0:3474 / /var/lib/docker/aufs/mnt/4444f583c2a79c66608f4673a32c9c812154f027045fbd558c2d69920c53f835 rw,relatime - aufs none rw,si=9b4a764479dbd39c
142 15 0:3475 / /var/lib/docker/aufs/mnt/6f11883af4a05ea362e0c54df89058da4859f977efd07b6f539e1f55c1d2a668 rw,relatime - aufs none rw,si=9b4a76402f30b39c
143 15 0:3476 / /var/lib/docker/aufs/mnt/453490dd32e7c2e9ef906f995d8fb3c2753923d1a5e0ba3fd3296e2e4dc238e7 rw,relatime - aufs none rw,si=9b4a76402f30c39c
144 15 0:3477 / /var/lib/docker/aufs/mnt/45e5945735ee102b5e891c91650c57ec4b52bb53017d68f02d50ea8a6e230610 rw,relatime - aufs none rw,si=9b4a76423260739c
147 15 0:3480 / /var/lib/docker/aufs/mnt/4727a64a5553a1125f315b96bed10d3073d6988225a292cce732617c925b56ab rw,relatime - aufs none rw,si=9b4a76443030339c
150 15 0:3483 / /var/lib/docker/aufs/mnt/4e348b5187b9a567059306afc72d42e0ec5c893b0d4abd547526d5f9b6fb4590 rw,relatime - aufs none rw,si=9b4a7644f5d8c39c
151 15 0:3484 / /var/lib/docker/aufs/mnt/4efc616bfbc3f906718b052da22e4335f8e9f91ee9b15866ed3a8029645189ef rw,relatime - aufs none rw,si=9b4a7644f5d8939c
152 15 0:3485 / /var/lib/docker/aufs/mnt/83e730ae9754d5adb853b64735472d98dfa17136b8812ac9cfcd1eba7f4e7d2d rw,relatime - aufs none rw,si=9b4a76469aa7139c
153 15 0:3486 / /var/lib/docker/aufs/mnt/4fc5ba8a5b333be2b7eefacccb626772eeec0ae8a6975112b56c9fb36c0d342f rw,relatime - aufs none rw,si=9b4a7640128dc39c
154 15 0:3487 / /var/lib/docker/aufs/mnt/50200d5edff5dfe8d1ef3c78b0bbd709793ac6e936aa16d74ff66f7ea577b6f9 rw,relatime - aufs none rw,si=9b4a7640128da39c
155 15 0:3488 / /var/lib/docker/aufs/mnt/51e5e51604361448f0b9777f38329f414bc5ba9cf238f26d465ff479bd574b61 rw,relatime - aufs none rw,si=9b4a76444f68939c
156 15 0:3489 / /var/lib/docker/aufs/mnt/52a142149aa98bba83df8766bbb1c629a97b9799944ead90dd206c4bdf0b8385 rw,relatime - aufs none rw,si=9b4a76444f68b39c
157 15 0:3490 / /var/lib/docker/aufs/mnt/52dd21a94a00f58a1ed489312fcfffb91578089c76c5650364476f1d5de031bc rw,relatime - aufs none rw,si=9b4a76444f68f39c
158 15 0:3491 / /var/lib/docker/aufs/mnt/ee562415ddaad353ed22c88d0ca768a0c74bfba6333b6e25c46849ee22d990da rw,relatime - aufs none rw,si=9b4a7640128d839c
159 15 0:3492 / /var/lib/docker/aufs/mnt/db47a9e87173f7554f550c8a01891de79cf12acdd32e01f95c1a527a08bdfb2c rw,relatime - aufs none rw,si=9b4a764405a1d39c
160 15 0:3493 / /var/lib/docker/aufs/mnt/55e827bf6d44d930ec0b827c98356eb8b68c3301e2d60d1429aa72e05b4c17df rw,relatime - aufs none rw,si=9b4a764405a1a39c
162 15 0:3495 / /var/lib/docker/aufs/mnt/578dc4e0a87fc37ec081ca098430499a59639c09f6f12a8f48de29828a091aa6 rw,relatime - aufs none rw,si=9b4a76406d7d439c
163 15 0:3496 / /var/lib/docker/aufs/mnt/728cc1cb04fa4bc6f7bf7a90980beda6d8fc0beb71630874c0747b994efb0798 rw,relatime - aufs none rw,si=9b4a76444f20e39c
164 15 0:3497 / /var/lib/docker/aufs/mnt/5850cc4bd9b55aea46c7ad598f1785117607974084ea643580f58ce3222e683a rw,relatime - aufs none rw,si=9b4a7644a824239c
165 15 0:3498 / /var/lib/docker/aufs/mnt/89443b3f766d5a37bc8b84e29da8b84e6a3ea8486d3cf154e2aae1816516e4a8 rw,relatime - aufs none rw,si=9b4a7644a824139c
166 15 0:3499 / /var/lib/docker/aufs/mnt/f5ae8fd5a41a337907d16515bc3162525154b59c32314c695ecd092c3b47943d rw,relatime - aufs none rw,si=9b4a7644a824439c
167 15 0:3500 / /var/lib/docker/aufs/mnt/5a430854f2a03a9e5f7cbc9f3fb46a8ebca526a5b3f435236d8295e5998798f5 rw,relatime - aufs none rw,si=9b4a7647fc82439c
168 15 0:3501 / /var/lib/docker/aufs/mnt/eda16901ae4cead35070c39845cbf1e10bd6b8cb0ffa7879ae2d8a186e460f91 rw,relatime - aufs none rw,si=9b4a76441e0df39c
169 15 0:3502 / /var/lib/docker/aufs/mnt/5a593721430c2a51b119ff86a7e06ea2b37e3b4131f8f1344d402b61b0c8d868 rw,relatime - aufs none rw,si=9b4a764248bad39c
170 15 0:3503 / /var/lib/docker/aufs/mnt/d662ad0a30fbfa902e0962108685b9330597e1ee2abb16dc9462eb5a67fdd23f rw,relatime - aufs none rw,si=9b4a764248bae39c
171 15 0:3504 / /var/lib/docker/aufs/mnt/5bc9de5c79812843fb36eee96bef1ddba812407861f572e33242f4ee10da2c15 rw,relatime - aufs none rw,si=9b4a764248ba839c
172 15 0:3505 / /var/lib/docker/aufs/mnt/5e763de8e9b0f7d58d2e12a341e029ab4efb3b99788b175090d8209e971156c1 rw,relatime - aufs none rw,si=9b4a764248baa39c
173 15 0:3506 / /var/lib/docker/aufs/mnt/b4431dc2739936f1df6387e337f5a0c99cf051900c896bd7fd46a870ce61c873 rw,relatime - aufs none rw,si=9b4a76401263539c
174 15 0:3507 / /var/lib/docker/aufs/mnt/5f37830e5a02561ab8c67ea3113137ba69f67a60e41c05cb0e7a0edaa1925b24 rw,relatime - aufs none rw,si=9b4a76401263639c
184 15 0:3508 / /var/lib/docker/aufs/mnt/62ea10b957e6533538a4633a1e1d678502f50ddcdd354b2ca275c54dd7a7793a rw,relatime - aufs none rw,si=9b4a76401263039c
187 15 0:3509 / /var/lib/docker/aufs/mnt/d56ee9d44195fe390e042fda75ec15af5132adb6d5c69468fa8792f4e54a6953 rw,relatime - aufs none rw,si=9b4a76401263239c
188 15 0:3510 / /var/lib/docker/aufs/mnt/6a300930673174549c2b62f36c933f0332a20735978c007c805a301f897146c5 rw,relatime - aufs none rw,si=9b4a76455d4c539c
189 15 0:3511 / /var/lib/docker/aufs/mnt/64496c45c84d348c24d410015456d101601c30cab4d1998c395591caf7e57a70 rw,relatime - aufs none rw,si=9b4a76455d4c639c
190 15 0:3512 / /var/lib/docker/aufs/mnt/65a6a645883fe97a7422cd5e71ebe0bc17c8e6302a5361edf52e89747387e908 rw,relatime - aufs none rw,si=9b4a76455d4c039c
191 15 0:3513 / /var/lib/docker/aufs/mnt/672be40695f7b6e13b0a3ed9fc996c73727dede3481f58155950fcfad57ed616 rw,relatime - aufs none rw,si=9b4a76455d4c239c
192 15 0:3514 / /var/lib/docker/aufs/mnt/d42438acb2bfb2169e1c0d8e917fc824f7c85d336dadb0b0af36dfe0f001b3ba rw,relatime - aufs none rw,si=9b4a7642bfded39c
193 15 0:3515 / /var/lib/docker/aufs/mnt/b48a54abf26d01cb2ddd908b1ed6034d17397c1341bf0eb2b251a3e5b79be854 rw,relatime - aufs none rw,si=9b4a7642bfdee39c
194 15 0:3516 / /var/lib/docker/aufs/mnt/76f27134491f052bfb87f59092126e53ef875d6851990e59195a9da16a9412f8 rw,relatime - aufs none rw,si=9b4a7642bfde839c
195 15 0:3517 / /var/lib/docker/aufs/mnt/6bd626a5462b4f8a8e1cc7d10351326dca97a59b2758e5ea549a4f6350ce8a90 rw,relatime - aufs none rw,si=9b4a7642bfdea39c
196 15 0:3518 / /var/lib/docker/aufs/mnt/f1fe3549dbd6f5ca615e9139d9b53f0c83a3b825565df37628eacc13e70cbd6d rw,relatime - aufs none rw,si=9b4a7642bfdf539c
197 15 0:3519 / /var/lib/docker/aufs/mnt/6d0458c8426a9e93d58d0625737e6122e725c9408488ed9e3e649a9984e15c34 rw,relatime - aufs none rw,si=9b4a7642bfdf639c
198 15 0:3520 / /var/lib/docker/aufs/mnt/6e4c97db83aa82145c9cf2bafc20d500c0b5389643b689e3ae84188c270a48c5 rw,relatime - aufs none rw,si=9b4a7642bfdf039c
199 15 0:3521 / /var/lib/docker/aufs/mnt/eb94d6498f2c5969eaa9fa11ac2934f1ab90ef88e2d002258dca08e5ba74ea27 rw,relatime - aufs none rw,si=9b4a7642bfdf239c
200 15 0:3522 / /var/lib/docker/aufs/mnt/fe3f88f0c511608a2eec5f13a98703aa16e55dbf930309723d8a37101f539fe1 rw,relatime - aufs none rw,si=9b4a7642bfc3539c
201 15 0:3523 / /var/lib/docker/aufs/mnt/6f40c229fb9cad85fabf4b64a2640a5403ec03fe5ac1a57d0609fb8b606b9c83 rw,relatime - aufs none rw,si=9b4a7642bfc3639c
202 15 0:3524 / /var/lib/docker/aufs/mnt/7513e9131f7a8acf58ff15248237feb767c78732ca46e159f4d791e6ef031dbc rw,relatime - aufs none rw,si=9b4a7642bfc3039c
203 15 0:3525 / /var/lib/docker/aufs/mnt/79f48b00aa713cdf809c6bb7c7cb911b66e9a8076c81d6c9d2504139984ea2da rw,relatime - aufs none rw,si=9b4a7642bfc3239c
204 15 0:3526 / /var/lib/docker/aufs/mnt/c3680418350d11358f0a96c676bc5aa74fa00a7c89e629ef5909d3557b060300 rw,relatime - aufs none rw,si=9b4a7642f47cd39c
205 15 0:3527 / /var/lib/docker/aufs/mnt/7a1744dd350d7fcc0cccb6f1757ca4cbe5453f203a5888b0f1014d96ad5a5ef9 rw,relatime - aufs none rw,si=9b4a7642f47ce39c
206 15 0:3528 / /var/lib/docker/aufs/mnt/7fa99662db046be9f03c33c35251afda9ccdc0085636bbba1d90592cec3ff68d rw,relatime - aufs none rw,si=9b4a7642f47c839c
207 15 0:3529 / /var/lib/docker/aufs/mnt/f815021ef20da9c9b056bd1d52d8aaf6e2c0c19f11122fc793eb2b04eb995e35 rw,relatime - aufs none rw,si=9b4a7642f47ca39c
208 15 0:3530 / /var/lib/docker/aufs/mnt/801086ae3110192d601dfcebdba2db92e86ce6b6a9dba6678ea04488e4513669 rw,relatime - aufs none rw,si=9b4a7642dc6dd39c
209 15 0:3531 / /var/lib/docker/aufs/mnt/822ba7db69f21daddda87c01cfbfbf73013fc03a879daf96d16cdde6f9b1fbd6 rw,relatime - aufs none rw,si=9b4a7642dc6de39c
210 15 0:3532 / /var/lib/docker/aufs/mnt/834227c1a950fef8cae3827489129d0dd220541e60c6b731caaa765bf2e6a199 rw,relatime - aufs none rw,si=9b4a7642dc6d839c
211 15 0:3533 / /var/lib/docker/aufs/mnt/83dccbc385299bd1c7cf19326e791b33a544eea7b4cdfb6db70ea94eed4389fb rw,relatime - aufs none rw,si=9b4a7642dc6da39c
212 15 0:3534 / /var/lib/docker/aufs/mnt/f1b8e6f0e7c8928b5dcdab944db89306ebcae3e0b32f9ff40d2daa8329f21600 rw,relatime - aufs none rw,si=9b4a7645a126039c
213 15 0:3535 / /var/lib/docker/aufs/mnt/970efb262c7a020c2404cbcc5b3259efba0d110a786079faeef05bc2952abf3a rw,relatime - aufs none rw,si=9b4a7644c8ed139c
214 15 0:3536 / /var/lib/docker/aufs/mnt/84b6d73af7450f3117a77e15a5ca1255871fea6182cd8e8a7be6bc744be18c2c rw,relatime - aufs none rw,si=9b4a76406559139c
215 15 0:3537 / /var/lib/docker/aufs/mnt/88be2716e026bc681b5e63fe7942068773efbd0b6e901ca7ba441412006a96b6 rw,relatime - aufs none rw,si=9b4a76406559339c
216 15 0:3538 / /var/lib/docker/aufs/mnt/c81939aa166ce50cd8bca5cfbbcc420a78e0318dd5cd7c755209b9166a00a752 rw,relatime - aufs none rw,si=9b4a76406559239c
217 15 0:3539 / /var/lib/docker/aufs/mnt/e0f241645d64b7dc5ff6a8414087cca226be08fb54ce987d1d1f6350c57083aa rw,relatime - aufs none rw,si=9b4a7647cfc0f39c
218 15 0:3540 / /var/lib/docker/aufs/mnt/e10e2bf75234ed51d8a6a4bb39e465404fecbe318e54400d3879cdb2b0679c78 rw,relatime - aufs none rw,si=9b4a7647cfc0939c
219 15 0:3541 / /var/lib/docker/aufs/mnt/8f71d74c8cfc3228b82564aa9f09b2e576cff0083ddfb6aa5cb350346063f080 rw,relatime - aufs none rw,si=9b4a7647cfc0a39c
220 15 0:3542 / /var/lib/docker/aufs/mnt/9159f1eba2aef7f5205cc18d015cda7f5933cd29bba3b1b8aed5ccb5824c69ee rw,relatime - aufs none rw,si=9b4a76468cedd39c
221 15 0:3543 / /var/lib/docker/aufs/mnt/932cad71e652e048e500d9fbb5b8ea4fc9a269d42a3134ce527ceef42a2be56b rw,relatime - aufs none rw,si=9b4a76468cede39c
222 15 0:3544 / /var/lib/docker/aufs/mnt/bf1e1b5f529e8943cc0144ee86dbaaa37885c1ddffcef29537e0078ee7dd316a rw,relatime - aufs none rw,si=9b4a76468ced839c
223 15 0:3545 / /var/lib/docker/aufs/mnt/949d93ecf3322e09f858ce81d5f4b434068ec44ff84c375de03104f7b45ee955 rw,relatime - aufs none rw,si=9b4a76468ceda39c
224 15 0:3546 / /var/lib/docker/aufs/mnt/d65c6087f92dc2a3841b5251d2fe9ca07d4c6e5b021597692479740816e4e2a1 rw,relatime - aufs none rw,si=9b4a7645a126239c
225 15 0:3547 / /var/lib/docker/aufs/mnt/98a0153119d0651c193d053d254f6e16a68345a141baa80c87ae487e9d33f290 rw,relatime - aufs none rw,si=9b4a7640787cf39c
226 15 0:3548 / /var/lib/docker/aufs/mnt/99daf7fe5847c017392f6e59aa9706b3dfdd9e6d1ba11dae0f7fffde0a60b5e5 rw,relatime - aufs none rw,si=9b4a7640787c839c
227 15 0:3549 / /var/lib/docker/aufs/mnt/9ad1f2fe8a5599d4e10c5a6effa7f03d932d4e92ee13149031a372087a359079 rw,relatime - aufs none rw,si=9b4a7640787ca39c
228 15 0:3550 / /var/lib/docker/aufs/mnt/c26d64494da782ddac26f8370d86ac93e7c1666d88a7b99110fc86b35ea6a85d rw,relatime - aufs none rw,si=9b4a7642fc6b539c
229 15 0:3551 / /var/lib/docker/aufs/mnt/a49e4a8275133c230ec640997f35f172312eb0ea5bd2bbe10abf34aae98f30eb rw,relatime - aufs none rw,si=9b4a7642fc6b639c
230 15 0:3552 / /var/lib/docker/aufs/mnt/b5e2740c867ed843025f49d84e8d769de9e8e6039b3c8cb0735b5bf358994bc7 rw,relatime - aufs none rw,si=9b4a7642fc6b039c
231 15 0:3553 / /var/lib/docker/aufs/mnt/a826fdcf3a7039b30570054579b65763db605a314275d7aef31b872c13311b4b rw,relatime - aufs none rw,si=9b4a7642fc6b239c
232 15 0:3554 / /var/lib/docker/aufs/mnt/addf3025babf5e43b5a3f4a0da7ad863dda3c01fb8365c58fd8d28bb61dc11bc rw,relatime - aufs none rw,si=9b4a76407871d39c
233 15 0:3555 / /var/lib/docker/aufs/mnt/c5b6c6813ab3e5ebdc6d22cb2a3d3106a62095f2c298be52b07a3b0fa20ff690 rw,relatime - aufs none rw,si=9b4a76407871e39c
234 15 0:3556 / /var/lib/docker/aufs/mnt/af0609eaaf64e2392060cb46f5a9f3d681a219bb4c651d4f015bf573fbe6c4cf rw,relatime - aufs none rw,si=9b4a76407871839c
235 15 0:3557 / /var/lib/docker/aufs/mnt/e7f20e3c37ecad39cd90a97cd3549466d0d106ce4f0a930b8495442634fa4a1f rw,relatime - aufs none rw,si=9b4a76407871a39c
237 15 0:3559 / /var/lib/docker/aufs/mnt/b57a53d440ffd0c1295804fa68cdde35d2fed5409484627e71b9c37e4249fd5c rw,relatime - aufs none rw,si=9b4a76444445a39c
238 15 0:3560 / /var/lib/docker/aufs/mnt/b5e7d7b8f35e47efbba3d80c5d722f5e7bd43e54c824e54b4a4b351714d36d42 rw,relatime - aufs none rw,si=9b4a7647932d439c
239 15 0:3561 / /var/lib/docker/aufs/mnt/f1b136def157e9465640658f277f3347de593c6ae76412a2e79f7002f091cae2 rw,relatime - aufs none rw,si=9b4a76445abcd39c
240 15 0:3562 / /var/lib/docker/aufs/mnt/b750fe79269d2ec9a3c593ef05b4332b1d1a02a62b4accb2c21d589ff2f5f2dc rw,relatime - aufs none rw,si=9b4a7644403b339c
241 15 0:3563 / /var/lib/docker/aufs/mnt/b89b140cdbc95063761864e0a23346207fa27ee4c5c63a1ae85c9069a9d9cf1d rw,relatime - aufs none rw,si=9b4a7644aa19739c
242 15 0:3564 / /var/lib/docker/aufs/mnt/bc6a69ed51c07f5228f6b4f161c892e6a949c0e7e86a9c3432049d4c0e5cd298 rw,relatime - aufs none rw,si=9b4a7644aa19139c
243 15 0:3565 / /var/lib/docker/aufs/mnt/be4e2ba3f136933e239f7cf3d136f484fb9004f1fbdfee24a62a2c7b0ab30670 rw,relatime - aufs none rw,si=9b4a7644aa19339c
244 15 0:3566 / /var/lib/docker/aufs/mnt/e04ca1a4a5171e30d20f0c92f90a50b8b6f8600af5459c4b4fb25e42e864dfe1 rw,relatime - aufs none rw,si=9b4a7647932d139c
245 15 0:3567 / /var/lib/docker/aufs/mnt/be61576b31db893129aaffcd3dcb5ce35e49c4b71b30c392a78609a45c7323d8 rw,relatime - aufs none rw,si=9b4a7642d85f739c
246 15 0:3568 / /var/lib/docker/aufs/mnt/dda42c191e56becf672327658ab84fcb563322db3764b91c2fefe4aaef04c624 rw,relatime - aufs none rw,si=9b4a7642d85f139c
247 15 0:3569 / /var/lib/docker/aufs/mnt/c0a7995053330f3d88969247a2e72b07e2dd692133f5668a4a35ea3905561072 rw,relatime - aufs none rw,si=9b4a7642d85f339c
249 15 0:3571 / /var/lib/docker/aufs/mnt/c3594b2e5f08c59ff5ed338a1ba1eceeeb1f7fc5d180068338110c00b1eb8502 rw,relatime - aufs none rw,si=9b4a7642738c739c
250 15 0:3572 / /var/lib/docker/aufs/mnt/c58dce03a0ab0a7588393880379dc3bce9f96ec08ed3f99cf1555260ff0031e8 rw,relatime - aufs none rw,si=9b4a7642738c139c
251 15 0:3573 / /var/lib/docker/aufs/mnt/c73e9f1d109c9d14cb36e1c7489df85649be3911116d76c2fd3648ec8fd94e23 rw,relatime - aufs none rw,si=9b4a7642738c339c
252 15 0:3574 / /var/lib/docker/aufs/mnt/c9eef28c344877cd68aa09e543c0710ab2b305a0ff96dbb859bfa7808c3e8d01 rw,relatime - aufs none rw,si=9b4a7642d85f439c
253 15 0:3575 / /var/lib/docker/aufs/mnt/feb67148f548d70cb7484f2aaad2a86051cd6867a561741a2f13b552457d666e rw,relatime - aufs none rw,si=9b4a76468c55739c
254 15 0:3576 / /var/lib/docker/aufs/mnt/cdf1f96c36d35a96041a896bf398ec0f7dc3b0fb0643612a0f4b6ff96e04e1bb rw,relatime - aufs none rw,si=9b4a76468c55139c
255 15 0:3577 / /var/lib/docker/aufs/mnt/ec6e505872353268451ac4bc034c1df00f3bae4a3ea2261c6e48f7bd5417c1b3 rw,relatime - aufs none rw,si=9b4a76468c55339c
256 15 0:3578 / /var/lib/docker/aufs/mnt/d6dc8aca64efd90e0bc10274001882d0efb310d42ccbf5712b99b169053b8b1a rw,relatime - aufs none rw,si=9b4a7642738c439c
257 15 0:3579 / /var/lib/docker/aufs/mnt/d712594e2ff6eaeb895bfd150d694bd1305fb927e7a186b2dab7df2ea95f8f81 rw,relatime - aufs none rw,si=9b4a76401268f39c
259 15 0:3581 / /var/lib/docker/aufs/mnt/dbfa1174cd78cde2d7410eae442af0b416c4a0e6f87ed4ff1e9f169a0029abc0 rw,relatime - aufs none rw,si=9b4a76401268b39c
260 15 0:3582 / /var/lib/docker/aufs/mnt/e883f5a82316d7856fbe93ee8c0af5a920b7079619dd95c4ffd88bbd309d28dd rw,relatime - aufs none rw,si=9b4a76468c55439c
261 15 0:3583 / /var/lib/docker/aufs/mnt/fdec3eff581c4fc2b09f87befa2fa021f3f2d373bea636a87f1fb5b367d6347a rw,relatime - aufs none rw,si=9b4a7644aa1af39c
262 15 0:3584 / /var/lib/docker/aufs/mnt/ef764e26712184653067ecf7afea18a80854c41331ca0f0ef03e1bacf90a6ffc rw,relatime - aufs none rw,si=9b4a7644aa1a939c
263 15 0:3585 / /var/lib/docker/aufs/mnt/f3176b40c41fce8ce6942936359a2001a6f1b5c1bb40ee224186db0789ec2f76 rw,relatime - aufs none rw,si=9b4a7644aa1ab39c
264 15 0:3586 / /var/lib/docker/aufs/mnt/f5daf06785d3565c6dd18ea7d953d9a8b9606107781e63270fe0514508736e6a rw,relatime - aufs none rw,si=9b4a76401268c39c
58 15 0:3587 / /var/lib/docker/aufs/mnt/cde8c40f6524b7361af4f5ad05bb857dc9ee247c20852ba666195c0739e3a2b8-init rw,relatime - aufs none rw,si=9b4a76444445839c
67 15 0:3588 / /var/lib/docker/aufs/mnt/cde8c40f6524b7361af4f5ad05bb857dc9ee247c20852ba666195c0739e3a2b8 rw,relatime - aufs none rw,si=9b4a7644badd339c
265 15 0:3610 / /var/lib/docker/aufs/mnt/e812472cd2c8c4748d1ef71fac4e77e50d661b9349abe66ce3e23511ed44f414 rw,relatime - aufs none rw,si=9b4a76427937d39c
270 15 0:3615 / /var/lib/docker/aufs/mnt/997636e7c5c9d0d1376a217e295c14c205350b62bc12052804fb5f90abe6f183 rw,relatime - aufs none rw,si=9b4a76406540739c
273 15 0:3618 / /var/lib/docker/aufs/mnt/d5794d080417b6e52e69227c3873e0e4c1ff0d5a845ebe3860ec2f89a47a2a1e rw,relatime - aufs none rw,si=9b4a76454814039c
278 15 0:3623 / /var/lib/docker/aufs/mnt/586bdd48baced671bb19bc4d294ec325f26c55545ae267db426424f157d59c48 rw,relatime - aufs none rw,si=9b4a7644b439f39c
281 15 0:3626 / /var/lib/docker/aufs/mnt/69739d022f89f8586908bbd5edbbdd95ea5256356f177f9ffcc6ef9c0ea752d2 rw,relatime - aufs none rw,si=9b4a7644a0f1b39c
286 15 0:3631 / /var/lib/docker/aufs/mnt/ff28c27d5f894363993622de26d5dd352dba072f219e4691d6498c19bbbc15a9 rw,relatime - aufs none rw,si=9b4a7642265b339c
289 15 0:3634 / /var/lib/docker/aufs/mnt/aa128fe0e64fdede333aa48fd9de39530c91a9244a0f0649a3c411c61e372daa rw,relatime - aufs none rw,si=9b4a764012ada39c
99 15 8:33 / /media/REMOVE\040ME rw,nosuid,nodev,relatime - fuseblk /dev/sdc1 rw,user_id=0,group_id=0,allow_other,blksize=4096`
)

func TestParseFedoraMountinfo(t *testing.T) {
	r := bytes.NewBuffer([]byte(fedoraMountinfo))
	_, err := parseInfoFile(r)
	if err != nil {
		t.Fatal(err)
	}
}

func TestParseUbuntuMountinfo(t *testing.T) {
	r := bytes.NewBuffer([]byte(ubuntuMountInfo))
	_, err := parseInfoFile(r)
	if err != nil {
		t.Fatal(err)
	}
}

func TestParseGentooMountinfo(t *testing.T) {
	r := bytes.NewBuffer([]byte(gentooMountinfo))
	_, err := parseInfoFile(r)
	if err != nil {
		t.Fatal(err)
	}
}

func TestParseFedoraMountinfoFields(t *testing.T) {
	r := bytes.NewBuffer([]byte(fedoraMountinfo))
	infos, err := parseInfoFile(r)
	if err != nil {
		t.Fatal(err)
	}
	expectedLength := 58
	if len(infos) != expectedLength {
		t.Fatalf("Expected %d entries, got %d", expectedLength, len(infos))
	}
	mi := MountInfo{
		Id:         15,
		Parent:     35,
		Major:      0,
		Minor:      3,
		Root:       "/",
		Mountpoint: "/proc",
		Opts:       "rw,nosuid,nodev,noexec,relatime",
		Optional:   "shared:5",
		Fstype:     "proc",
		Source:     "proc",
		VfsOpts:    "rw",
	}

	if *infos[0] != mi {
		t.Fatalf("expected %#v, got %#v", mi, infos[0])
	}
}
