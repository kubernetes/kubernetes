# NAME
   runc update - update container resource constraints

# SYNOPSIS
   runc update [command options] <container-id>

# DESCRIPTION
   The data can be read from a file or the standard input, the
accepted format is as follow (unchanged values can be omitted):

   {
     "memory": {
       "limit": 0,
       "reservation": 0,
       "swap": 0,
       "kernel": 0,
       "kernelTCP": 0
     },
     "cpu": {
       "shares": 0,
       "quota": 0,
       "period": 0,
       "realtimeRuntime": 0,
       "realtimePeriod": 0,
       "cpus": "",
       "mems": ""
     },
     "blockIO": {
       "blkioWeight": 0
     }
   }

Note: if data is to be read from a file or the standard input, all
other options are ignored.

# OPTIONS
   --resources value, -r value  path to the file containing the resources to update or '-' to read from the standard input
   --blkio-weight value         Specifies per cgroup weight, range is from 10 to 1000 (default: 0)
   --cpu-period value           CPU CFS period to be used for hardcapping (in usecs). 0 to use system default
   --cpu-quota value            CPU CFS hardcap limit (in usecs). Allowed cpu time in a given period
   --cpu-rt-period value        CPU realtime period to be used for hardcapping (in usecs). 0 to use system default
   --cpu-rt-runtime value       CPU realtime hardcap limit (in usecs). Allowed cpu time in a given period
   --cpu-share value            CPU shares (relative weight vs. other containers)
   --cpuset-cpus value          CPU(s) to use
   --cpuset-mems value          Memory node(s) to use
   --kernel-memory value        Kernel memory limit (in bytes)
   --kernel-memory-tcp value    Kernel memory limit (in bytes) for tcp buffer
   --memory value               Memory limit (in bytes)
   --memory-reservation value   Memory reservation or soft_limit (in bytes)
   --memory-swap value          Total memory usage (memory + swap); set '-1' to enable unlimited swap
   --pids-limit value           Maximum number of pids allowed in the container (default: 0)
