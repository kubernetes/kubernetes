# The UDP Input

## A note on UDP/IP OS Buffer sizes

Some OSes (most notably, Linux) place very restricive limits on the performance
of UDP protocols. It is _highly_ recommended that you increase these OS limits to
at least 25MB before trying to run UDP traffic to your instance.
25MB is just a recommendation, and should be adjusted to be inline with your
`read-buffer` plugin setting.

### Linux
Check the current UDP/IP receive buffer default and limit by typing the following commands:

```
sysctl net.core.rmem_max
sysctl net.core.rmem_default
```

If the values are less than 26214400 bytes (25MB) you should add the following lines to the /etc/sysctl.conf file:

```
net.core.rmem_max=26214400
net.core.rmem_default=26214400
```

Changes to /etc/sysctl.conf do not take effect until reboot.  To update the values immediately, type the following commands as root:

```
sysctl -w net.core.rmem_max=26214400
sysctl -w net.core.rmem_default=26214400
```

### BSD/Darwin

On BSD/Darwin systems you need to add about a 15% padding to the kernel limit
socket buffer. Meaning if you want an 25MB buffer (8388608 bytes) you need to set
the kernel limit to `26214400*1.15 = 30146560`. This is not documented anywhere but
happens
[in the kernel here.](https://github.com/freebsd/freebsd/blob/master/sys/kern/uipc_sockbuf.c#L63-L64)

Check the current UDP/IP buffer limit by typing the following command:

```
sysctl kern.ipc.maxsockbuf
```

If the value is less than 30146560 bytes you should add the following lines to the /etc/sysctl.conf file (create it if necessary):

```
kern.ipc.maxsockbuf=30146560
```

Changes to /etc/sysctl.conf do not take effect until reboot.  To update the values immediately, type the following commands as root:

```
sysctl -w kern.ipc.maxsockbuf=30146560
```

### Using the read-buffer option for the UDP listener

The `read-buffer` option allows users to set the buffer size for the UDP listener.
It Sets the size of the operating system's receive buffer associated with
the UDP traffic. Keep in mind that the OS must be able
to handle the number set here or the UDP listener will error and exit.

`read-buffer = 0` means to use the OS default, which is usually too
small for high UDP performance.

## Configuration

Each UDP input allows the binding address, target database, and target retention policy to be set. If the database does not exist, it will be created automatically when the input is initialized. If the retention policy is not configured, then the default retention policy for the database is used. However if the retention policy is set, the retention policy must be explicitly created. The input will not automatically create it.

Each UDP input also performs internal batching of the points it receives, as batched writes to the database are more efficient. The default _batch size_ is 1000, _pending batch_ factor is 5, with a _batch timeout_ of 1 second. This means the input will write batches of maximum size 1000, but if a batch has not reached 1000 points within 1 second of the first point being added to a batch, it will emit that batch regardless of size. The pending batch factor controls how many batches can be in memory at once, allowing the input to transmit a batch, while still building other batches.

## Processing

The UDP input can receive up to 64KB per read, and splits the received data by newline. Each part is then interpreted as line-protocol encoded points, and parsed accordingly.

## UDP is connectionless

Since UDP is a connectionless protocol there is no way to signal to the data source if any error occurs, and if data has even been successfully indexed. This should be kept in mind when deciding if and when to use the UDP input. The built-in UDP statistics are useful for monitoring the UDP inputs.

## Config Examples

One UDP listener

```
# influxd.conf
...
[[udp]]
  enabled = true
  bind-address = ":8089" # the bind address
  database = "telegraf" # Name of the database that will be written to
  batch-size = 5000 # will flush if this many points get buffered
  batch-timeout = "1s" # will flush at least this often even if the batch-size is not reached
  batch-pending = 10 # number of batches that may be pending in memory
  read-buffer = 0 # UDP read buffer, 0 means to use OS default
...
```

Multiple UDP listeners

```
# influxd.conf
...
[[udp]]
  # Default UDP for Telegraf
  enabled = true
  bind-address = ":8089" # the bind address
  database = "telegraf" # Name of the database that will be written to
  batch-size = 5000 # will flush if this many points get buffered
  batch-timeout = "1s" # will flush at least this often even if the batch-size is not reached
  batch-pending = 10 # number of batches that may be pending in memory
  read-buffer = 0 # UDP read buffer size, 0 means to use OS default

[[udp]]
  # High-traffic UDP
  enabled = true
  bind-address = ":80891" # the bind address
  database = "mymetrics" # Name of the database that will be written to
  batch-size = 5000 # will flush if this many points get buffered
  batch-timeout = "1s" # will flush at least this often even if the batch-size is not reached
  batch-pending = 100 # number of batches that may be pending in memory
  read-buffer = 8388608 # (8*1024*1024) UDP read buffer size
...
```


