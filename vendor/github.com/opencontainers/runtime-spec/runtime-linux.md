# <a name="linuxRuntime" />Linux Runtime

## <a name="runtimeLinuxFileDescriptors" />File descriptors

By default, only the `stdin`, `stdout` and `stderr` file descriptors are kept open for the application by the runtime.
The runtime MAY pass additional file descriptors to the application to support features such as [socket activation][socket-activated-containers].
Some of the file descriptors MAY be redirected to `/dev/null` even though they are open.

## <a name="runtimeLinuxDevSymbolicLinks" /> Dev symbolic links

While creating the container (step 2 in the [lifecycle](runtime.md#lifecycle)), runtimes MUST create the following symlinks if the source file exists after processing [`mounts`](config.md#mounts):

|    Source       | Destination |
| --------------- | ----------- |
| /proc/self/fd   | /dev/fd     |
| /proc/self/fd/0 | /dev/stdin  |
| /proc/self/fd/1 | /dev/stdout |
| /proc/self/fd/2 | /dev/stderr |


[socket-activated-containers]: http://0pointer.de/blog/projects/socket-activated-containers.html
