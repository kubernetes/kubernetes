Docker device tool for devicemapper storage driver backend
===================

The ./contrib/docker-device-tool contains a tool to manipulate devicemapper thin-pool.

Compile
========

    $ make shell
    ## inside build container
    $ go build contrib/docker-device-tool/device_tool.go

    # if devicemapper version is old and compilation fails, compile with `libdm_no_deferred_remove` tag
    $ go build -tags libdm_no_deferred_remove contrib/docker-device-tool/device_tool.go
