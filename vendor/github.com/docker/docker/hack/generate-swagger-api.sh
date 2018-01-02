#!/bin/sh
set -eu

swagger generate model -f api/swagger.yaml \
    -t api -m types --skip-validator -C api/swagger-gen.yaml \
    -n ErrorResponse \
    -n GraphDriverData \
    -n IdResponse \
    -n ImageDeleteResponseItem \
    -n ImageSummary \
    -n Plugin -n PluginDevice -n PluginMount -n PluginEnv -n PluginInterfaceType \
    -n Port \
    -n ServiceUpdateResponse \
    -n Volume

swagger generate operation -f api/swagger.yaml \
    -t api -a types -m types -C api/swagger-gen.yaml \
    -T api/templates --skip-responses --skip-parameters --skip-validator \
    -n Authenticate \
    -n ContainerChanges \
    -n ContainerCreate \
    -n ContainerTop \
    -n ContainerUpdate \
    -n ContainerWait \
    -n ImageHistory \
    -n VolumesCreate \
    -n VolumesList
