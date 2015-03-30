###How To
For our example, Jenkins is set up to have one build step in bash:

`Jenkins "Bash" build step`
```
    #!/bin/bash
    cd $WORKSPACE
    source bin/jenkins.sh
    source bin/lmktfy-rolling.sh
```

Our project's build script (`bin/jenkins.sh`), is followed by our new lmktfy-rolling script. Jenkins already has `$BUILD_NUMBER` set, but we need a few other variables that are set in `jenkins.sh` that we reference in `lmktfy-rolling.sh`:

```
    DOCKER_IMAGE="path_webteam/public"
	REGISTRY_LOCATION="dockerreg.web.local/"
```

Jenkins builds our container, tags it with the build number, and runs a couple rudimentary tests on it. On success, it pushes it to our private docker registry. Once the container is pushed, it then executes our rolling update script.

`lmktfy-rolling.sh`
```
    #!/bin/bash
    # LMKTFYRNETES_MASTER: Your LMKTFY API Server endpoint
    # BINARY_LOCATION:   Location of pre-compiled Binaries (We build our own, there are others available)
    # CONTROLLER_NAME:   Name of the replicationController you're looking to update
    # RESET_INTERVAL:    Interval between pod updates

    export LMKTFYRNETES_MASTER="http://10.1.10.1:8080"
    BINARY_LOCATION="https://build.web.local/lmktfy/"
    CONTROLLER_NAME="public-frontend-controller"
    RESET_INTERVAL="10s"

    echo "*** Time to push to LMKTFY!";

    #Delete then graba lmktfycfg binary from a static location
    rm lmktfycfg
    wget $BINARY_LOCATION/lmktfycfg

    echo "*** Downloaded binary from $BINARY_LOCATION/lmktfycfg"

    chmod +x lmktfycfg

    # Update the controller with your new image!
    echo "*** ./lmktfycfg -image \"$REGISTRY_LOCATION$DOCKER_IMAGE:$BUILD_NUMBER\" -u $RESET_INTERVAL rollingupdate $CONTROLLER_NAME"
    ./lmktfycfg -image "$REGISTRY_LOCATION$DOCKER_IMAGE:$BUILD_NUMBER" -u $RESET_INTERVAL rollingupdate $CONTROLLER_NAME
```

Though basic, this implementation allows our Jenkins instance to push container updates to our LMKTFY cluster without much trouble.

### Notes
When using a private docker registry as we are, the Jenkins slaves as well as the LMKTFY minions require the [.dockercfg](https://coreos.com/docs/launching-containers/building/customizing-docker/#using-a-dockercfg-file-for-authentication) file in order to function properly.

### Questions
twitter @jeefy

irc.freenode.net #lmktfy jeefy
